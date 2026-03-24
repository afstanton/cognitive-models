// Vendored from candle-transformers `models/phi3.rs`, with an added
// `forward_hidden` that returns the full sequence hidden states before lm_head.
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)] // Vendored model - keeping full API available

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_no_bias as linear};
use candle_transformers::utils;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl crate::model::ModelConfig for Config {
    fn vector_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    /// Forward pass with a cognitive soft prompt prepended to token embeddings.
    /// Returns logits for the last (most recent) token.
    ///
    /// `seqlen_offset` must be 0: the KV cache is cleared unconditionally.
    pub fn forward_with_soft_prompt(
        &mut self,
        soft_prompt: &Tensor,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        if seqlen_offset != 0 {
            candle_core::bail!(
                "forward_with_soft_prompt clears the KV cache unconditionally; \
                 seqlen_offset must be 0, got {seqlen_offset}"
            );
        }
        self.clear_kv_cache();
        let (b_size, seq_len) = input_ids.dims2()?;
        let (_b, soft_len, _h) = soft_prompt.dims3()?;
        let total_len = soft_len + seq_len;
        let attention_mask = if total_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, total_len, seqlen_offset)?;
            Some(mask)
        };
        let xs_tokens = self.embed_tokens.forward(input_ids)?;
        let xs = Tensor::cat(&[soft_prompt, &xs_tokens], 1)?;
        let mut xs = xs;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, total_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    pub fn forward_hidden(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.apply(&self.norm)
    }

    pub fn embed_sequence(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

// ============================================================================
// CognitiveLM implementation
// ============================================================================

impl crate::model::CognitiveLM for Model {
    fn hidden_dim(&self) -> usize {
        self.hidden_size
    }

    fn dtype(&self) -> candle_core::DType {
        self.dtype
    }

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle_core::Result<Tensor> {
        Model::forward(self, input_ids, seqlen_offset)
    }

    fn forward_with_soft_prompt(
        &mut self,
        soft_prompt: &Tensor,
        input_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        Model::forward_with_soft_prompt(self, soft_prompt, input_ids, 0)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        Model::forward_hidden(self, input_ids, seqlen_offset)
    }

    fn embed_sequence(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        Model::embed_sequence(self, input_ids)
    }

    fn clear_kv_cache(&mut self) {
        Model::clear_kv_cache(self)
    }
}
