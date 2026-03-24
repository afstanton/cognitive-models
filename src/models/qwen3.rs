// Qwen3 dense model implementation for cognitive-models.
//
// Covers all dense Qwen3 sizes (0.6B, 1.7B, 4B, 8B, 14B, 32B) and the
// Qwen3-Embedding variants (0.6B, 4B, 8B), which share the same architecture.
// MoE variants (30B-A3B, 235B-A22B) are not supported here.
//
// Key architectural differences from Phi-3:
//   - Separate q/k/v projections (not fused QKV)
//   - QKNorm: RMSNorm applied to Q and K after projection, before RoPE
//   - head_dim is explicit in config
//   - Sliding window attention for layers beyond max_window_layers
//   - SwiGLU MLP with separate gate/up/down projections

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_no_bias as linear};
use candle_transformers::utils;
use std::sync::Arc;

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    /// Number of layers that use FULL attention.
    /// Layers at index >= max_window_layers use sliding window attention.
    /// For Qwen3-8B this equals num_hidden_layers (all full), but smaller
    /// models may have fewer full-attention layers.
    #[serde(default = "Config::default_max_window_layers")]
    pub max_window_layers: usize,
    #[serde(default = "Config::default_sliding_window")]
    pub sliding_window: usize,
}

impl Config {
    fn default_max_window_layers() -> usize {
        // Safe fallback: all layers use full attention.
        usize::MAX
    }

    fn default_sliding_window() -> usize {
        4096
    }

    pub fn uses_sliding_window(&self, layer_idx: usize) -> bool {
        layer_idx >= self.max_window_layers
    }
}

impl crate::model::ModelConfig for Config {
    fn vector_size(&self) -> usize {
        self.hidden_size
    }
}

// ============================================================================
// RoPE
// ============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
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

    fn apply_rotary_emb_qkv(
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

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// QKNorm: applied to Q after projection, before RoPE.
    q_norm: RmsNorm,
    /// QKNorm: applied to K after projection, before RoPE.
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_sliding_window: bool,
    sliding_window: usize,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden = cfg.hidden_size;

        let q_proj = linear(hidden, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden, vb.pp("o_proj"))?;

        // QKNorm — Qwen3 normalises Q and K with per-head RMSNorm.
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
            use_sliding_window: cfg.uses_sliding_window(layer_idx),
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Project Q, K, V separately.
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to [batch, heads, seq, head_dim].
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QKNorm: normalise each head independently.
        // Shape is [batch, heads, seq, head_dim] — norm acts on last dim.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE.
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?;

        // KV cache.
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };

        // Apply sliding window if this layer uses it.
        let (k, v) = if self.use_sliding_window && k.dim(2)? > self.sliding_window {
            let kv_len = k.dim(2)?;
            let start = kv_len - self.sliding_window;
            (
                k.narrow(2, start, self.sliding_window)?,
                v.narrow(2, start, self.sliding_window)?,
            )
        } else {
            (k, v)
        };

        self.kv_cache = Some((k.clone(), v.clone()));

        let k = utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ============================================================================
// MLP (SwiGLU with separate gate/up/down projections)
// ============================================================================

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear(h, i, vb.pp("gate_proj"))?,
            up_proj: linear(h, i, vb.pp("up_proj"))?,
            down_proj: linear(i, h, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: down(silu(gate) * up)
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ============================================================================
// Decoder layer
// ============================================================================

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, layer_idx, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
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
        let xs = self
            .post_attention_layernorm
            .forward(&xs)?
            .apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ============================================================================
// Model
// ============================================================================

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
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?);
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
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let zeros = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&zeros, &mask], candle_core::D::Minus1)?
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
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    /// Forward pass with cognitive soft prompt prepended to token embeddings.
    /// Returns logits for the last position. KV cache is cleared unconditionally.
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
        let (_, soft_len, _) = soft_prompt.dims3()?;
        let total_len = soft_len + seq_len;
        let attention_mask = if total_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, total_len, seqlen_offset)?)
        };
        let xs_tokens = self.embed_tokens.forward(input_ids)?;
        let mut xs = Tensor::cat(&[soft_prompt, &xs_tokens], 1)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.narrow(1, total_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    /// Returns full hidden states [batch, seq, hidden_size] — use for embedding.
    pub fn forward_hidden(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn embed_sequence(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
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

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        Model::forward(self, input_ids, seqlen_offset)
    }

    fn forward_with_soft_prompt(
        &mut self,
        soft_prompt: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        Model::forward_with_soft_prompt(self, soft_prompt, input_ids, 0)
    }

    fn forward_hidden(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        Model::forward_hidden(self, input_ids, seqlen_offset)
    }

    fn embed_sequence(&self, input_ids: &Tensor) -> Result<Tensor> {
        Model::embed_sequence(self, input_ids)
    }

    fn clear_kv_cache(&mut self) {
        Model::clear_kv_cache(self)
    }
}
