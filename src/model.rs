//! The `CognitiveLM` trait: a model-agnostic interface for language model operations.
//!
//! Any language model used for generation and embedding must implement this trait.
//! This enables swapping the underlying checkpoint (e.g. Phi-3 → Qwen3) without
//! touching `LanguageEngine`, `TranslationOracle`, or any higher-level code.
//!
//! # Injection contract
//!
//! The core operation Familiar depends on is *soft prompt injection*: a stack of
//! cognitive state vectors (drives, valence, focus, etc.) is prepended to the real
//! token embeddings before the first transformer block. Implementations must honour
//! this contract:
//!
//! - `forward_with_soft_prompt` concatenates `[soft_prompt | embed(input_ids)]` along
//!   the sequence axis and passes the combined tensor through all transformer layers.
//! - Position IDs (RoPE offsets) must cover the combined sequence: soft tokens occupy
//!   positions `0..n_soft`, real tokens occupy `n_soft..n_soft+seq_len`.
//! - The attention mask must allow real tokens to attend to all soft prefix tokens.
//! - The KV cache should be cleared at the start of each `forward_with_soft_prompt`
//!   call so that calls are self-contained.

use candle_core::{DType, Result, Tensor};

/// A language model capable of autoregressive generation and soft prompt injection.
///
/// Implementors must be `Send` so they can live inside a `tokio::sync::Mutex`.
pub trait CognitiveLM: Send {
    /// The hidden dimension of this model.
    ///
    /// This is the expected size of each vector in a soft prompt and the output
    /// dimension of `forward_hidden` / `embed_sequence`.
    fn hidden_dim(&self) -> usize;

    /// The DType this model was built with (e.g. `DType::F16` or `DType::F32`).
    ///
    /// Callers should cast soft prompt tensors to this dtype before passing them
    /// to `forward_with_soft_prompt`.
    fn dtype(&self) -> DType;

    /// Standard autoregressive forward pass returning logits for the last token.
    ///
    /// Output shape: `[batch, 1, vocab_size]` (or `[batch, seq_len, vocab_size]`
    /// for the prefill pass).
    ///
    /// `seqlen_offset` is the number of tokens **already in the KV cache**.
    /// Pass `0` for the first call on a fresh sequence; increment by the number
    /// of newly processed tokens on every subsequent call.
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Forward pass with a cognitive soft prompt prepended to token embeddings.
    ///
    /// `soft_prompt` shape: `[batch, n_soft, hidden_dim]`
    /// `input_ids` shape:   `[batch, seq_len]`
    ///
    /// The implementation must:
    /// 1. Clear the KV cache, making this call fully self-contained.
    /// 2. Embed `input_ids` → `[batch, seq_len, hidden_dim]`.
    /// 3. Concatenate `[soft_prompt | token_embeddings]` on the sequence axis.
    /// 4. Run the combined tensor through all transformer layers.
    /// 5. Return logits for the **last** position (the final real token).
    ///
    /// The `seqlen_offset` parameter is intentionally absent: because the KV
    /// cache is cleared at the start of each call, the offset is always 0.
    /// Removing it from the signature makes it impossible for callers to
    /// accidentally pass a non-zero value and silently reintroduce the
    /// RoPE/KV desync bug.
    fn forward_with_soft_prompt(
        &mut self,
        soft_prompt: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor>;

    /// Forward pass returning full hidden states (for encoding / embedding).
    ///
    /// Output shape: `[batch, seq_len, hidden_dim]`.
    ///
    /// `seqlen_offset` follows the same convention as `forward`.
    fn forward_hidden(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Project token IDs into the model's embedding space without running
    /// any transformer layers.
    ///
    /// Output shape: `[batch, seq_len, hidden_dim]`.
    ///
    /// Useful for constructing soft prompts from token sequences.
    fn embed_sequence(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// Clear the KV cache.
    ///
    /// Must be called before each new independent sequence to prevent attention
    /// from accidentally spanning unrelated context.
    fn clear_kv_cache(&mut self);
}

// ---------------------------------------------------------------------------
// Test support: a minimal mock CognitiveLM for unit testing without weights.
//
// Gated behind `#[cfg(any(test, feature = "test-support"))]` so it is:
//   - always available within this crate's own tests, and
//   - available to downstream crates (like Familiar) that add
//     cognitive-models with the "test-support" feature in [dev-dependencies].
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    //! A minimal `CognitiveLM` mock for unit-testing without model weights.
    //!
    //! `RecordingModel` records every call to `forward` and
    //! `forward_with_soft_prompt`, capturing the `seqlen_offset` argument so
    //! tests can assert that the caller computed offsets correctly.

    use super::CognitiveLM;
    use candle_core::{DType, Device, Result, Tensor};
    use std::sync::{Arc, Mutex};

    /// One recorded call to `forward` or `forward_with_soft_prompt`.
    #[derive(Debug, Clone)]
    pub struct ForwardCall {
        pub kind: CallKind,
        pub seqlen_offset: usize,
        /// Sequence length of the input_ids tensor (not including soft prefix).
        pub seq_len: usize,
        /// Soft prefix length (0 for plain `forward` calls).
        pub soft_len: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CallKind {
        Forward,
        ForwardWithSoftPrompt,
        ForwardHidden,
    }

    /// A tiny `CognitiveLM` that always predicts token 0 and records every call.
    ///
    /// `VOCAB_SIZE = 4` is large enough to satisfy softmax without numerical
    /// issues, and small enough that the logits tensor is negligible.
    pub struct RecordingModel {
        pub hidden: usize,
        pub calls: Arc<Mutex<Vec<ForwardCall>>>,
        device: Device,
    }

    impl RecordingModel {
        pub const VOCAB_SIZE: usize = 4;

        pub fn new(hidden: usize) -> Self {
            Self {
                hidden,
                calls: Arc::new(Mutex::new(Vec::new())),
                device: Device::Cpu,
            }
        }

        /// Drain and return all recorded calls, leaving the log empty.
        #[allow(dead_code)]
        pub fn drain_calls(&self) -> Vec<ForwardCall> {
            self.calls.lock().unwrap().drain(..).collect()
        }

        /// Return logits shaped [1, seq, vocab] with token 0 heavily favoured.
        fn make_logits(&self, seq: usize) -> Result<Tensor> {
            let row: Vec<f32> = (0..Self::VOCAB_SIZE)
                .map(|i| if i == 0 { 10.0 } else { 0.0 })
                .collect();
            let flat: Vec<f32> = row.repeat(seq);
            Tensor::from_vec(flat, (1, seq, Self::VOCAB_SIZE), &self.device)
        }
    }

    impl CognitiveLM for RecordingModel {
        fn hidden_dim(&self) -> usize {
            self.hidden
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
            let seq_len = input_ids.dim(1)?;
            self.calls.lock().unwrap().push(ForwardCall {
                kind: CallKind::Forward,
                seqlen_offset,
                seq_len,
                soft_len: 0,
            });
            self.make_logits(seq_len)
        }

        fn forward_with_soft_prompt(
            &mut self,
            soft_prompt: &Tensor,
            input_ids: &Tensor,
        ) -> Result<Tensor> {
            let seq_len = input_ids.dim(1)?;
            let (_, soft_len, _) = soft_prompt.dims3()?;
            self.calls.lock().unwrap().push(ForwardCall {
                kind: CallKind::ForwardWithSoftPrompt,
                seqlen_offset: 0,
                seq_len,
                soft_len,
            });
            self.make_logits(1)
        }

        fn forward_hidden(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
            let seq_len = input_ids.dim(1)?;
            self.calls.lock().unwrap().push(ForwardCall {
                kind: CallKind::ForwardHidden,
                seqlen_offset,
                seq_len,
                soft_len: 0,
            });
            Tensor::zeros((1, seq_len, self.hidden), DType::F32, &self.device)
        }

        fn embed_sequence(&self, input_ids: &Tensor) -> Result<Tensor> {
            let seq_len = input_ids.dim(1)?;
            Tensor::zeros((1, seq_len, self.hidden), DType::F32, &self.device)
        }

        fn clear_kv_cache(&mut self) {}
    }
}
