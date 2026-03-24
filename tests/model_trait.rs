//! Integration tests for the CognitiveLM trait and RecordingModel mock.
//!
//! These tests verify the trait contract without requiring model weights.
//! They run against `RecordingModel` which is gated behind the `test-support`
//! feature — this file requires that feature to be active, which Cargo
//! automatically enables for `cargo test` within this crate.

use candle_core::{DType, Device, Tensor};
use cognitive_models::model::test_support::{CallKind, RecordingModel};
use cognitive_models::CognitiveLM;

// -----------------------------------------------------------------------
// RecordingModel trait contract
// -----------------------------------------------------------------------

#[test]
fn recording_model_hidden_dim_matches_constructor() {
    let m = RecordingModel::new(128);
    assert_eq!(m.hidden_dim(), 128);
}

#[test]
fn recording_model_dtype_is_f32() {
    let m = RecordingModel::new(16);
    assert_eq!(m.dtype(), DType::F32);
}

#[test]
fn forward_returns_correct_shape() {
    let mut m = RecordingModel::new(16);
    let ids = Tensor::zeros((1, 3usize), DType::I64, &Device::Cpu).unwrap();
    let logits = m.forward(&ids, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 3, RecordingModel::VOCAB_SIZE]);
}

#[test]
fn forward_with_soft_prompt_returns_last_position_only() {
    let mut m = RecordingModel::new(16);
    let soft = Tensor::zeros((1, 2usize, 16usize), DType::F32, &Device::Cpu).unwrap();
    let ids = Tensor::zeros((1, 3usize), DType::I64, &Device::Cpu).unwrap();
    let logits = m.forward_with_soft_prompt(&soft, &ids).unwrap();
    // Trait contract: returns last position only → [batch=1, 1, vocab]
    assert_eq!(logits.dims(), &[1, 1, RecordingModel::VOCAB_SIZE]);
}

#[test]
fn forward_hidden_returns_full_sequence_shape() {
    let mut m = RecordingModel::new(16);
    let ids = Tensor::zeros((1, 4usize), DType::I64, &Device::Cpu).unwrap();
    let hidden = m.forward_hidden(&ids, 0).unwrap();
    assert_eq!(hidden.dims(), &[1, 4, 16]);
}

#[test]
fn embed_sequence_returns_correct_shape() {
    let m = RecordingModel::new(32);
    let ids = Tensor::zeros((1, 5usize), DType::I64, &Device::Cpu).unwrap();
    let embedded = m.embed_sequence(&ids).unwrap();
    assert_eq!(embedded.dims(), &[1, 5, 32]);
}

#[test]
fn forward_records_seqlen_offset() {
    let mut m = RecordingModel::new(16);
    let ids = Tensor::zeros((1, 2usize), DType::I64, &Device::Cpu).unwrap();
    m.forward(&ids, 7).unwrap();
    let calls = m.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].seqlen_offset, 7);
    assert_eq!(calls[0].seq_len, 2);
    assert_eq!(calls[0].kind, CallKind::Forward);
}

#[test]
fn forward_with_soft_prompt_records_soft_len() {
    let mut m = RecordingModel::new(16);
    let soft = Tensor::zeros((1, 3usize, 16usize), DType::F32, &Device::Cpu).unwrap();
    let ids = Tensor::zeros((1, 2usize), DType::I64, &Device::Cpu).unwrap();
    m.forward_with_soft_prompt(&soft, &ids).unwrap();
    let calls = m.calls.lock().unwrap();
    assert_eq!(calls[0].kind, CallKind::ForwardWithSoftPrompt);
    assert_eq!(calls[0].soft_len, 3);
    assert_eq!(calls[0].seq_len, 2);
    // seqlen_offset is always 0 by trait contract
    assert_eq!(calls[0].seqlen_offset, 0);
}

#[test]
fn clear_kv_cache_does_not_panic() {
    let mut m = RecordingModel::new(16);
    // Should be a no-op — just verify it doesn't panic
    m.clear_kv_cache();
}

#[test]
fn drain_calls_empties_log() {
    let mut m = RecordingModel::new(16);
    let ids = Tensor::zeros((1, 1usize), DType::I64, &Device::Cpu).unwrap();
    m.forward(&ids, 0).unwrap();
    m.forward(&ids, 1).unwrap();
    let drained = m.drain_calls();
    assert_eq!(drained.len(), 2);
    assert!(m.calls.lock().unwrap().is_empty());
}
