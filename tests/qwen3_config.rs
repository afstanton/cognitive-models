//! Tests for the Qwen3 model config and architecture.
//!
//! These tests do not load model weights — they verify that:
//!   - Config deserialises correctly from JSON matching real checkpoints
//!   - vector_size() returns the correct hidden_size for each model variant
//!   - Sliding window layer logic is correct

use cognitive_models::ModelConfig;
use cognitive_models::models::qwen3::Config;

// -----------------------------------------------------------------------
// Config deserialization and vector_size()
// -----------------------------------------------------------------------

fn make_config(
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_window_layers: Option<usize>,
) -> Config {
    let max_wl = max_window_layers.unwrap_or(num_hidden_layers);
    let json = format!(
        r#"{{
            "vocab_size": 151936,
            "hidden_size": {hidden_size},
            "intermediate_size": 8192,
            "num_hidden_layers": {num_hidden_layers},
            "num_attention_heads": {num_attention_heads},
            "num_key_value_heads": {num_key_value_heads},
            "head_dim": {head_dim},
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "max_window_layers": {max_wl},
            "sliding_window": 4096
        }}"#
    );
    serde_json::from_str(&json).expect("config deserialization failed")
}

#[test]
fn qwen3_0_6b_vector_size() {
    let cfg = make_config(1024, 28, 16, 8, 64, None);
    assert_eq!(cfg.vector_size(), 1024);
}

#[test]
fn qwen3_4b_vector_size() {
    let cfg = make_config(2560, 36, 32, 8, 80, None);
    assert_eq!(cfg.vector_size(), 2560);
}

#[test]
fn qwen3_8b_vector_size() {
    let cfg = make_config(4096, 36, 32, 8, 128, Some(36));
    assert_eq!(cfg.vector_size(), 4096);
}

#[test]
fn qwen3_14b_vector_size() {
    let cfg = make_config(5120, 40, 40, 8, 128, Some(40));
    assert_eq!(cfg.vector_size(), 5120);
}

#[test]
fn qwen3_32b_vector_size() {
    let cfg = make_config(5120, 64, 64, 8, 128, Some(64));
    assert_eq!(cfg.vector_size(), 5120);
}

#[test]
fn qwen3_special_tokens_come_from_config() {
    let cfg = make_config(2560, 36, 32, 8, 80, None);
    assert_eq!(cfg.bos_token_id(), Some(151643));
    assert_eq!(cfg.eos_token_id(), Some(151645));
    assert_eq!(cfg.default_stop_tokens(), vec![151645]);
}

#[test]
fn qwen3_tied_word_embeddings_deserializes_from_config() {
    let cfg: Config = serde_json::from_str(
        r#"{
          "vocab_size": 151936,
          "hidden_size": 2560,
          "intermediate_size": 9728,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "rms_norm_eps": 1e-06,
          "rope_theta": 1000000,
          "max_position_embeddings": 40960,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "tie_word_embeddings": true
        }"#,
    )
    .expect("config deserialization failed");

    assert!(cfg.tie_word_embeddings);
}

// -----------------------------------------------------------------------
// Sliding window layer logic
// -----------------------------------------------------------------------

#[test]
fn qwen3_8b_all_layers_use_full_attention() {
    // For 8B: max_window_layers == num_hidden_layers == 36
    let cfg = make_config(4096, 36, 32, 8, 128, Some(36));
    for i in 0..36 {
        assert!(
            !cfg.uses_sliding_window(i),
            "layer {i} should use full attention for Qwen3-8B"
        );
    }
}

#[test]
fn sliding_window_applies_beyond_max_window_layers() {
    // A hypothetical config where only the first 4 layers use full attention.
    let cfg = make_config(1024, 8, 16, 8, 64, Some(4));
    for i in 0..4 {
        assert!(
            !cfg.uses_sliding_window(i),
            "layer {i} should be full attention"
        );
    }
    for i in 4..8 {
        assert!(
            cfg.uses_sliding_window(i),
            "layer {i} should use sliding window"
        );
    }
}

#[test]
fn default_max_window_layers_means_all_full_attention() {
    // Config without max_window_layers — defaults to all full attention.
    let json = r#"{
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960
    }"#;
    let cfg: Config = serde_json::from_str(json).unwrap();
    for i in 0..28 {
        assert!(!cfg.uses_sliding_window(i));
    }
}

#[test]
fn vector_size_equals_hidden_size() {
    let cfg = make_config(4096, 36, 32, 8, 128, Some(36));
    // vector_size() is a convenience alias — must always equal hidden_size.
    assert_eq!(cfg.vector_size(), cfg.hidden_size);
}
