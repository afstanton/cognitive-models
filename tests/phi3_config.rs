//! Tests for Phi-3 config and ModelConfig trait implementation.

use cognitive_models::ModelConfig;
use cognitive_models::models::phi3::Config;

fn make_phi3_config(hidden_size: usize) -> Config {
    let json = format!(
        r#"{{
            "vocab_size": 32064,
            "hidden_act": "silu",
            "hidden_size": {hidden_size},
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096
        }}"#
    );
    serde_json::from_str(&json).expect("config deserialization failed")
}

#[test]
fn phi3_mini_vector_size() {
    // Phi-3-mini-4k-instruct: hidden_size = 3072
    let cfg = make_phi3_config(3072);
    assert_eq!(cfg.vector_size(), 3072);
}

#[test]
fn phi3_medium_vector_size() {
    // Phi-3-medium: hidden_size = 5120
    let cfg = make_phi3_config(5120);
    assert_eq!(cfg.vector_size(), 5120);
}

#[test]
fn phi3_vector_size_equals_hidden_size() {
    let cfg = make_phi3_config(3072);
    assert_eq!(cfg.vector_size(), cfg.hidden_size);
}

#[test]
fn phi3_head_dim_derived_correctly() {
    // head_dim = hidden_size / num_attention_heads = 3072 / 32 = 96
    let cfg = make_phi3_config(3072);
    assert_eq!(cfg.head_dim(), 96);
}
