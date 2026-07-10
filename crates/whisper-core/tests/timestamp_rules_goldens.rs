//! L2: ApplyTimestampRules parity against crafted logit fixtures.

mod common;
use candle_core::Tensor;
use common::{fixtures_dir, load_json};
use std::collections::HashMap;
use whisper_core::decode::{apply_timestamp_rules, TimestampRules};
use whisper_core::tokenizer::{get_tokenizer, Task};

#[test]
fn timestamp_rules_match_python() {
    let meta = load_json("timestamp_rules_meta.json");
    let sample_begin = meta["sample_begin"].as_u64().unwrap() as usize;
    let max_initial = meta["max_initial_timestamp_index"].as_u64().unwrap() as usize;

    let tok = get_tokenizer(true, 100, Some("en"), Some(Task::Transcribe)).unwrap();
    let rules = TimestampRules {
        no_timestamps: tok.no_timestamps,
        timestamp_begin: tok.timestamp_begin,
        eot: tok.eot,
        sample_begin,
        max_initial_timestamp_index: Some(max_initial),
    };

    let npz = Tensor::read_npz(fixtures_dir().join("timestamp_rules_goldens.npz")).unwrap();
    let tensors: HashMap<String, Tensor> = npz.into_iter().collect();

    for (idx, case) in meta["cases"].as_object().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ctx: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let mut logits: Vec<f32> = tensors[&format!("logits_in_{idx}")]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expected: Vec<f32> = tensors[&format!("logits_out_{idx}")]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        apply_timestamp_rules(&mut logits, &ctx, &rules);

        let mut mismatches = 0;
        for (i, (a, e)) in logits.iter().zip(&expected).enumerate() {
            let ok = if e.is_infinite() {
                a.is_infinite() && a.is_sign_negative()
            } else {
                (a - e).abs() < 1e-6
            };
            if !ok {
                if mismatches < 5 {
                    eprintln!("case {name}: index {i}: got {a}, expected {e}");
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "case {name}: {mismatches} mismatched logits");
    }
}
