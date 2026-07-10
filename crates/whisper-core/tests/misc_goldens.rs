//! L0: compression ratio + timestamp formatting parity.

mod common;
use common::load_json;
use whisper_core::utils::{compression_ratio, format_timestamp};

#[test]
fn compression_ratio_matches_zlib() {
    // zlib-rs compresses within a byte or two of C zlib at level 6, but not
    // identically. The ratio only feeds the `> 2.4` repetitiveness check in
    // the transcribe fallback, so 5% relative agreement preserves behavior.
    let g = load_json("misc_goldens.json");
    for case in g["compression_ratio"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected = case["ratio"].as_f64().unwrap();
        let actual = compression_ratio(text);
        assert!(
            (actual - expected).abs() / expected < 0.05,
            "compression_ratio({text:?}) = {actual}, expected {expected}"
        );
    }
}

#[test]
fn format_timestamp_matches_python() {
    let g = load_json("misc_goldens.json");
    for case in g["format_timestamp"].as_array().unwrap() {
        let seconds = case["seconds"].as_f64().unwrap();
        let expected = case["formatted"].as_str().unwrap();
        assert_eq!(format_timestamp(seconds, false, "."), expected, "format({seconds})");
    }
}
