//! L0: tokenizer parity with tokenizer_goldens.json (exact equality).

mod common;
use common::load_json;
use whisper_core::tokenizer::{get_tokenizer, Task};

#[test]
fn tokenizer_matches_python_goldens() {
    let g = load_json("tokenizer_goldens.json");
    let num_languages = g["num_languages"].as_u64().unwrap() as usize;
    let tok = get_tokenizer(true, num_languages, Some("en"), Some(Task::Transcribe)).unwrap();

    assert_eq!(tok.eot as u64, g["eot"].as_u64().unwrap());
    assert_eq!(tok.sot as u64, g["sot"].as_u64().unwrap());
    assert_eq!(tok.sot_prev as u64, g["sot_prev"].as_u64().unwrap());
    assert_eq!(tok.sot_lm as u64, g["sot_lm"].as_u64().unwrap());
    assert_eq!(tok.no_speech as u64, g["no_speech"].as_u64().unwrap());
    assert_eq!(
        tok.no_timestamps as u64,
        g["no_timestamps"].as_u64().unwrap()
    );
    assert_eq!(
        tok.timestamp_begin as u64,
        g["timestamp_begin"].as_u64().unwrap()
    );
    assert_eq!(tok.transcribe as u64, g["transcribe"].as_u64().unwrap());
    assert_eq!(tok.translate as u64, g["translate"].as_u64().unwrap());
    assert_eq!(
        tok.language_token().unwrap() as u64,
        g["language_token_en"].as_u64().unwrap()
    );

    let expect_u32 = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };

    assert_eq!(
        tok.sot_sequence,
        expect_u32(&g["sot_sequence_en_transcribe"])
    );
    assert_eq!(
        tok.sot_sequence_including_notimestamps(),
        expect_u32(&g["sot_sequence_including_notimestamps"])
    );
    // Python iterates a set() to build these, so its *order* is hash-arbitrary;
    // compare the token<->code correspondence instead.
    let fixture_tokens = expect_u32(&g["all_language_tokens"]);
    let fixture_codes: Vec<String> = g["all_language_codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut expected_pairs: Vec<(u32, String)> =
        fixture_tokens.iter().copied().zip(fixture_codes).collect();
    expected_pairs.sort();
    let mut actual_pairs: Vec<(u32, String)> = tok
        .all_language_tokens()
        .into_iter()
        .zip(tok.all_language_codes().into_iter().map(str::to_string))
        .collect();
    actual_pairs.sort();
    assert_eq!(actual_pairs, expected_pairs, "language token/code mapping");

    assert_eq!(
        tok.non_speech_tokens(),
        expect_u32(&g["non_speech_tokens"]),
        "non_speech_tokens"
    );

    for case in g["encode_cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected = expect_u32(&case["tokens"]);
        assert_eq!(tok.encode(text), expected, "encode({text:?})");
        // decode round-trips for plain text
        assert_eq!(
            tok.decode(&expected),
            text,
            "decode round-trip for {text:?}"
        );
    }

    // word splitting
    let split = &g["split_to_word_tokens"];
    let input = expect_u32(&split["input_tokens"]);
    let expected_words: Vec<String> = split["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let expected_tokens: Vec<Vec<u32>> = split["word_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(expect_u32)
        .collect();
    let (words, word_tokens) = tok.split_to_word_tokens(&input);
    assert_eq!(words, expected_words);
    assert_eq!(word_tokens, expected_tokens);

    // full special-token table
    for (name, id) in g["special_tokens"].as_object().unwrap() {
        assert_eq!(
            tok.special_token(name),
            Some(id.as_u64().unwrap() as u32),
            "special token {name}"
        );
    }
}
