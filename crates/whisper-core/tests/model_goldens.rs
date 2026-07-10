//! L1/L3: model parity tests. These download whisper-tiny (~150 MB) from the
//! HF Hub on first run, so they are #[ignore]d by default:
//!
//!     cargo test -p whisper-candle-core --test model_goldens -- --ignored

mod common;
use candle_core::{Device, Tensor};
use common::{cosine_similarity, fixtures_dir, load_json, mean_abs_diff};
use whisper_core::audio::{log_mel_spectrogram, N_FRAMES, N_SAMPLES};
use whisper_core::tokenizer::get_tokenizer;
use whisper_core::{DecodingOptions, Task, TranscribeOptions, WhisperModel};

fn load_tiny() -> WhisperModel {
    let files = whisper_core::fetch_model(whisper_core::WhichModel::Tiny).unwrap();
    WhisperModel::load(&files.config, &files.weights, &Device::Cpu).unwrap()
}

fn jfk_mel_window(model: &WhisperModel) -> Tensor {
    let pcm = Tensor::read_npy(fixtures_dir().join("audio_jfk_pcm.npy"))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mel = log_mel_spectrogram(&pcm, model.config.num_mel_bins, N_SAMPLES).unwrap();
    let window = mel.window(0, N_FRAMES, N_FRAMES);
    Tensor::from_vec(window, (1, model.config.num_mel_bins, N_FRAMES), &Device::Cpu).unwrap()
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn encoder_output_matches_pytorch() {
    let mut model = load_tiny();
    let mel = jfk_mel_window(&model);
    let features = model.encoder_forward(&mel, true).unwrap();

    let expected = Tensor::read_npy(fixtures_dir().join("encoder_out_tiny.npy")).unwrap();
    assert_eq!(features.dims(), expected.dims());

    let a = features.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mad = mean_abs_diff(&a, &b);
    let cos = cosine_similarity(&a, &b);
    eprintln!("encoder: mean_abs_diff={mad:.2e} cosine={cos:.6}");
    assert!(mad <= 5e-3, "encoder mean abs diff {mad}");
    assert!(cos >= 0.9995, "encoder cosine similarity {cos}");
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn sot_logits_match_pytorch() {
    let g = load_json("decode_goldens_tiny.json");
    let sot_sequence: Vec<u32> = g["sot_sequence"]
        .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let mut model = load_tiny();
    let mel = jfk_mel_window(&model);
    let features = model.encoder_forward(&mel, true).unwrap();
    let tokens = Tensor::from_vec(sot_sequence.clone(), (1, sot_sequence.len()), &Device::Cpu).unwrap();
    let hidden = model.decoder_forward(&tokens, &features, true).unwrap();
    let logits = model.decoder_final_linear(&hidden).unwrap();

    let expected = Tensor::read_npy(fixtures_dir().join("logits_sot_tiny.npy")).unwrap();
    assert_eq!(logits.dims(), expected.dims());

    let a = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mad = mean_abs_diff(&a, &b);
    let cos = cosine_similarity(&a, &b);
    eprintln!("logits: mean_abs_diff={mad:.2e} cosine={cos:.6}");
    assert!(mad <= 5e-3, "logits mean abs diff {mad}");
    assert!(cos >= 0.9995, "logits cosine similarity {cos}");
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn language_detection_matches() {
    let g = load_json("decode_goldens_tiny.json");
    let expected_lang = g["detect_language_top10"][0][0].as_str().unwrap();
    let expected_prob = g["detect_language_top10"][0][1].as_f64().unwrap();

    let mut model = load_tiny();
    let tok = get_tokenizer(true, model.num_languages(), Some("en"), Some(Task::Transcribe)).unwrap();
    let mel = jfk_mel_window(&model);
    let features = model.encoder_forward(&mel, true).unwrap();
    let (lang, probs) = whisper_core::detect_language(&mut model, &tok, &features).unwrap();

    eprintln!("detected {lang} p={} (python: {expected_lang} p={expected_prob:.4})", probs[&lang]);
    assert_eq!(lang, expected_lang);
    assert!((probs[&lang] as f64 - expected_prob).abs() < 0.02);
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn greedy_decode_matches_pytorch() {
    let g = load_json("decode_goldens_tiny.json");
    let expected_tokens: Vec<u32> = g["greedy"]["tokens"]
        .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let expected_text = g["greedy"]["text"].as_str().unwrap();
    let expected_avg_logprob = g["greedy"]["avg_logprob"].as_f64().unwrap();
    let expected_no_speech = g["greedy"]["no_speech_prob"].as_f64().unwrap();

    let mut model = load_tiny();
    let tok = get_tokenizer(true, model.num_languages(), Some("en"), Some(Task::Transcribe)).unwrap();
    let mel = jfk_mel_window(&model);
    let options = DecodingOptions {
        language: Some("en".to_string()),
        ..Default::default()
    };
    let result = whisper_core::decode(&mut model, &tok, &mel, options).unwrap();

    eprintln!("text: {}", result.text);
    eprintln!(
        "avg_logprob: {:.4} (python {expected_avg_logprob:.4}), no_speech: {:.4} (python {expected_no_speech:.4})",
        result.avg_logprob, result.no_speech_prob
    );
    // text must match; tokens expected-exact (investigate any failure)
    assert_eq!(result.text, expected_text, "transcript text");
    assert_eq!(result.tokens, expected_tokens, "token-level parity");
    assert!((result.avg_logprob - expected_avg_logprob).abs() < 0.02, "avg_logprob");
    assert!((result.no_speech_prob - expected_no_speech).abs() < 0.02, "no_speech_prob");
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn transcribe_segments_match_pytorch() {
    let g = load_json("decode_goldens_tiny.json");
    let t = &g["transcribe"];
    let expected_text = t["text"].as_str().unwrap();
    let expected_segments = t["segments"].as_array().unwrap();

    let mut model = load_tiny();
    let options = TranscribeOptions {
        temperatures: vec![0.0],
        decode_options: DecodingOptions {
            language: Some("en".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    // decode-parity-pure: feed the ffmpeg-decoded PCM fixture so audio
    // resampling differences can't leak into this comparison
    let pcm = Tensor::read_npy(fixtures_dir().join("audio_jfk_pcm.npy"))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let result = whisper_core::transcribe(&mut model, &pcm, &options).unwrap();

    eprintln!("text: {}", result.text);
    assert_eq!(result.text, expected_text, "full transcript");
    assert_eq!(result.segments.len(), expected_segments.len(), "segment count");
    for (seg, exp) in result.segments.iter().zip(expected_segments) {
        assert!((seg.start - exp["start"].as_f64().unwrap()).abs() <= 0.021, "segment start");
        assert!((seg.end - exp["end"].as_f64().unwrap()).abs() <= 0.021, "segment end");
        assert_eq!(seg.text, exp["text"].as_str().unwrap(), "segment text");
    }
}

#[test]
#[ignore = "downloads whisper-tiny"]
fn end_to_end_flac_transcription() {
    // full pipeline including symphonia decode + rubato resample; asserts the
    // transcript text only (audio path is not sample-exact vs ffmpeg)
    let g = load_json("decode_goldens_tiny.json");
    let expected_text = g["transcribe"]["text"].as_str().unwrap();

    let mut model = load_tiny();
    let options = TranscribeOptions {
        temperatures: vec![0.0],
        decode_options: DecodingOptions {
            language: Some("en".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let result =
        whisper_core::transcribe_file(&mut model, fixtures_dir().join("jfk.flac"), &options)
            .unwrap();
    eprintln!("text: {}", result.text);
    assert_eq!(result.text.trim(), expected_text.trim(), "end-to-end transcript");
}
