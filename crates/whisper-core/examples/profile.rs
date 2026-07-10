//! Rough phase timing: `cargo run --release --features accelerate --example profile -- <audio> [model]`

use candle_core::{Device, Tensor};
use std::time::Instant;
use whisper_core::audio::{load_audio, log_mel_spectrogram, N_FRAMES, N_SAMPLES};
use whisper_core::tokenizer::{get_tokenizer, Task};
use whisper_core::DecodingOptions;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args.get(1).map(String::as_str).unwrap_or("tests/fixtures/jfk_60s.wav");
    let model_name = args.get(2).map(String::as_str).unwrap_or("tiny");
    let quantization = args.get(3).map(String::as_str);

    let device = Device::Cpu;
    let mut model = match quantization {
        Some(q) => whisper_core::load_model_quantized(model_name, q.parse()?, &device)?,
        None => whisper_core::load_model(model_name, &device)?,
    };
    let tokenizer = get_tokenizer(model.is_multilingual(), model.num_languages(), Some("en"), Some(Task::Transcribe))?;

    let t0 = Instant::now();
    let pcm = load_audio(audio_path)?;
    println!("load_audio: {:?} ({} samples)", t0.elapsed(), pcm.len());

    let t0 = Instant::now();
    let mel = log_mel_spectrogram(&pcm, model.config.num_mel_bins, N_SAMPLES)?;
    println!("log_mel:    {:?} ({} frames)", t0.elapsed(), mel.n_frames);

    let n_mels = model.config.num_mel_bins;
    let window = mel.window(0, N_FRAMES, N_FRAMES);
    let mel_t = Tensor::from_vec(window, (1, n_mels, N_FRAMES), &device)?;

    // encoder (warm + timed)
    let _ = model.encoder_forward(&mel_t, true)?;
    let t0 = Instant::now();
    let features = model.encoder_forward(&mel_t, true)?;
    println!("encoder:    {:?}", t0.elapsed());

    // decode one window
    let t0 = Instant::now();
    let opts = DecodingOptions { language: Some("en".into()), ..Default::default() };
    let result = whisper_core::decode(&mut model, &tokenizer, &mel_t, opts)?;
    let n_tokens = result.tokens.len();
    println!("decode:     {:?} ({} tokens, {:.1} ms/token)", t0.elapsed(), n_tokens, t0.elapsed().as_secs_f64() * 1000.0 / (n_tokens + 4) as f64);

    // decoder steps alone: run 50 single-token steps against cached features
    let sot = tokenizer.sot_sequence.clone();
    let toks = Tensor::from_vec(sot.clone(), (1, sot.len()), &device)?;
    let _ = model.decoder_forward(&toks, &features, true)?;
    let one = Tensor::from_vec(vec![100u32], (1, 1), &device)?;
    let t0 = Instant::now();
    for _ in 0..50 {
        let h = model.decoder_forward(&one, &features, false)?;
        let _ = model.logits_at(&h, 0)?;
    }
    println!("decoder step: {:.2} ms (incl. final_linear)", t0.elapsed().as_secs_f64() * 1000.0 / 50.0);

    // final_linear alone
    let h = model.decoder_forward(&one, &features, false)?;
    let t0 = Instant::now();
    for _ in 0..50 {
        let _ = model.logits_at(&h, 0)?;
    }
    println!("final_linear: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0 / 50.0);

    Ok(())
}
