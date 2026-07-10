//! Full-file transcription: 30-second sliding window with temperature
//! fallback. Port of `whisper/transcribe.py` (word timestamps and
//! hallucination heuristics land in Phase 4).

use crate::audio::{
    load_audio, log_mel_spectrogram, FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, N_SAMPLES,
    SAMPLE_RATE,
};
use crate::decode::{decode, detect_language, DecodingOptions, DecodingResult};
use crate::model::WhisperModel;
use crate::tokenizer::{get_tokenizer, Task, Tokenizer};
use anyhow::{bail, Result};
use candle_core::Tensor;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Fallback ladder; each entry is tried until thresholds pass.
    pub temperatures: Vec<f64>,
    pub compression_ratio_threshold: Option<f64>,
    pub logprob_threshold: Option<f64>,
    pub no_speech_threshold: Option<f64>,
    pub condition_on_previous_text: bool,
    pub initial_prompt: Option<String>,
    /// start,end pairs in seconds; empty means the whole file.
    pub clip_timestamps: Vec<f64>,
    /// Extract word-level timestamps via cross-attention DTW alignment.
    pub word_timestamps: bool,
    /// Punctuation merged with the following word (word_timestamps only).
    pub prepend_punctuations: String,
    /// Punctuation merged with the previous word (word_timestamps only).
    pub append_punctuations: String,
    pub decode_options: DecodingOptions,
    /// None: silent. Some(false): progress only. Some(true): print segments.
    pub verbose: Option<bool>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
            condition_on_previous_text: true,
            initial_prompt: None,
            clip_timestamps: Vec::new(),
            word_timestamps: false,
            prepend_punctuations: "\"'“¿([{-".to_string(),
            append_punctuations: "\"'.。,，!！?？:：”)]}、".to_string(),
            decode_options: DecodingOptions::default(),
            verbose: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub id: usize,
    pub seek: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
}

pub fn transcribe_file<P: AsRef<Path>>(
    model: &mut WhisperModel,
    path: P,
    options: &TranscribeOptions,
) -> Result<TranscribeResult> {
    let audio = load_audio(path)?;
    transcribe(model, &audio, options)
}

pub fn transcribe(
    model: &mut WhisperModel,
    audio: &[f32],
    options: &TranscribeOptions,
) -> Result<TranscribeResult> {
    let device = model.device.clone();
    let n_mels = model.config.num_mel_bins;

    // Pad 30 seconds of silence so the last window can always be sliced.
    let mel = log_mel_spectrogram(audio, n_mels, N_SAMPLES)?;
    let content_frames = mel.n_frames - N_FRAMES;
    let content_duration = content_frames as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

    let mel_window = |seek: usize, size: usize| -> Result<Tensor> {
        let data = mel.window(seek, size, N_FRAMES);
        Ok(Tensor::from_vec(data, (1, n_mels, N_FRAMES), &device)?)
    };

    // language
    let language = match &options.decode_options.language {
        Some(l) => l.clone(),
        None if !model.is_multilingual() => "en".to_string(),
        None => {
            if options.verbose == Some(true) {
                println!("Detecting language using up to the first 30 seconds.");
            }
            let tok = get_tokenizer(true, model.num_languages(), Some("en"), Some(Task::Transcribe))?;
            let window = mel_window(0, N_FRAMES)?;
            let features = model.encoder_forward(&window, true)?;
            let (lang, _) = detect_language(model, &tok, &features)?;
            if options.verbose.is_some() {
                println!("Detected language: {lang}");
            }
            lang
        }
    };
    let task = options.decode_options.task;
    let tokenizer = get_tokenizer(
        model.is_multilingual(),
        model.num_languages(),
        if model.is_multilingual() { Some(language.as_str()) } else { None },
        if model.is_multilingual() { Some(task) } else { None },
    )?;

    // seek clips
    let mut seek_points: Vec<usize> = options
        .clip_timestamps
        .iter()
        .map(|ts| (ts * FRAMES_PER_SECOND as f64).round() as usize)
        .collect();
    if seek_points.is_empty() {
        seek_points.push(0);
    }
    if seek_points.len() % 2 == 1 {
        seek_points.push(content_frames);
    }
    let seek_clips: Vec<(usize, usize)> = seek_points
        .chunks(2)
        .map(|c| (c[0], c[1]))
        .collect();

    let input_stride = N_FRAMES / model.n_audio_ctx(); // mel frames per output token: 2
    let time_precision = input_stride as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64; // 0.02s

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut prompt_reset_since = 0usize;

    let initial_prompt_tokens = match &options.initial_prompt {
        Some(p) => {
            let toks = tokenizer.encode(&format!(" {}", p.trim()));
            all_tokens.extend(&toks);
            toks
        }
        None => Vec::new(),
    };

    let decode_with_fallback = |model: &mut WhisperModel,
                                tokenizer: &Tokenizer,
                                segment: &Tensor,
                                prompt: Vec<u32>|
     -> Result<DecodingResult> {
        let mut result: Option<DecodingResult> = None;
        for &t in &options.temperatures {
            let mut opts = options.decode_options.clone();
            opts.language = Some(language.clone());
            opts.temperature = t;
            opts.prompt = prompt.clone();
            if t > 0.0 {
                opts.beam_size = None;
                opts.patience = None;
            } else {
                opts.best_of = None;
            }
            let r = decode(model, tokenizer, segment, opts)?;

            let mut needs_fallback = false;
            if let Some(threshold) = options.compression_ratio_threshold {
                if r.compression_ratio > threshold {
                    needs_fallback = true; // too repetitive
                }
            }
            if let Some(threshold) = options.logprob_threshold {
                if r.avg_logprob < threshold {
                    needs_fallback = true; // average log probability too low
                }
            }
            if let (Some(ns), Some(lp)) = (options.no_speech_threshold, options.logprob_threshold) {
                if r.no_speech_prob > ns && r.avg_logprob < lp {
                    needs_fallback = false; // silence
                }
            }
            result = Some(r);
            if !needs_fallback {
                break;
            }
        }
        result.ok_or_else(|| anyhow::anyhow!("empty temperature ladder"))
    };

    let mut clip_idx = 0usize;
    let mut seek = seek_clips[0].0;
    let timestamp_begin = tokenizer.timestamp_begin;
    let mut last_speech_timestamp = 0.0f64;

    while clip_idx < seek_clips.len() {
        let (seek_clip_start, seek_clip_end) = seek_clips[clip_idx];
        if seek < seek_clip_start {
            seek = seek_clip_start;
        }
        if seek >= seek_clip_end {
            clip_idx += 1;
            if clip_idx < seek_clips.len() {
                seek = seek_clips[clip_idx].0;
            }
            continue;
        }
        let time_offset = seek as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
        let segment_size = N_FRAMES
            .min(content_frames - seek)
            .min(seek_clip_end - seek);
        let segment_duration = segment_size as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
        let mel_segment = mel_window(seek, segment_size)?;

        // when condition_on_previous_text is false, prompt_reset_since advances
        // to len(all_tokens) after every window, so this stays in sync with Python
        let prompt = all_tokens[prompt_reset_since..].to_vec();

        let result = decode_with_fallback(model, &tokenizer, &mel_segment, prompt)?;
        let tokens = &result.tokens;

        if let Some(ns_threshold) = options.no_speech_threshold {
            let mut should_skip = result.no_speech_prob > ns_threshold;
            if let Some(lp_threshold) = options.logprob_threshold {
                if result.avg_logprob > lp_threshold {
                    // don't skip: confident despite no_speech_prob
                    should_skip = false;
                }
            }
            if should_skip {
                seek += segment_size;
                continue;
            }
        }

        let previous_seek = seek;
        let mut current_segments: Vec<Segment> = Vec::new();

        let is_timestamp: Vec<bool> = tokens.iter().map(|&t| t >= timestamp_begin).collect();
        let single_timestamp_ending = is_timestamp.len() >= 2
            && !is_timestamp[is_timestamp.len() - 2]
            && is_timestamp[is_timestamp.len() - 1];

        // indices where two consecutive timestamp tokens meet (second index)
        let consecutive: Vec<usize> = (1..tokens.len())
            .filter(|&i| is_timestamp[i - 1] && is_timestamp[i])
            .collect();

        let new_segment = |start: f64, end: f64, seg_tokens: &[u32], result: &DecodingResult| {
            let text_tokens: Vec<u32> = seg_tokens
                .iter()
                .copied()
                .filter(|&t| t < tokenizer.eot)
                .collect();
            Segment {
                id: 0, // assigned on push
                seek,
                start,
                end,
                text: tokenizer.decode(&text_tokens),
                tokens: seg_tokens.to_vec(),
                temperature: result.temperature,
                avg_logprob: result.avg_logprob,
                compression_ratio: result.compression_ratio,
                no_speech_prob: result.no_speech_prob,
                words: None,
            }
        };

        if !consecutive.is_empty() {
            let mut slices = consecutive.clone();
            if single_timestamp_ending {
                slices.push(tokens.len());
            }
            let mut last_slice = 0usize;
            for &current_slice in &slices {
                let sliced = &tokens[last_slice..current_slice];
                let start_pos = (sliced[0] - timestamp_begin) as f64;
                let end_pos = (sliced[sliced.len() - 1] - timestamp_begin) as f64;
                current_segments.push(new_segment(
                    time_offset + start_pos * time_precision,
                    time_offset + end_pos * time_precision,
                    sliced,
                    &result,
                ));
                last_slice = current_slice;
            }
            if single_timestamp_ending {
                // no speech after the last timestamp
                seek += segment_size;
            } else {
                // ignore the unfinished segment; seek to the last timestamp
                let last_ts_pos = (tokens[last_slice - 1] - timestamp_begin) as usize;
                seek += last_ts_pos * input_stride;
            }
        } else {
            let mut duration = segment_duration;
            let timestamps: Vec<u32> = tokens
                .iter()
                .copied()
                .filter(|&t| t >= timestamp_begin)
                .collect();
            if let Some(&last) = timestamps.last() {
                if last != timestamp_begin {
                    duration = (last - timestamp_begin) as f64 * time_precision;
                }
            }
            current_segments.push(new_segment(
                time_offset,
                time_offset + duration,
                tokens,
                &result,
            ));
            seek += segment_size;
        }

        if options.word_timestamps {
            crate::timing::add_word_timestamps(
                &mut current_segments,
                model,
                &tokenizer,
                &mel_segment,
                segment_size,
                &options.prepend_punctuations,
                &options.append_punctuations,
                last_speech_timestamp,
            )?;
            if !single_timestamp_ending {
                if let Some(last_word_end) = get_end(&current_segments) {
                    if last_word_end > time_offset {
                        seek = (last_word_end * FRAMES_PER_SECOND as f64).round() as usize;
                    }
                }
            }
            if let Some(last_word_end) = get_end(&current_segments) {
                last_speech_timestamp = last_word_end;
            }
        }

        if options.verbose == Some(true) {
            for s in &current_segments {
                println!(
                    "[{} --> {}] {}",
                    crate::utils::format_timestamp(s.start, false, "."),
                    crate::utils::format_timestamp(s.end, false, "."),
                    s.text
                );
            }
        }

        // clear instantaneous or empty segments
        for s in current_segments.iter_mut() {
            if s.start == s.end || s.text.trim().is_empty() {
                s.text = String::new();
                s.tokens = Vec::new();
                if s.words.is_some() {
                    s.words = Some(Vec::new());
                }
            }
        }

        for mut s in current_segments {
            s.id = all_segments.len();
            all_tokens.extend(&s.tokens);
            all_segments.push(s);
        }

        if !options.condition_on_previous_text || result.temperature > 0.5 {
            // don't feed the prompt tokens if a high temperature was used
            prompt_reset_since = all_tokens.len();
        }

        debug_assert!(seek > previous_seek, "seek must advance");
        if options.verbose == Some(false) {
            let done = seek.min(content_frames) as f64 / content_frames.max(1) as f64;
            eprint!("\rtranscribing: {:5.1}% ", done * 100.0);
        }
    }
    if options.verbose == Some(false) {
        eprintln!();
    }
    let _ = content_duration;

    Ok(TranscribeResult {
        text: tokenizer.decode(&all_tokens[initial_prompt_tokens.len()..]),
        segments: all_segments,
        language,
    })
}

/// Last word-end time across segments (utils.py::get_end).
fn get_end(segments: &[Segment]) -> Option<f64> {
    segments
        .iter()
        .rev()
        .filter_map(|s| s.words.as_ref())
        .flat_map(|w| w.iter().rev())
        .map(|w| w.end)
        .next()
}

/// Guard used by the CLI: English-only models reject other languages.
pub fn validate_language_for_model(model: &WhisperModel, language: Option<&str>) -> Result<()> {
    if !model.is_multilingual() {
        if let Some(l) = language {
            if l != "en" && l != "english" {
                bail!("this model is English-only; --language {l} is not supported");
            }
        }
    }
    Ok(())
}
