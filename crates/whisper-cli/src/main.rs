//! `whisper-candle` — CLI mirroring the reference `whisper` command.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use whisper_core::writers::OutputFormat;
use whisper_core::{DecodingOptions, Task, TranscribeOptions};

#[derive(Parser, Debug)]
#[command(name = "whisper-candle", about = "Pure-Rust Whisper transcription on candle")]
struct Args {
    /// Audio file(s) to transcribe
    #[arg(required = true)]
    audio: Vec<PathBuf>,

    /// Model name (tiny, base, small, medium, large-v3, turbo, ... or *.en variants)
    #[arg(long, default_value = "base")]
    model: String,

    /// Device: cpu, metal, cuda
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Directory to save the outputs
    #[arg(long, short, default_value = ".")]
    output_dir: PathBuf,

    /// Output format: txt, srt, vtt, tsv, json, or all
    #[arg(long, short = 'f', default_value = "all")]
    output_format: String,

    /// Language spoken in the audio (code or name); detected when omitted
    #[arg(long)]
    language: Option<String>,

    /// transcribe or translate
    #[arg(long, default_value = "transcribe")]
    task: String,

    /// Sampling temperature (start of the fallback ladder)
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Temperature step for fallback attempts; 0 disables fallback
    #[arg(long, default_value_t = 0.2)]
    temperature_increment_on_fallback: f64,

    /// Candidates when sampling at temperature > 0
    #[arg(long)]
    best_of: Option<usize>,

    /// Optional text prompt for the first window
    #[arg(long)]
    initial_prompt: Option<String>,

    /// Feed previous output as prompt for the next window
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    condition_on_previous_text: bool,

    /// Gzip compression ratio above which decoding is treated as failed
    #[arg(long, default_value_t = 2.4)]
    compression_ratio_threshold: f64,

    /// Average logprob below which decoding is treated as failed
    #[arg(long, default_value_t = -1.0)]
    logprob_threshold: f64,

    /// No-speech probability above which (with failed logprob) a segment is skipped
    #[arg(long, default_value_t = 0.6)]
    no_speech_threshold: f64,

    /// Comma-separated start,end,... clip timestamps in seconds
    #[arg(long)]
    clip_timestamps: Option<String>,

    /// Extract word-level timestamps (included in json output)
    #[arg(long, default_value_t = false)]
    word_timestamps: bool,

    /// Prepend --initial-prompt to every window's prompt
    #[arg(long, default_value_t = false)]
    carry_initial_prompt: bool,

    /// (requires --word-timestamps) skip silences longer than this many
    /// seconds around probable hallucinations
    #[arg(long)]
    hallucination_silence_threshold: Option<f64>,

    /// Print progress and decoded text
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = whisper_core::device(&args.device)?;
    eprintln!("loading model {} ...", args.model);
    let start = std::time::Instant::now();
    let mut model = whisper_core::load_model(&args.model, &device)?;
    eprintln!("model loaded in {:.1}s", start.elapsed().as_secs_f32());

    let language = args
        .language
        .as_deref()
        .map(whisper_core::tokenizer::normalize_language)
        .transpose()?;
    whisper_core::transcribe::validate_language_for_model(&model, language.as_deref())?;

    let task = match args.task.as_str() {
        "transcribe" => Task::Transcribe,
        "translate" => Task::Translate,
        other => anyhow::bail!("unknown task {other}"),
    };

    let temperatures: Vec<f64> = {
        let inc = args.temperature_increment_on_fallback;
        if inc > 0.0 {
            let mut t = args.temperature;
            let mut v = Vec::new();
            while t <= 1.0 + 1e-6 {
                v.push(t);
                t += inc;
            }
            v
        } else {
            vec![args.temperature]
        }
    };

    let clip_timestamps: Vec<f64> = match &args.clip_timestamps {
        None => Vec::new(),
        Some(s) => s
            .split(',')
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<f64>().context("invalid clip timestamp"))
            .collect::<Result<_>>()?,
    };

    let options = TranscribeOptions {
        temperatures,
        compression_ratio_threshold: Some(args.compression_ratio_threshold),
        logprob_threshold: Some(args.logprob_threshold),
        no_speech_threshold: Some(args.no_speech_threshold),
        condition_on_previous_text: args.condition_on_previous_text,
        initial_prompt: args.initial_prompt.clone(),
        clip_timestamps,
        word_timestamps: args.word_timestamps,
        carry_initial_prompt: args.carry_initial_prompt,
        hallucination_silence_threshold: args.hallucination_silence_threshold,
        decode_options: DecodingOptions {
            task,
            language,
            best_of: args.best_of,
            ..Default::default()
        },
        verbose: Some(args.verbose),
        ..Default::default()
    };

    let formats: Vec<OutputFormat> = if args.output_format == "all" {
        OutputFormat::ALL.to_vec()
    } else {
        vec![args.output_format.parse()?]
    };

    std::fs::create_dir_all(&args.output_dir)?;
    for audio_path in &args.audio {
        let start = std::time::Instant::now();
        match whisper_core::transcribe_file(&mut model, audio_path, &options) {
            Ok(result) => {
                eprintln!(
                    "transcribed {} in {:.2}s",
                    audio_path.display(),
                    start.elapsed().as_secs_f32()
                );
                for format in &formats {
                    let path = format.write_for(&result, audio_path, &args.output_dir)?;
                    eprintln!("wrote {}", path.display());
                }
            }
            Err(e) => {
                eprintln!("skipping {} due to error: {e:#}", audio_path.display());
            }
        }
    }
    Ok(())
}
