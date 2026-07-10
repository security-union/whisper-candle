//! Pure-Rust OpenAI Whisper inference on candle — no C/C++ bindings.
//!
//! Port of [openai/whisper](https://github.com/openai/whisper); see DESIGN.md
//! at the repository root for the architecture and parity test strategy.

pub mod audio;
pub mod decode;
pub mod hub;
pub mod model;
pub mod nn;
pub mod quantize;
pub mod timing;
pub mod tokenizer;
pub mod transcribe;
pub mod utils;
pub mod writers;

pub use decode::{decode, detect_language, DecodingOptions, DecodingResult};
pub use hub::{fetch_model, WhichModel};
pub use model::WhisperModel;
pub use quantize::Quantization;
pub use tokenizer::{get_tokenizer, Task, Tokenizer};
pub use transcribe::{transcribe, transcribe_file, Segment, TranscribeOptions, TranscribeResult};

use anyhow::Result;
use candle_core::Device;

/// Pick a device: Metal/CUDA when compiled in and requested, else CPU.
pub fn device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" | "gpu" => Ok(Device::new_metal(0)?),
        #[cfg(feature = "cuda")]
        "cuda" | "gpu" => Ok(Device::new_cuda(0)?),
        other => anyhow::bail!(
            "device {other} not available in this build (enable the `metal`/`cuda` feature)"
        ),
    }
}

/// Load a model by name, downloading from the HF Hub on first use.
pub fn load_model(name: &str, device: &Device) -> Result<WhisperModel> {
    let which: WhichModel = name.parse()?;
    let files = fetch_model(which)?;
    let mut model = WhisperModel::load(&files.config, &files.weights, device)?;
    if let Some(gc) = &files.generation_config {
        model.set_alignment_heads_from_file(gc)?;
    }
    Ok(model)
}

/// Load a model quantized to the given GGML dtype. The f32 safetensors are
/// downloaded from the HF Hub, quantized locally once, and the resulting GGUF
/// is cached under `~/.cache/whisper-candle/`.
pub fn load_model_quantized(
    name: &str,
    quantization: Quantization,
    device: &Device,
) -> Result<WhisperModel> {
    let which: WhichModel = name.parse()?;
    let files = fetch_model(which)?;

    let cache_root = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .ok_or_else(|| anyhow::anyhow!("cannot determine cache directory"))?
        .join("whisper-candle");
    let gguf = cache_root.join(format!("{name}-{}.gguf", quantization.name()));
    if !gguf.exists() {
        eprintln!("quantizing {name} to {} (one-time)...", quantization.name());
        let (q, kept) = quantize::quantize_to_gguf(&files.weights, &gguf, quantization)?;
        eprintln!("wrote {} ({q} tensors quantized, {kept} kept f32)", gguf.display());
    }

    let mut model = WhisperModel::load_quantized(&files.config, &gguf, device)?;
    if let Some(gc) = &files.generation_config {
        model.set_alignment_heads_from_file(gc)?;
    }
    Ok(model)
}
