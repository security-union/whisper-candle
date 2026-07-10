//! Pure-Rust OpenAI Whisper inference on candle — no C/C++ bindings.
//!
//! Port of [openai/whisper](https://github.com/openai/whisper); see DESIGN.md
//! at the repository root for the architecture and parity test strategy.

pub mod audio;
pub mod decode;
pub mod hub;
pub mod model;
pub mod tokenizer;
pub mod transcribe;
pub mod utils;
pub mod writers;

pub use decode::{decode, detect_language, DecodingOptions, DecodingResult};
pub use hub::{fetch_model, WhichModel};
pub use model::WhisperModel;
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
    WhisperModel::load(&files.config, &files.weights, device)
}
