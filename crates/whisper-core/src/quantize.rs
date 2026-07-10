//! Local safetensors -> GGUF quantization, so every Whisper size can run
//! quantized without depending on third-party GGUF uploads.
//!
//! Policy (matching candle's tensor-tools): quantize 2D+ weight matrices
//! whose last dimension divides the target block size; keep everything else
//! (biases, layer norms, small/odd-shaped tensors) as F32. HF tensor names
//! are preserved so `nn::Whisper::load_gguf` reads the file directly.

use anyhow::{bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::Device;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Quantization {
    pub fn ggml_dtype(&self) -> GgmlDType {
        match self {
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q4_1 => GgmlDType::Q4_1,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q5_1 => GgmlDType::Q5_1,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q2K => GgmlDType::Q2K,
            Self::Q3K => GgmlDType::Q3K,
            Self::Q4K => GgmlDType::Q4K,
            Self::Q5K => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q2K => "q2k",
            Self::Q3K => "q3k",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
        }
    }
}

impl FromStr for Quantization {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "q4_0" => Self::Q4_0,
            "q4_1" => Self::Q4_1,
            "q5_0" => Self::Q5_0,
            "q5_1" => Self::Q5_1,
            "q8_0" => Self::Q8_0,
            "q2k" => Self::Q2K,
            "q3k" => Self::Q3K,
            "q4k" => Self::Q4K,
            "q5k" => Self::Q5K,
            "q6k" => Self::Q6K,
            _ => bail!("unknown quantization {s}; expected q4_0|q4_1|q5_0|q5_1|q8_0|q2k..q6k"),
        })
    }
}

/// Quantize an HF whisper safetensors checkpoint into a GGUF file.
/// Returns (quantized tensor count, kept-f32 tensor count).
pub fn quantize_to_gguf(
    weights: &Path,
    out: &Path,
    quantization: Quantization,
) -> Result<(usize, usize)> {
    let dtype = quantization.ggml_dtype();
    let block = dtype.block_size();
    let tensors =
        candle_core::safetensors::load(weights, &Device::Cpu).context("loading safetensors")?;

    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(tensors.len());
    let (mut quantized, mut kept) = (0usize, 0usize);
    for (name, tensor) in tensors {
        let dims = tensor.dims();
        let is_weight_matrix = dims.len() >= 2 && name.ends_with(".weight");
        // positional embeddings are consumed as f32 lookups, not matmuls
        let is_positional = name.contains("embed_positions");
        let last = *dims.last().unwrap();
        let qt = if is_weight_matrix && !is_positional && last % block == 0 {
            quantized += 1;
            QTensor::quantize(&tensor, dtype)?
        } else if is_weight_matrix && !is_positional && last % GgmlDType::Q8_0.block_size() == 0 {
            // K-quants need the last dim divisible by 256, which d_model=384
            // models (tiny) fail; fall back to q8_0 rather than f32 — an f32
            // QMatMul takes a slow broadcast path at inference time
            quantized += 1;
            QTensor::quantize(&tensor, GgmlDType::Q8_0)?
        } else {
            kept += 1;
            QTensor::quantize(&tensor, GgmlDType::F32)?
        };
        qtensors.push((name, qt));
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file =
        std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut file, &[], &refs)?;
    Ok((quantized, kept))
}
