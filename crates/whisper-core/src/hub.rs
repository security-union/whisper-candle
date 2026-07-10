//! Model download from the Hugging Face Hub (safetensors + config.json).
//! Replaces the Azure-CDN `.pt` download in `whisper/__init__.py`.

use anyhow::{bail, Result};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV1,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
}

impl WhichModel {
    pub fn hf_repo(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::TinyEn => "openai/whisper-tiny.en",
            Self::Base => "openai/whisper-base",
            Self::BaseEn => "openai/whisper-base.en",
            Self::Small => "openai/whisper-small",
            Self::SmallEn => "openai/whisper-small.en",
            Self::Medium => "openai/whisper-medium",
            Self::MediumEn => "openai/whisper-medium.en",
            Self::LargeV1 => "openai/whisper-large",
            Self::LargeV2 => "openai/whisper-large-v2",
            Self::LargeV3 => "openai/whisper-large-v3",
            Self::LargeV3Turbo => "openai/whisper-large-v3-turbo",
        }
    }

    pub fn is_multilingual(&self) -> bool {
        !matches!(self, Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn)
    }

    pub const ALL_NAMES: &'static [&'static str] = &[
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
        "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo",
    ];
}

impl FromStr for WhichModel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "tiny" => Self::Tiny,
            "tiny.en" => Self::TinyEn,
            "base" => Self::Base,
            "base.en" => Self::BaseEn,
            "small" => Self::Small,
            "small.en" => Self::SmallEn,
            "medium" => Self::Medium,
            "medium.en" => Self::MediumEn,
            "large-v1" => Self::LargeV1,
            "large-v2" => Self::LargeV2,
            "large-v3" | "large" => Self::LargeV3,
            "large-v3-turbo" | "turbo" => Self::LargeV3Turbo,
            _ => bail!("unknown model {s}; expected one of {:?}", Self::ALL_NAMES),
        })
    }
}

pub struct ModelFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    /// generation_config.json (carries `alignment_heads` for word timestamps);
    /// not present in every repo.
    pub generation_config: Option<PathBuf>,
}

/// Download (or reuse from the HF cache) config.json and model.safetensors.
pub fn fetch_model(which: WhichModel) -> Result<ModelFiles> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(which.hf_repo().to_string());
    let config = repo.get("config.json")?;
    let weights = repo.get("model.safetensors")?;
    let generation_config = repo.get("generation_config.json").ok();
    Ok(ModelFiles { config, weights, generation_config })
}
