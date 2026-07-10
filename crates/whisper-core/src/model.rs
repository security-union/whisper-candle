//! Thin wrapper around candle-transformers' Whisper model.
//! Mirrors the properties of `whisper/model.py::Whisper`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{model as m, Config};
use std::path::Path;

pub struct WhisperModel {
    inner: m::Whisper,
    pub config: Config,
    pub device: Device,
}

impl WhisperModel {
    pub fn load<P: AsRef<Path>>(config_path: P, weights_path: P, device: &Device) -> Result<Self> {
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_path.as_ref()).context("reading config.json")?,
        )?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, device)?
        };
        let inner = m::Whisper::load(&vb, config.clone())?;
        Ok(Self { inner, config, device: device.clone() })
    }

    pub fn is_multilingual(&self) -> bool {
        self.config.vocab_size >= 51865
    }

    pub fn num_languages(&self) -> usize {
        self.config.vocab_size - 51765 - usize::from(self.is_multilingual())
    }

    pub fn n_text_ctx(&self) -> usize {
        self.config.max_target_positions
    }

    pub fn n_audio_ctx(&self) -> usize {
        self.config.max_source_positions
    }

    /// Encode a mel window (batch, n_mels, n_frames) -> (batch, n_audio_ctx, d_model).
    pub fn encoder_forward(&mut self, mel: &Tensor, flush: bool) -> Result<Tensor> {
        Ok(self.inner.encoder.forward(mel, flush)?)
    }

    /// Decoder forward over the full token sequence -> hidden states
    /// (batch, seq, d_model). `flush` recomputes the cross-attention KV cache;
    /// pass true on the first call for a given set of audio features.
    pub fn decoder_forward(&mut self, tokens: &Tensor, audio_features: &Tensor, flush: bool) -> Result<Tensor> {
        Ok(self.inner.decoder.forward(tokens, audio_features, flush)?)
    }

    /// Project hidden states to vocabulary logits.
    pub fn decoder_final_linear(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.inner.decoder.final_linear(hidden)?)
    }

    /// Logits at a single sequence position: (batch, vocab).
    pub fn logits_at(&self, hidden: &Tensor, position: usize) -> Result<Tensor> {
        let h = hidden.i((.., position..position + 1, ..))?;
        Ok(self.decoder_final_linear(&h)?.squeeze(1)?)
    }

    pub fn reset_kv_cache(&mut self) {
        self.inner.reset_kv_cache();
    }
}
