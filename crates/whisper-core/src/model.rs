//! Thin wrapper around candle-transformers' Whisper model.
//! Mirrors the properties of `whisper/model.py::Whisper`.

use crate::nn;
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::Config;
use std::path::Path;

pub struct WhisperModel {
    inner: nn::Whisper,
    pub config: Config,
    pub device: Device,
    /// (layer, head) pairs of cross-attention heads correlated with word
    /// timing. From generation_config.json when available; defaults to all
    /// heads in the upper half of decoder layers (model.py::Whisper.__init__).
    pub alignment_heads: Option<Vec<(usize, usize)>>,
}

impl WhisperModel {
    pub fn load<P: AsRef<Path>>(config_path: P, weights_path: P, device: &Device) -> Result<Self> {
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_path.as_ref()).context("reading config.json")?,
        )?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, device)?
        };
        let inner = nn::Whisper::load(&vb, config.clone())?;
        Ok(Self { inner, config, device: device.clone(), alignment_heads: None })
    }

    /// Read `alignment_heads` from a generation_config.json if it has them.
    pub fn set_alignment_heads_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if let Some(heads) = v.get("alignment_heads").and_then(|h| h.as_array()) {
            let pairs: Vec<(usize, usize)> = heads
                .iter()
                .filter_map(|p| {
                    let p = p.as_array()?;
                    Some((p.first()?.as_u64()? as usize, p.get(1)?.as_u64()? as usize))
                })
                .collect();
            if !pairs.is_empty() {
                self.alignment_heads = Some(pairs);
            }
        }
        Ok(())
    }

    /// Alignment heads, falling back to the reference default: every head in
    /// the upper half of the decoder layers.
    pub fn alignment_heads(&self) -> Vec<(usize, usize)> {
        match &self.alignment_heads {
            Some(h) => h.clone(),
            None => {
                let layers = self.config.decoder_layers;
                let heads = self.config.decoder_attention_heads;
                (layers / 2..layers)
                    .flat_map(|l| (0..heads).map(move |h| (l, h)))
                    .collect()
            }
        }
    }

    /// Full-sequence decoder forward that also returns per-layer
    /// cross-attention QK matrices; used for word-timestamp alignment.
    pub fn decoder_forward_with_cross_qk(
        &mut self,
        tokens: &Tensor,
        audio_features: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        Ok(self.inner.decoder.forward_with_cross_qk(tokens, audio_features)?)
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

    /// Incremental decoder forward -> hidden states (batch, seq, d_model).
    /// Pass the full prompt with `flush = true` on the first call, then only
    /// the newly sampled token(s) with `flush = false`.
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

    /// Reorder decoder self-attention KV caches for beam search.
    pub fn rearrange_kv_cache(&mut self, source_indices: &[usize]) -> Result<()> {
        Ok(self.inner.decoder.rearrange_kv_cache(source_indices)?)
    }
}
