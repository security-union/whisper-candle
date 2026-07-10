//! Token decoding for 30-second windows. Port of `whisper/decoding.py`
//! (greedy + temperature sampling; beam search lands in Phase 3).

use crate::model::WhisperModel;
use crate::tokenizer::{Task, Tokenizer};
use crate::utils::compression_ratio;
use anyhow::{bail, Result};
use candle_core::{Device, IndexOp, Tensor};
use rand::Rng;
use std::collections::HashMap;

pub const CHUNK_LENGTH: usize = 30;

#[derive(Debug, Clone)]
pub struct DecodingOptions {
    pub task: Task,
    /// None triggers per-window language detection.
    pub language: Option<String>,
    pub temperature: f64,
    /// Max tokens to sample; defaults to n_text_ctx / 2.
    pub sample_len: Option<usize>,
    /// Independent samples when temperature > 0.
    pub best_of: Option<usize>,
    /// Beam search width when temperature == 0 (not yet implemented).
    pub beam_size: Option<usize>,
    pub patience: Option<f64>,
    /// Google-NMT style length penalty alpha; None = simple length norm.
    pub length_penalty: Option<f64>,
    /// Tokens of previous-window context, injected after <|startofprev|>.
    pub prompt: Vec<u32>,
    /// Tokens forced at the start of the sample.
    pub prefix: Option<String>,
    /// Extra token ids to suppress; `default_suppress` adds the non-speech set.
    pub suppress_tokens: Vec<u32>,
    pub default_suppress: bool,
    pub suppress_blank: bool,
    pub without_timestamps: bool,
    pub max_initial_timestamp: Option<f64>,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            task: Task::Transcribe,
            language: None,
            temperature: 0.0,
            sample_len: None,
            best_of: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            prompt: Vec::new(),
            prefix: None,
            suppress_tokens: Vec::new(),
            default_suppress: true,
            suppress_blank: true,
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodingResult {
    pub language: String,
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

/// Detect the spoken language from encoded audio features.
/// Returns (language_code, probabilities over all languages).
pub fn detect_language(
    model: &mut WhisperModel,
    tokenizer: &Tokenizer,
    audio_features: &Tensor,
) -> Result<(String, HashMap<String, f32>)> {
    let device = model.device.clone();
    let n_audio = audio_features.dim(0)?;
    let sot = tokenizer.sot;
    let x = Tensor::from_vec(vec![sot; n_audio], (n_audio, 1), &device)?;
    let hidden = model.decoder_forward(&x, audio_features, true)?;
    let logits = model.logits_at(&hidden, 0)?; // (n_audio, vocab)
    let row: Vec<f32> = logits.i(0)?.to_vec1()?;

    let lang_tokens = tokenizer.all_language_tokens();
    let lang_codes = tokenizer.all_language_codes();

    // softmax restricted to language tokens (everything else masked to -inf)
    let mut max = f32::NEG_INFINITY;
    for &t in &lang_tokens {
        max = max.max(row[t as usize]);
    }
    let mut sum = 0f32;
    let mut probs: Vec<f32> = lang_tokens
        .iter()
        .map(|&t| {
            let p = (row[t as usize] - max).exp();
            sum += p;
            p
        })
        .collect();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let map: HashMap<String, f32> = lang_codes
        .iter()
        .zip(&probs)
        .map(|(c, p)| (c.to_string(), *p))
        .collect();
    Ok((lang_codes[best].to_string(), map))
}

/// Logit filters, applied in-place per batch row. Port of decoding.py:423-505.
enum LogitFilter {
    SuppressBlank { tokens: Vec<u32>, sample_begin: usize },
    SuppressTokens { tokens: Vec<u32> },
    TimestampRules(TimestampRules),
}

pub struct TimestampRules {
    pub no_timestamps: u32,
    pub timestamp_begin: u32,
    pub eot: u32,
    pub sample_begin: usize,
    pub max_initial_timestamp_index: Option<usize>,
}

impl LogitFilter {
    fn apply(&self, logits: &mut [f32], ctx: &[u32]) {
        match self {
            Self::SuppressBlank { tokens, sample_begin } => {
                if ctx.len() == *sample_begin {
                    for &t in tokens {
                        logits[t as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            Self::SuppressTokens { tokens } => {
                for &t in tokens {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }
            Self::TimestampRules(r) => apply_timestamp_rules(logits, ctx, r),
        }
    }
}

/// Exposed for fixture tests (timestamp_rules_goldens.npz).
pub fn apply_timestamp_rules(logits: &mut [f32], ctx: &[u32], r: &TimestampRules) {
    let ts = r.timestamp_begin as usize;
    // suppress <|notimestamps|>
    logits[r.no_timestamps as usize] = f32::NEG_INFINITY;

    let seq = &ctx[r.sample_begin.min(ctx.len())..];
    let last_was_timestamp = !seq.is_empty() && seq[seq.len() - 1] >= r.timestamp_begin;
    let penultimate_was_timestamp = seq.len() < 2 || seq[seq.len() - 2] >= r.timestamp_begin;

    if last_was_timestamp {
        if penultimate_was_timestamp {
            // has to be non-timestamp
            for v in logits[ts..].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        } else {
            // cannot be normal text tokens
            for v in logits[..r.eot as usize].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // timestamps shouldn't decrease; also force nonzero-length segments
    if let Some(&last_ts) = seq.iter().rev().find(|&&t| t >= r.timestamp_begin) {
        let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
            last_ts as usize
        } else {
            last_ts as usize + 1
        };
        for v in logits[ts..timestamp_last].iter_mut() {
            *v = f32::NEG_INFINITY;
        }
    }

    if ctx.len() == r.sample_begin {
        // suppress generating non-timestamp tokens at the beginning
        for v in logits[..ts].iter_mut() {
            *v = f32::NEG_INFINITY;
        }
        if let Some(max_idx) = r.max_initial_timestamp_index {
            let last_allowed = ts + max_idx;
            if last_allowed + 1 < logits.len() {
                for v in logits[last_allowed + 1..].iter_mut() {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // if the probability mass on timestamps exceeds any single text token,
    // sample a timestamp
    let logprobs = log_softmax(logits);
    let timestamp_logprob = logsumexp(&logprobs[ts..]);
    let max_text = logprobs[..ts].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if timestamp_logprob > max_text {
        for v in logits[..ts].iter_mut() {
            *v = f32::NEG_INFINITY;
        }
    }
}

pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|&x| x - max - log_sum).collect()
}

fn logsumexp(logprobs: &[f32]) -> f32 {
    let max = logprobs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = logprobs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

pub struct DecodingTask<'a> {
    model: &'a mut WhisperModel,
    tokenizer: &'a Tokenizer,
    options: DecodingOptions,
    n_group: usize,
    n_ctx: usize,
    sample_len: usize,
    sot_index: usize,
    sample_begin: usize,
    initial_tokens: Vec<u32>,
    filters: Vec<LogitFilter>,
}

impl<'a> DecodingTask<'a> {
    pub fn new(
        model: &'a mut WhisperModel,
        tokenizer: &'a Tokenizer,
        options: DecodingOptions,
    ) -> Result<Self> {
        if options.beam_size.is_some() {
            bail!("beam search is not implemented yet; use greedy (temperature=0) or best_of sampling");
        }
        if options.temperature == 0.0 && options.best_of.is_some() {
            bail!("best_of with greedy sampling (temperature=0) is not compatible");
        }
        if let Some(lp) = options.length_penalty {
            if !(0.0..=1.0).contains(&lp) {
                bail!("length_penalty (alpha) should be between 0 and 1");
            }
        }

        let n_group = options.beam_size.or(options.best_of).unwrap_or(1);
        let n_ctx = model.n_text_ctx();
        let sample_len = options.sample_len.unwrap_or(n_ctx / 2);

        let sot_sequence = if options.without_timestamps {
            tokenizer.sot_sequence_including_notimestamps()
        } else {
            tokenizer.sot_sequence.clone()
        };

        // _get_initial_tokens
        let mut tokens = sot_sequence.clone();
        if let Some(prefix) = &options.prefix {
            let mut prefix_tokens = tokenizer.encode(&format!(" {}", prefix.trim()));
            let max_prefix_len = n_ctx / 2 - sample_len.min(n_ctx / 2);
            if prefix_tokens.len() > max_prefix_len {
                prefix_tokens = prefix_tokens[prefix_tokens.len() - max_prefix_len..].to_vec();
            }
            tokens.extend(prefix_tokens);
        }
        if !options.prompt.is_empty() {
            let max_prompt = n_ctx / 2 - 1;
            let tail = if options.prompt.len() > max_prompt {
                &options.prompt[options.prompt.len() - max_prompt..]
            } else {
                &options.prompt[..]
            };
            let mut with_prev = vec![tokenizer.sot_prev];
            with_prev.extend_from_slice(tail);
            with_prev.extend(tokens);
            tokens = with_prev;
        }
        let initial_tokens = tokens;
        let sample_begin = initial_tokens.len();
        let sot_index = initial_tokens
            .iter()
            .position(|&t| t == tokenizer.sot)
            .expect("sot missing from initial tokens");

        let mut filters = Vec::new();
        if options.suppress_blank {
            let mut blank = tokenizer.encode(" ");
            blank.push(tokenizer.eot);
            filters.push(LogitFilter::SuppressBlank { tokens: blank, sample_begin });
        }
        {
            // _get_suppress_tokens
            let mut suppress = options.suppress_tokens.clone();
            if options.default_suppress {
                suppress.extend(tokenizer.non_speech_tokens());
            }
            suppress.extend([
                tokenizer.transcribe,
                tokenizer.translate,
                tokenizer.sot,
                tokenizer.sot_prev,
                tokenizer.sot_lm,
                tokenizer.no_speech,
            ]);
            suppress.sort_unstable();
            suppress.dedup();
            filters.push(LogitFilter::SuppressTokens { tokens: suppress });
        }
        if !options.without_timestamps {
            let precision = CHUNK_LENGTH as f64 / model.n_audio_ctx() as f64; // 0.02s
            let max_initial_timestamp_index = options
                .max_initial_timestamp
                .map(|m| (m / precision).round() as usize);
            filters.push(LogitFilter::TimestampRules(TimestampRules {
                no_timestamps: tokenizer.no_timestamps,
                timestamp_begin: tokenizer.timestamp_begin,
                eot: tokenizer.eot,
                sample_begin,
                max_initial_timestamp_index,
            }));
        }

        Ok(Self {
            model,
            tokenizer,
            options,
            n_group,
            n_ctx,
            sample_len,
            sot_index,
            sample_begin,
            initial_tokens,
            filters,
        })
    }

    /// Decode one 30-second mel window (1, n_mels, N_FRAMES).
    pub fn run(&mut self, mel: &Tensor) -> Result<DecodingResult> {
        let device: Device = self.model.device.clone();
        let audio_features = self.model.encoder_forward(mel, true)?;

        // language detection (per-window) when not specified
        let mut initial_tokens = self.initial_tokens.clone();
        let language = match &self.options.language {
            Some(l) => l.clone(),
            None => {
                let (lang, _) = detect_language(self.model, self.tokenizer, &audio_features)?;
                let lang_token = self.tokenizer.to_language_token(&lang)?;
                initial_tokens[self.sot_index + 1] = lang_token;
                lang
            }
        };

        let n = self.n_group;
        let features = if n > 1 {
            audio_features.repeat((n, 1, 1))?
        } else {
            audio_features
        };

        let mut rows: Vec<Vec<u32>> = vec![initial_tokens.clone(); n];
        let mut sum_logprobs = vec![0f64; n];
        let mut no_speech_prob = f64::NAN;
        let eot = self.tokenizer.eot;
        let mut rng = rand::thread_rng();

        for i in 0..self.sample_len {
            let seq_len = rows[0].len();
            let flat: Vec<u32> = rows.iter().flatten().copied().collect();
            let tokens_t = Tensor::from_vec(flat, (n, seq_len), &device)?;
            let hidden = self.model.decoder_forward(&tokens_t, &features, i == 0)?;

            if i == 0 {
                // probability of <|nospeech|> at the sot position
                let logits_sot = self.model.logits_at(&hidden, self.sot_index)?;
                let row: Vec<f32> = logits_sot.i(0)?.to_vec1()?;
                let probs = softmax(&row);
                no_speech_prob = probs[self.tokenizer.no_speech as usize] as f64;
            }

            let logits_last = self.model.logits_at(&hidden, seq_len - 1)?;
            let logits_rows: Vec<Vec<f32>> = logits_last.to_vec2()?;

            let mut completed = true;
            for (g, mut logits) in logits_rows.into_iter().enumerate() {
                for filter in &self.filters {
                    filter.apply(&mut logits, &rows[g]);
                }

                let last = *rows[g].last().unwrap();
                let next = if last == eot {
                    eot
                } else {
                    let chosen = if self.options.temperature == 0.0 {
                        argmax(&logits)
                    } else {
                        sample_gumbel(&logits, self.options.temperature, &mut rng)
                    };
                    let logprobs = log_softmax(&logits);
                    sum_logprobs[g] += logprobs[chosen as usize] as f64;
                    chosen
                };
                rows[g].push(next);
                if next != eot {
                    completed = false;
                }
            }

            if completed || rows[0].len() > self.n_ctx {
                break;
            }
        }

        // finalize: ensure a trailing eot, slice sample_begin..eot
        let sliced: Vec<Vec<u32>> = rows
            .iter()
            .map(|row| {
                let end = row[self.sample_begin..]
                    .iter()
                    .position(|&t| t == eot)
                    .map(|p| self.sample_begin + p)
                    .unwrap_or(row.len());
                row[self.sample_begin..end].to_vec()
            })
            .collect();

        // MaximumLikelihoodRanker
        let selected = (0..n)
            .max_by(|&a, &b| {
                let score = |g: usize| {
                    let length = sliced[g].len() as f64;
                    let penalty = match self.options.length_penalty {
                        None => length,
                        Some(alpha) => ((5.0 + length) / 6.0).powf(alpha),
                    };
                    sum_logprobs[g] / penalty
                };
                score(a).partial_cmp(&score(b)).unwrap()
            })
            .unwrap_or(0);

        let tokens = sliced[selected].clone();
        let text = self.tokenizer.decode(&tokens).trim().to_string();
        let avg_logprob = sum_logprobs[selected] / (tokens.len() as f64 + 1.0);

        Ok(DecodingResult {
            language,
            compression_ratio: compression_ratio(&text),
            text,
            tokens,
            avg_logprob,
            no_speech_prob,
            temperature: self.options.temperature,
        })
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Categorical sampling via the Gumbel-max trick (distributionally equivalent
/// to torch.distributions.Categorical on logits/T; not RNG-compatible).
fn sample_gumbel(logits: &[f32], temperature: f64, rng: &mut impl Rng) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l == f32::NEG_INFINITY {
            continue;
        }
        let u: f32 = rng.gen_range(1e-9..1.0);
        let g = -(-u.ln()).ln();
        let v = l / temperature as f32 + g;
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Convenience one-shot decode of a single mel window.
pub fn decode(
    model: &mut WhisperModel,
    tokenizer: &Tokenizer,
    mel: &Tensor,
    options: DecodingOptions,
) -> Result<DecodingResult> {
    DecodingTask::new(model, tokenizer, options)?.run(mel)
}
