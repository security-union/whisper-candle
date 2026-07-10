# whisper-rs — Pure-Rust Whisper Port on Candle

**Status:** v1 · 2026-07-09 · **Phases 0–4 complete** (minus two transcribe flags)

> **Implementation status (2026-07-09):**
> - Phases 0–2 done: all L0–L3 golden tests green, including **token-exact greedy
>   decode parity with PyTorch** on whisper-tiny, and end-to-end flac
>   transcription through the pure-Rust audio path.
> - Phase 3 done: **beam search is token-exact vs PyTorch** (python-dict candidate
>   ordering + KV-cache reordering), temperature fallback ladder, best_of
>   sampling, prompts, language detection, all writers, CLI.
> - Phase 4 done: **word timestamps within 0.1s of PyTorch** — cross-attention QK
>   capture (explicit return values instead of hooks), alignment heads from HF
>   generation_config.json, DTW + median filter at exact fixture parity.
>   All transcribe flags are now ported, including `hallucination_silence_threshold`
>   and `carry_initial_prompt`.
> - Model is vendored (`src/nn.rs`) with incremental KV-cache decoding.
> - Metal backend works with **correct** transcripts (torch MPS produces garbage);
>   base on Metal beats CPU (6.9s vs 11.1s on the 60s clip), tiny doesn't
>   (kernel-launch overhead dominates small models).
> - Phase 5 first pass done: KV caches now live in pre-scaled head layout, so
>   the single-token decode step never re-lays-out the 1500-frame cross-attention
>   keys (profiling showed that relayout was ~70% of the 6ms step; now 1.45ms).
>   Measured on the 60s clip (M4 Max): CPU+accelerate tiny 2.0s (30× RTF),
>   base 4.5s (13×), small 12.3s; Metal tiny 2.4s, base 3.1s, small 8.4s —
>   Metal wins from base up. Python torch CPU remains ~4× faster (tiny 0.47s,
>   base 1.0s): the residual gap is per-op overhead in candle's decode path →
>   next levers are fused logit math, fewer allocations, quantized models.
> - Known accepted deviations: resampled audio is not sample-exact vs ffmpeg
>   (different sinc filters, fractional delay; RMS ≈ 0.02 on jfk.flac);
>   compression_ratio uses zlib-rs (within ~2% of C zlib — only feeds the
>   `> 2.4` threshold); language-token *order* matches by mapping, not by
>   Python's set-iteration order.
**Reference implementation:** `../whisper` (openai/whisper @ local checkout)
**No C/C++ bindings.** All inference through [candle](https://github.com/huggingface/candle) (`candle-core` / `candle-nn` / `candle-transformers`).

---

## 1. Goals & non-goals

### Goals
- Pure-Rust, dependency-light Whisper inference: audio file → segments with timestamps → txt/srt/vtt/json.
- Behavioral parity with the Python reference, enforced by golden-fixture tests generated from `../whisper`.
- All model sizes (`tiny` → `large-v3`, `turbo`) via safetensors from the Hugging Face Hub.
- CPU (fp32, Accelerate-backed on macOS) and Metal backends from day one; CUDA behind a feature flag.
- Library crate first (`whisper-core`), thin CLI on top — so it can be embedded (e.g. in a streaming server) later.

### Non-goals
- Training / fine-tuning.
- Triton GPU kernels (`triton_ops.py`) — the CPU DTW/median-filter paths are fast enough in Rust.
- Text normalizers (`normalizers/`) — only needed for WER *evaluation*; deferred to Phase 5 and only inside the eval harness.
- Bitwise-identical outputs to PyTorch (impossible across BLAS backends; see §6 tolerance policy).

### Naming
`whisper-rs` on crates.io is taken (whisper.cpp bindings). Working name for the published crates: **`whisper-candle`** (`whisper-candle-core`, `whisper-candle-cli`). Repo directory name stays as-is.

---

## 2. Workspace layout

```
whisper-rs/
├── Cargo.toml                  # workspace
├── DESIGN.md                   # this file
├── crates/
│   ├── whisper-core/           # the library (everything below)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── audio.rs        # decode + resample + log-mel
│   │   │   ├── tokenizer.rs    # tiktoken BPE + special tokens
│   │   │   ├── model.rs        # encoder/decoder (vendored from candle-transformers, extended)
│   │   │   ├── decode.rs       # DecodingTask: greedy, beam, logit filters
│   │   │   ├── transcribe.rs   # 30s sliding-window orchestration
│   │   │   ├── timing.rs       # word timestamps: median filter + DTW
│   │   │   ├── writers.rs      # txt / srt / vtt / tsv / json
│   │   │   └── hub.rs          # HF Hub download + cache (hf-hub crate)
│   │   ├── assets/             # multilingual.tiktoken, gpt2.tiktoken, mel filters (include_bytes!)
│   │   └── tests/              # golden-fixture tests (see §6)
│   └── whisper-cli/            # clap CLI mirroring `whisper` CLI flags
├── tools/
│   ├── gen_fixtures.py         # generates tests/fixtures/** from ../whisper  ← ALREADY WRITTEN
│   └── bench_python.py         # PyTorch baseline benchmarks               ← ALREADY WRITTEN
├── tests/fixtures/             # committed goldens (.npy/.npz + .json, ~15 MB)
└── benches/                    # criterion micro-benches + hyperfine scripts
```

## 3. Module mapping (Python → Rust)

| Python (`../whisper`) | Rust module | Key crates | Notes |
|---|---|---|---|
| `audio.py` | `audio.rs` | `symphonia`, `rubato`, candle | §4.2 |
| `tokenizer.py` | `tokenizer.rs` | `tiktoken-rs` | §4.3 |
| `model.py` | `model.rs` | `candle-nn` | §4.1 — vendored & extended |
| `decoding.py` | `decode.rs` | — | §4.4 — the main porting work |
| `transcribe.py` | `transcribe.rs` | `flate2` (compression ratio) | §4.5 |
| `timing.py` | `timing.rs` | — | plain-Rust DTW replaces numba |
| `utils.py` | `writers.rs` | `serde_json` | direct port |
| `__init__.py` (download) | `hub.rs` | `hf-hub`, `sha2` | HF safetensors, not .pt |
| `triton_ops.py` | — | — | dropped |
| `normalizers/` | eval harness only | `fancy-regex` | Phase 5 |

---

## 4. Key design decisions

### 4.1 Model: vendor `candle-transformers::models::whisper`, don't just depend on it

`candle-transformers` ships a working Whisper `model.rs` (+ `quantized_model.rs`) with built-in KV caching, and its `audio` module already implements whisper-exact `pcm_to_mel`. We start from it but **vendor `model.rs` into `whisper-core::model`** because we need three things its public API doesn't expose:

1. **Cross-attention QK matrices** for word timestamps (`timing.py` gets them via PyTorch forward hooks; Rust has no hooks — the decoder forward takes a `capture_cross_qk: bool` and returns them explicitly).
2. **KV-cache reordering** for beam search (`rearrange_kv_cache` in `decoding.py:172` → `index_select` on the batch dim of each layer's cached K/V).
3. **`no_speech` logits at the SOT position** (needs logits for the *full* prompt on the first forward pass, not just the last position).

The Python "forward hook" pattern (`model.py:310`, `timing.py:187`) translates to: KV cache is an explicit `Vec<Option<(Tensor, Tensor)>>` owned by the decoding task and passed `&mut` into `decoder.forward()`. No globals, no interior mutability.

**Weights:** primary path is safetensors from HF Hub repos (`openai/whisper-tiny` … `openai/whisper-large-v3-turbo`) via `hf-hub` — candle's whisper model already uses HF parameter naming, and `generation_config.json` in those repos carries `alignment_heads`, `suppress_tokens`, etc., which lets us **drop the base85+gzip alignment-head blobs entirely**. Loading original OpenAI `.pt` checkpoints (candle has a pickle reader) is a stretch goal, not Phase 1.

**Dtype policy:** fp32 everywhere in "parity mode" (all golden tests run fp32 CPU). f16 on Metal as an opt-in `--fp16` once parity is green. Quantized (GGUF via `quantized_model.rs`) in Phase 5.

### 4.2 Audio: pure Rust, ffmpeg as escape hatch

- Decode: `symphonia` (wav/flac/mp3/ogg/m4a features on).
- Resample to 16 kHz mono: `rubato` (only when the source isn't already 16k — the test fixture `jfk.flac` is 16k, so parity tests bypass resampling entirely).
- Feature `ffmpeg-fallback`: shell out exactly like `audio.py:load_audio` for exotic containers.
- Log-mel: start with `candle_transformers::models::whisper::audio::{pcm_to_mel, log_mel_spectrogram_}` (already a line-for-line port of `audio.py:110-157`, mel filters included). If its naive FFT shows up in profiles, swap in `realfft` behind the same function signature — golden test stays identical.

### 4.3 Tokenizer: `tiktoken-rs` CoreBPE with OpenAI's own vocab files

tiktoken's engine is Rust already; `tiktoken_rs::CoreBPE::new(ranks, specials, pattern)` accepts a custom vocabulary. We embed `multilingual.tiktoken` / `gpt2.tiktoken` (~800 KB each, `include_bytes!`) and build the special-token table programmatically exactly as `tokenizer.py:330-363` does (languages, `<|0.00|>`…`<|30.00|>` = 1501 timestamp tokens, etc.). The BPE regex has a negative lookahead (`\s+(?!\S)`) — handled by `fancy-regex`, which tiktoken-rs uses internally.

Port surface: `sot_sequence`, `non_speech_tokens` (`tokenizer.py:242-275`), `split_to_word_tokens` / `split_tokens_on_unicode` (U+FFFD replacement-char trick, `tokenizer.py:286-309`). All pure logic, all golden-tested.

### 4.4 Decoding: same shape as `decoding.py`, Rust idioms

```rust
pub trait LogitFilter { fn apply(&self, logits: &mut Tensor, tokens: &TokenBatch) -> Result<()>; }
// SuppressBlank, SuppressTokens, ApplyTimestampRules — direct ports of decoding.py:423-505

pub enum TokenDecoder {
    Greedy { temperature: f32, eot: u32 },
    BeamSearch { beam_size: usize, patience: f32, /* finished: Vec<HashMap<Vec<u32>, f64>> */ },
}

pub struct DecodingTask<'m> { /* model, tokenizer, options, kv_cache, filters */ }
impl DecodingTask<'_> { pub fn run(&mut self, mel: &Tensor) -> Result<Vec<DecodingResult>>; }
```

- `GreedyDecoder` first (covers default CLI usage: greedy + temperature fallback).
- `BeamSearchDecoder` (`decoding.py:301-404`) second: `HashMap<Vec<u32>, f64>` for finished sequences, KV reorder via `index_select`. This is the fiddliest correctness surface in the whole port — it gets the densest fixtures (§6 L3).
- Sampling at t>0 uses seeded `rand` (`StdRng`) so tests are reproducible; note the Python reference is *not* seed-compatible, so t>0 paths are tested by invariants + score distributions, not token equality.

### 4.5 Transcribe loop

Direct port of `transcribe.py:184-508`: seek/clip arithmetic, temperature fallback ladder `(0.0, 0.2, …, 1.0)`, `compression_ratio` via `flate2` zlib (must match Python's `zlib.compress` level 6 default — golden-tested), no-speech skip, `condition_on_previous_text` prompt window, and (Phase 4) the hallucination-silence heuristics. Progress bar via `indicatif`. Every magic number gets a named const with a comment pointing at the Python source line.

---

## 5. TDD strategy

Yes — this project is unusually well-suited to TDD because the reference implementation is sitting in `../whisper` and can emit a golden fixture for every layer of the stack. **Rule: no module lands without a failing fixture test first.** `tools/gen_fixtures.py` (already written) produces everything below into `tests/fixtures/`; tensors as `.npy`/`.npz` (candle reads npy/npz natively), scalars/tokens as JSON.

### 5.1 The test pyramid

| Level | What | Fixture | Pass criterion |
|---|---|---|---|
| **L0 — pure logic** | tokenizer encode/decode, special-token ids, `sot_sequence`, `non_speech_tokens`, word splitting; `pad_or_trim`; compression ratio; srt/vtt formatting | `tokenizer_goldens.json`, `misc_goldens.json` | exact equality |
| **L1 — numerics** | log-mel of `jfk.flac`; mel filterbank matrix; encoder output (tiny, fp32); first-step decoder logits; `detect_language` probs | `mel_jfk.npy`, `encoder_out_tiny.npy`, `logits_step0_tiny.npy`, `lang_probs.json` | tolerance (§5.2) |
| **L2 — algorithms** | DTW vs `timing.dtw_cpu` on seeded random matrices; `median_filter`; `ApplyTimestampRules` logit masks on crafted token states | `dtw_cases.npz`, `timestamp_rule_cases.json` | exact (DTW paths) / tolerance (masks) |
| **L3 — integration** | greedy decode tokens for `jfk.flac` (tiny + base, fp32, t=0); beam-5 decode; full `transcribe()` segment JSON; word-timestamps JSON | `decode_greedy_tiny.json`, `transcribe_tiny.json`, `word_ts_tiny.json` | text exact; tokens expected-exact (§5.2 caveat); timestamps ±0.02s |
| **L4 — properties** | proptest: tokenizer round-trip on arbitrary UTF-8; timestamp rules invariants (pairs, non-decreasing); pad_or_trim shape laws; DTW path validity (monotone, boundary) | — | invariants hold |

### 5.2 Tolerance policy (candle-vs-PyTorch will never be bitwise equal)

| Signal | Tolerance | Rationale |
|---|---|---|
| mel spectrogram | `max_abs ≤ 1e-4` | different FFTs, same math |
| encoder output | `mean_abs ≤ 5e-3` **and** cosine sim `≥ 0.9995` | GEMM accumulation order |
| step-0 logits | same as encoder | |
| greedy tokens (t=0, fp32) | **expected exact**; failure = investigate, not auto-relax | a near-tie argmax flip is possible but rare on clean audio; if one appears, the test compares normalized *text* and asserts `avg_logprob` within 0.02 |
| segment timestamps | ±1 frame (0.02 s) | |

### 5.3 Red/green cadence per phase

Each phase (§7) opens by un-`#[ignore]`-ing its fixture tests (red), closes when they're green. Fast tests (`cargo test`, <30 s, no network) run the committed fixtures; tests that need real weights (`tiny` = 151 MB safetensors from HF) are `#[ignore]` + `--features model-tests`, cached in `~/.cache/huggingface` by `hf-hub`. CI: fast suite on every push, model suite nightly.

---

## 6. Benchmarks — Python baseline as the target

`tools/bench_python.py` (already written) produces the reference numbers on this machine (Apple M4 Max, 36 GB). Methodology: `jfk.flac` (11 s) and a 60 s concat, greedy (`temperature=0`, `beam_size=None`), fp32 CPU and MPS, best-of-3 after 1 warm-up, model load time excluded. Metric: **RTF = audio_seconds / wall_seconds** (higher is better).

### Python baseline (measured 2026-07-09, Apple M4 Max, torch 2.8.0 — `tests/fixtures/bench_python*.json`)

| model | device | dtype | 11 s clip: wall / RTF | 60 s clip: wall / RTF |
|---|---|---|---|---|
| tiny | cpu | fp32 | 0.125 s / **88×** | 0.466 s / **129×** |
| base | cpu | fp32 | 0.228 s / **48×** | 1.004 s / **60×** |
| tiny | mps | fp16 | 1.73 s / 6.4× ⚠ | 3.52 s / 17× ⚠ |
| base | mps | fp16 | 0.10 s / 114× ⚠ | 3.00 s / 20× ⚠ |

⚠ **MPS numbers are not a usable baseline.** Two findings from the run:
1. Stock openai-whisper **crashes outright on MPS** (`alignment_heads` is a sparse buffer; sparse tensors can't move to MPS). `bench_python.py` densifies the buffer before `.to("mps")` to get it running at all. This validates §4.1's choice to carry alignment heads as a plain index list from HF `generation_config.json`.
2. Even patched, **fp16-on-MPS transcripts are garbage** ("Are. Are. Are. …" repetition loops; the anomalously fast base/11s run emitted almost no tokens). So the Python reference is effectively CPU-only on this machine — **the Rust Metal target below is measured against the Python *CPU* baseline**, and beating it while producing *correct* transcripts is already a win over PyTorch here.

### Rust targets

| Milestone | Target |
|---|---|
| Phase 2 exit (correctness first) | ≥ 0.5× Python CPU RTF, same transcripts |
| Phase 5 (candle `accelerate` feature, KV-cache & mel tuned) | ≥ 1.0× Python CPU RTF |
| Phase 5 Metal (`--features metal`, f16) | ≥ 1.0× Python **CPU** RTF *with correct transcripts* (Python MPS is broken — see baseline note) |
| Stretch (quantized q5) | ≥ 2× Python CPU RTF |

Rust harness: `criterion` micro-benches (mel, tokenizer encode, DTW, one decoder step) + `hyperfine` end-to-end vs the venv CLI, driven by a `justfile` so the comparison is one command: `just bench-compare`.

---

## 7. Phases & exit criteria

| Phase | Scope | Exit criterion (tests green) |
|---|---|---|
| **0. Scaffold** | workspace, CI, fixtures generated & committed, `hub.rs` downloads tiny safetensors | fixture files exist; `cargo test` runs L0 skeletons |
| **1. Foundations** | `audio.rs`, `tokenizer.rs`, `model.rs` (vendored), weight loading | **L0 + L1 green** (mel, encoder, logits parity) |
| **2. Greedy transcription** | `decode.rs` (greedy + all logit filters), `transcribe.rs` core loop, txt writer, minimal CLI | **L2 + L3-greedy green**: `whisper-cli jfk.flac` == Python transcript |
| **3. Full decoding** | beam search, temperature fallback, prompts/prefix, language detection, all writers, full CLI flags | **L3-beam + transcribe-JSON green** |
| **4. Word timestamps** | cross-QK capture, median filter, DTW, punctuation merge, hallucination heuristics | **L3 word-ts green** (±0.02 s) |
| **5. Performance & polish** | Metal f16, `accelerate`, quantized models, criterion suite, WER eval harness (+normalizers) | bench targets in §6; WER within 0.1 abs of Python on LibriSpeech test-clean subset |
| **6. Stretch** | streaming/chunked API, `.pt` loading, WASM feasibility spike | — |

Rough effort: Phases 0–2 ≈ 2 weeks, 3–4 ≈ 2–3 weeks, 5 ≈ 1–2 weeks.

---

## 8. Dependencies (whisper-core)

```toml
candle-core = { version = "0.9", features = [] }   # +accelerate on macOS, +metal/+cuda features
candle-nn = "0.9"
candle-transformers = "0.9"     # mel/audio utils; model.rs vendored from here (Apache-2.0/MIT)
tiktoken-rs = "0.7"
fancy-regex = "0.14"
symphonia = { version = "0.5", features = ["flac", "mp3", "isomp4", "aac"] }
rubato = "0.16"
hf-hub = "0.4"
flate2 = "1"
serde / serde_json / clap / anyhow / thiserror / indicatif / rand
# dev: proptest, criterion, approx
```
(Versions indicative — pin at Phase 0.)

## 9. Risks

| Risk | Mitigation |
|---|---|
| Greedy argmax flips on near-tie logits → transcript diverges | fp32 parity mode; text-level fallback assertion + logprob tolerance (§5.2); fixtures on two model sizes |
| Beam-search subtle bugs (KV reorder, patience) | densest fixture coverage; L4 invariants; diff beam *score tables* not just outputs |
| candle mel/FFT slower than torch | swap `realfft` behind stable interface; mel is <5% of runtime anyway |
| HF safetensors ≠ OpenAI .pt numerics | same weights, re-serialized; L1 tolerance catches drift |
| MPS/Metal f16 numeric drift | parity suite is CPU-fp32 only; Metal validated by WER, not tensors |
| tiktoken-rs API can't express Whisper specials | it can (CoreBPE::new with custom specials); fallback: vendor the ~300-line BPE core |
| Crate-name collision (`whisper-rs` taken) | publish as `whisper-candle` |
