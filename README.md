# whisper-candle

**Native [OpenAI Whisper](https://github.com/openai/whisper) in pure Rust. No Python, no PyTorch, no C/C++ bindings.**

whisper-candle is a from-scratch port of Whisper's full inference pipeline to
Rust on [candle](https://github.com/huggingface/candle). One static binary
replaces the reference stack (Python + PyTorch + ffmpeg + numba + tiktoken):
audio decoding, log-mel spectrogram, encoder/decoder, greedy & beam-search
decoding, word-level timestamps, and subtitle writers — all native.

It is not a wrapper: unlike `whisper-rs` (whisper.cpp bindings), there is no C
or C++ in the dependency tree.

## Why

- **Zero-dependency deploys** — embed speech-to-text in a Rust service or ship
  a single binary; no Python runtime or conda environment to drag along.
- **Verified parity, not "inspired by"** — developed test-first against golden
  fixtures generated from the reference implementation: greedy **and** beam
  decoding are token-exact vs PyTorch on the test set, word timestamps within
  0.1 s ([DESIGN.md](DESIGN.md) documents the test pyramid and tolerances).
- **Runs everywhere candle runs** — CPU (Accelerate/BLAS), Metal (correct
  where torch MPS is broken), CUDA behind a feature flag.

## Quick start

```bash
cargo run --release --features accelerate -p whisper-candle-cli -- \
    audio.mp3 --model base --output-format srt -o out/
```

Models (`tiny` … `large-v3`, `turbo`, `.en` variants) download automatically
from the Hugging Face Hub as safetensors. Useful flags:

```bash
--device metal              # Apple GPU
--quantization q4k          # quantize locally to GGUF once (turbo: 0.5 GB vs 1.6 GB)
--word-timestamps           # per-word timing in the json output
--task translate --language ja
```

As a library:

```rust
let device = whisper_core::device("cpu")?;
let mut model = whisper_core::load_model("base", &device)?;
let result = whisper_core::transcribe_file(&mut model, "audio.wav", &Default::default())?;
println!("{}", result.text);
```

## Status

Transcription is at parity with the Python reference (see DESIGN.md for the
current phase); remaining work is performance tuning — CPU decoding currently
runs ~10–30× real-time on Apple Silicon, within ~4× of PyTorch CPU.

Python appears in this repo only to *test* the port: `tools/` regenerates the
golden fixtures and baseline benchmarks from a reference checkout. It is never
needed to build or run whisper-candle.

```bash
cargo test -p whisper-candle-core                # fast golden tests, no network
cargo test -p whisper-candle-core --release --features accelerate \
    --test model_goldens -- --ignored            # parity suite (downloads whisper-tiny)
```

## Provenance

This port was generated with [Claude Code](https://claude.com/claude-code)
(Claude Fable 5), guided by golden-fixture TDD against openai/whisper.

## License

MIT. `crates/whisper-core/src/nn.rs` derives from candle-transformers
(Apache-2.0/MIT); tokenizer vocabularies and mel filterbanks are from
openai/whisper (MIT).
