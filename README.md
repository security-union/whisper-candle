# whisper-candle

Pure-Rust port of [OpenAI Whisper](https://github.com/openai/whisper) built on
[candle](https://github.com/huggingface/candle) — no C/C++ bindings, no
PyTorch. Developed test-first against golden fixtures generated from the
Python reference implementation (see [DESIGN.md](DESIGN.md)).

**Status:** transcription is at parity with PyTorch — greedy *and* beam-search
decoding are token-exact on the test fixtures; word-level timestamps
(`--word-timestamps`) match within 0.1s. Includes temperature-fallback, best_of
sampling, language detection, prompts, and txt/srt/vtt/tsv/json writers.
Remaining work is performance tuning and quantized models (DESIGN.md §7).

## Usage

```bash
# CPU (Accelerate BLAS on macOS)
cargo run --release --features accelerate -p whisper-candle-cli -- \
    audio.mp3 --model base --output-format srt -o out/

# Metal
cargo build --release --features metal
target/release/whisper-candle audio.flac --model large-v3 --device metal
```

Models download automatically from the Hugging Face Hub (safetensors) into
`~/.cache/huggingface` on first use. Supported: `tiny`, `base`, `small`,
`medium`, `large-v1/v2/v3`, `turbo`, and the `.en` variants.

### Quantized models

```bash
whisper-candle audio.mp3 --model turbo --quantization q4k
```

Quantizes the HF f32 weights locally on first use (GGUF cached under
`~/.cache/whisper-candle/`, e.g. turbo q4k ≈ 0.5 GB vs 1.6 GB f32) and runs a
hybrid model: f32 encoder (BLAS is much faster for the big prefill GEMMs) +
quantized decoder. Supported dtypes: `q4_0 q4_1 q5_0 q5_1 q8_0 q2k q3k q4k
q5k q6k`. `q8_0` is near-lossless (matches the f32 transcript on the test
fixtures). Note: on machines with a fast BLAS, quantization trades some speed
for a ~3-6x smaller memory/disk footprint — it's a capacity feature, not a
speed feature.

As a library:

```rust
let device = whisper_core::device("cpu")?;
let mut model = whisper_core::load_model("base", &device)?;
let result = whisper_core::transcribe_file(&mut model, "audio.wav", &Default::default())?;
println!("{}", result.text);
```

## Testing

Golden fixtures are generated from the Python reference into `tests/fixtures/`
(committed). Fast suite (no network):

```bash
cargo test -p whisper-candle-core
```

Model parity suite (downloads whisper-tiny, ~150 MB, once):

```bash
cargo test -p whisper-candle-core --release --features accelerate \
    --test model_goldens -- --ignored
```

To regenerate fixtures or the Python baseline numbers
(`tests/fixtures/bench_python*.json`), see `tools/gen_fixtures.py` and
`tools/bench_python.py` — both run against a checkout of openai/whisper.

## License

MIT. `crates/whisper-core/src/nn.rs` is derived from candle-transformers
(Apache-2.0/MIT); tokenizer vocabularies and mel filterbanks are from
openai/whisper (MIT).
