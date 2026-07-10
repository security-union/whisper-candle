# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/security-union/whisper-candle/compare/whisper-candle-cli-v0.1.1...whisper-candle-cli-v0.1.2) - 2026-07-10

### Fixed

- point repository metadata at the renamed repo

## [0.1.1](https://github.com/security-union/whisper-candle/compare/whisper-candle-cli-v0.1.0...whisper-candle-cli-v0.1.1) - 2026-07-10

### Fixed

- *(cli)* expose --version; release on any unpublished version bump

## [0.1.0](https://github.com/security-union/whisper-candle/releases/tag/whisper-candle-cli-v0.1.0) - 2026-07-10

### Other

- README rewrite, release-plz + CI workflows, crates.io metadata, rustfmt
- Quantized model support: local safetensors->GGUF + hybrid runtime
- Port carry_initial_prompt and hallucination_silence_threshold
- Update README and design doc status for Phases 3-4
- Word timestamps: cross-attention QK capture + DTW alignment (Phase 4)
- README, design doc status update, clippy clean
- Phase 0-2 scaffold: workspace, tokenizer, audio, decode, transcribe, CLI + golden tests
