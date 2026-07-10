#!/usr/bin/env python3
"""Generate golden test fixtures for the Rust port from the reference Python whisper.

Usage:
    python tools/gen_fixtures.py [--whisper-repo ../whisper] [--out tests/fixtures] [--models tiny,base]

Everything runs fp32 on CPU for determinism. Tensors are saved as .npy/.npz
(candle reads these natively); scalars/tokens as JSON.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"  wrote {path}")


def save_npy(path: Path, tensor):
    arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    np.save(path, arr)
    print(f"  wrote {path} shape={arr.shape} dtype={arr.dtype}")


def gen_tokenizer_goldens(out: Path):
    from whisper.tokenizer import get_tokenizer

    tok = get_tokenizer(True, num_languages=100, language="en", task="transcribe")

    cases = [
        "Hello, world!",
        " And so my fellow Americans, ask not what your country can do for you",
        "日本語のテスト",
        "¿Dónde está la biblioteca?",
        "  leading spaces and\ttabs\n",
        "123 456.789 -- $10,000!",
        "naïve façade — em–dash",
        "🦀 Rust + 🐍 Python",
        "",
    ]
    encode_cases = [{"text": t, "tokens": tok.encode(t)} for t in cases]

    # word splitting goldens (used by word timestamps)
    sample_tokens = tok.encode(" And so my fellow Americans, ask not what your country can do")
    words, word_tokens = tok.split_to_word_tokens(sample_tokens + [tok.eot])

    goldens = {
        "encoding_name": "multilingual",
        "num_languages": 100,
        "eot": tok.eot,
        "sot": tok.sot,
        "sot_prev": tok.sot_prev,
        "sot_lm": tok.sot_lm,
        "no_speech": tok.no_speech,
        "no_timestamps": tok.no_timestamps,
        "timestamp_begin": tok.timestamp_begin,
        "transcribe": tok.transcribe,
        "translate": tok.translate,
        "language_token_en": tok.language_token,
        "sot_sequence_en_transcribe": list(tok.sot_sequence),
        "sot_sequence_including_notimestamps": list(tok.sot_sequence_including_notimestamps),
        "all_language_tokens": list(tok.all_language_tokens),
        "all_language_codes": list(tok.all_language_codes),
        "non_speech_tokens": list(tok.non_speech_tokens),
        "encode_cases": encode_cases,
        "split_to_word_tokens": {
            "input_tokens": sample_tokens + [tok.eot],
            "words": words,
            "word_tokens": word_tokens,
        },
        "special_tokens": tok.special_tokens,
    }
    save_json(out / "tokenizer_goldens.json", goldens)


def gen_misc_goldens(out: Path):
    from whisper.utils import compression_ratio, format_timestamp

    texts = [
        "hello hello hello hello hello hello hello hello",
        "The quick brown fox jumps over the lazy dog.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "And so my fellow Americans, ask not what your country can do for you.",
    ]
    save_json(out / "misc_goldens.json", {
        "compression_ratio": [{"text": t, "ratio": compression_ratio(t)} for t in texts],
        "format_timestamp": [
            {"seconds": s, "formatted": format_timestamp(s)}
            for s in [0.0, 1.5, 61.02, 3599.999, 3661.5]
        ],
    })


def gen_audio_goldens(out: Path, whisper_repo: Path):
    import whisper
    from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

    jfk = whisper_repo / "tests" / "jfk.flac"
    shutil.copy(jfk, out / "jfk.flac")

    audio = whisper.load_audio(str(jfk))
    save_npy(out / "audio_jfk_pcm.npy", audio)  # decoded 16k mono f32 pcm

    mel = log_mel_spectrogram(audio, n_mels=80)
    save_npy(out / "mel_jfk.npy", mel)

    mel_padded = pad_or_trim(log_mel_spectrogram(audio, n_mels=80, padding=N_SAMPLES), N_FRAMES)
    save_npy(out / "mel_jfk_padded_3000.npy", mel_padded)

    # mel filterbanks, re-exported for convenience
    filters = np.load(whisper_repo / "whisper" / "assets" / "mel_filters.npz")
    np.savez(out / "mel_filters.npz", **{k: filters[k] for k in filters.files})
    print(f"  wrote {out / 'mel_filters.npz'}")


def gen_dtw_and_medfilt_goldens(out: Path):
    from whisper.timing import dtw_cpu, median_filter

    rng = np.random.default_rng(42)
    arrays = {}
    for i, (n, m) in enumerate([(10, 20), (37, 101), (100, 100), (2, 3)]):
        x = rng.standard_normal((n, m)).astype(np.float64)
        path = dtw_cpu(x)  # (2, path_len)
        arrays[f"dtw_in_{i}"] = x
        arrays[f"dtw_out_{i}"] = path.astype(np.int64)

    torch.manual_seed(42)
    for i, (shape, width) in enumerate([((129,), 7), ((4, 10, 64), 9), ((1, 5, 30), 7)]):
        x = torch.randn(*shape, dtype=torch.float32)
        y = median_filter(x, width)
        arrays[f"medfilt_in_{i}"] = x.numpy()
        arrays[f"medfilt_out_{i}"] = y.numpy()
        arrays[f"medfilt_width_{i}"] = np.array([width])

    np.savez(out / "dtw_medfilt_goldens.npz", **arrays)
    print(f"  wrote {out / 'dtw_medfilt_goldens.npz'}")


def gen_timestamp_rule_goldens(out: Path):
    """Apply ApplyTimestampRules to seeded random logits over crafted token contexts."""
    from whisper.decoding import ApplyTimestampRules
    from whisper.tokenizer import get_tokenizer

    tok = get_tokenizer(True, num_languages=100, language="en", task="transcribe")
    sot_seq = list(tok.sot_sequence)
    sample_begin = len(sot_seq)
    n_vocab = 51866  # large-v3 vocab; tiny/base use 51865 but rules only index special ids

    ts = tok.timestamp_begin
    contexts = {
        "at_sample_begin": sot_seq,
        "after_text": sot_seq + [ts, 50257, 50258],  # <|0.00|> + text tokens
        "after_single_timestamp": sot_seq + [ts, 50257, ts + 25],
        "after_timestamp_pair": sot_seq + [ts, 50257, ts + 25, ts + 25],
        "long_text_run": sot_seq + [ts] + [1000 + i for i in range(10)],
    }

    rule = ApplyTimestampRules(tok, sample_begin, max_initial_timestamp_index=50)
    torch.manual_seed(7)
    arrays, meta = {}, {}
    for i, (name, ctx) in enumerate(contexts.items()):
        logits = torch.randn(1, n_vocab, dtype=torch.float32)
        arrays[f"logits_in_{i}"] = logits.numpy().copy()
        tokens = torch.tensor([ctx])
        rule.apply(logits, tokens)
        arrays[f"logits_out_{i}"] = logits.numpy()
        meta[str(i)] = {"name": name, "tokens": ctx}

    np.savez(out / "timestamp_rules_goldens.npz", **arrays)
    save_json(out / "timestamp_rules_meta.json", {
        "sample_begin": sample_begin, "max_initial_timestamp_index": 50,
        "n_vocab": n_vocab, "cases": meta,
    })


def gen_model_goldens(out: Path, whisper_repo: Path, model_name: str, full_tensors: bool):
    import whisper
    from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from whisper.decoding import DecodingOptions
    from whisper.tokenizer import get_tokenizer

    print(f"loading model {model_name} (cpu fp32)...")
    model = whisper.load_model(model_name, device="cpu")
    model.eval()
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        language="en", task="transcribe")

    jfk = whisper_repo / "tests" / "jfk.flac"
    audio = whisper.load_audio(str(jfk))
    mel = pad_or_trim(log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES), N_FRAMES)

    tag = model_name.replace(".", "_")

    with torch.no_grad():
        feats = model.embed_audio(mel.unsqueeze(0))
        if full_tensors:
            save_npy(out / f"encoder_out_{tag}.npy", feats)

        sot = torch.tensor([list(tok.sot_sequence)])
        logits = model.logits(sot, feats)
        if full_tensors:
            save_npy(out / f"logits_sot_{tag}.npy", logits)

        _, lang_probs = model.detect_language(mel)
        top = sorted(lang_probs.items(), key=lambda kv: -kv[1])[:10]

        greedy = whisper.decode(model, mel, DecodingOptions(
            fp16=False, temperature=0.0, language="en", task="transcribe"))
        beam = whisper.decode(model, mel, DecodingOptions(
            fp16=False, temperature=0.0, language="en", task="transcribe", beam_size=5))

    result = model.transcribe(str(jfk), fp16=False, temperature=0.0, language="en",
                              verbose=None)
    result_wt = model.transcribe(str(jfk), fp16=False, temperature=0.0, language="en",
                                 word_timestamps=True, verbose=None)

    def dec_result(r):
        return {"tokens": list(r.tokens), "text": r.text,
                "avg_logprob": r.avg_logprob, "no_speech_prob": r.no_speech_prob,
                "compression_ratio": r.compression_ratio}

    save_json(out / f"decode_goldens_{tag}.json", {
        "model": model_name,
        "dims": model.dims.__dict__,
        "sot_sequence": list(tok.sot_sequence),
        "detect_language_top10": top,
        "greedy": dec_result(greedy),
        "beam5": dec_result(beam),
        "transcribe": {"text": result["text"], "language": result["language"],
                       "segments": result["segments"]},
        "transcribe_word_timestamps": {"text": result_wt["text"],
                                       "segments": result_wt["segments"]},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-repo", default=str(HERE.parent.parent / "whisper"))
    ap.add_argument("--out", default=str(HERE.parent / "tests" / "fixtures"))
    ap.add_argument("--models", default="tiny,base",
                    help="comma-separated; first model gets full tensor goldens")
    args = ap.parse_args()

    whisper_repo = Path(args.whisper_repo).resolve()
    sys.path.insert(0, str(whisper_repo))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 4)

    print("== tokenizer =="); gen_tokenizer_goldens(out)
    print("== misc =="); gen_misc_goldens(out)
    print("== audio/mel =="); gen_audio_goldens(out, whisper_repo)
    print("== dtw / median filter =="); gen_dtw_and_medfilt_goldens(out)
    print("== timestamp rules =="); gen_timestamp_rule_goldens(out)
    for i, name in enumerate(args.models.split(",")):
        print(f"== model goldens: {name} ==")
        gen_model_goldens(out, whisper_repo, name.strip(), full_tensors=(i == 0))
    print("done.")


if __name__ == "__main__":
    main()
