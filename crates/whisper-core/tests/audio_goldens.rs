//! L1: audio decode + log-mel parity with fixtures from the Python reference.

mod common;
use candle_core::Tensor;
use common::{cosine_similarity, fixtures_dir, max_abs_diff};
use whisper_core::audio::{load_audio, log_mel_spectrogram, N_FRAMES, N_SAMPLES};

fn read_npy_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = Tensor::read_npy(fixtures_dir().join(name)).expect("read npy");
    let shape = t.dims().to_vec();
    let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    (flat, shape)
}

#[test]
fn pcm_decode_matches_ffmpeg() {
    let (expected, _) = read_npy_f32("audio_jfk_pcm.npy");
    let pcm = load_audio(fixtures_dir().join("jfk.flac")).unwrap();
    assert_eq!(pcm.len(), expected.len(), "sample count");
    // jfk.flac is 44.1 kHz stereo, so this exercises downmix + resample.
    // rubato and ffmpeg's swresample use different lowpass filters (and the
    // true delay is fractional), so sample-exact parity is impossible here;
    // require tight RMS agreement instead. Native-16k inputs skip the
    // resampler entirely and stay exact.
    // The true filter delay lands between integer samples for 44.1k->16k
    // (residual ~0.5 sample), which inflates sample-wise RMS on sibilant
    // content; 0.05 reflects that floor, not decoder error.
    let rms = (pcm
        .iter()
        .zip(&expected)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / pcm.len() as f64)
        .sqrt();
    assert!(rms <= 0.05, "pcm rms diff {rms}");
    let diff = max_abs_diff(&pcm, &expected);
    assert!(diff <= 0.35, "pcm max abs diff {diff}");
}

#[test]
fn log_mel_matches_torch_stft() {
    let (expected, shape) = read_npy_f32("mel_jfk.npy");
    let (pcm, _) = read_npy_f32("audio_jfk_pcm.npy");
    let mel = log_mel_spectrogram(&pcm, 80, 0).unwrap();
    assert_eq!(vec![mel.n_mels, mel.n_frames], shape, "mel shape");
    let diff = max_abs_diff(&mel.data, &expected);
    assert!(diff <= 1e-4, "mel max abs diff {diff}");
    let cos = cosine_similarity(&mel.data, &expected);
    assert!(cos >= 0.99999, "mel cosine similarity {cos}");
}

#[test]
fn padded_mel_window_matches_pad_or_trim() {
    let (expected, shape) = read_npy_f32("mel_jfk_padded_3000.npy");
    assert_eq!(shape, vec![80, N_FRAMES]);
    let (pcm, _) = read_npy_f32("audio_jfk_pcm.npy");
    let mel = log_mel_spectrogram(&pcm, 80, N_SAMPLES).unwrap();
    let window = mel.window(0, N_FRAMES, N_FRAMES);
    let diff = max_abs_diff(&window, &expected);
    assert!(diff <= 1e-4, "padded mel max abs diff {diff}");
}
