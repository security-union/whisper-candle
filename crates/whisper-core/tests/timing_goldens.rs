//! L2: DTW + median filter parity with numba/PyTorch fixtures.

mod common;
use candle_core::Tensor;
use common::fixtures_dir;
use std::collections::HashMap;
use whisper_core::timing::{dtw, median_filter_rows};

fn load_npz() -> HashMap<String, Tensor> {
    Tensor::read_npz(fixtures_dir().join("dtw_medfilt_goldens.npz"))
        .unwrap()
        .into_iter()
        .collect()
}

#[test]
fn dtw_matches_numba() {
    let t = load_npz();
    for i in 0..4 {
        let input = &t[&format!("dtw_in_{i}")];
        let (n, m) = (input.dims()[0], input.dims()[1]);
        let x: Vec<f64> = input.flatten_all().unwrap().to_vec1().unwrap();
        let expected: Vec<Vec<i64>> = t[&format!("dtw_out_{i}")].to_vec2().unwrap();

        let (text_indices, time_indices) = dtw(&x, n, m);
        let got_text: Vec<i64> = text_indices.iter().map(|&v| v as i64).collect();
        let got_time: Vec<i64> = time_indices.iter().map(|&v| v as i64).collect();
        assert_eq!(got_text, expected[0], "case {i}: text indices");
        assert_eq!(got_time, expected[1], "case {i}: time indices");
    }
}

#[test]
fn median_filter_matches_pytorch() {
    let t = load_npz();
    for i in 0..3 {
        let input = &t[&format!("medfilt_in_{i}")];
        let width = t[&format!("medfilt_width_{i}")]
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap()[0] as usize;
        let dims = input.dims().to_vec();
        let n_cols = *dims.last().unwrap();
        let mut data: Vec<f32> = input.flatten_all().unwrap().to_vec1().unwrap();
        let expected: Vec<f32> = t[&format!("medfilt_out_{i}")]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        median_filter_rows(&mut data, n_cols, width);
        assert_eq!(data.len(), expected.len(), "case {i}: length");
        let max_diff = data
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff == 0.0, "case {i}: max diff {max_diff}");
    }
}
