use std::f64::consts::PI;

use crate::{
    config::{DIV_BY_ZERO, N_CELLS},
    meshgrid::Meshgrid3,
};
use ndarray::{s, Array1, Array3};
use rustfft::{num_complex::Complex, FftPlanner};

pub fn forward(a: &mut Array3<f64>) {
    let (x_dim, y_dim, z_dim) = a.dim();

    // Should be a cube matrix
    assert!(x_dim == y_dim && y_dim == z_dim);

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(z_dim);

    for x in 0..x_dim {
        for y in 0..y_dim {
            let b = a.slice(s![x, y, ..]);
            let mut buf = b
                .map(|x| Complex { re: *x, im: 0.0 })
                .as_slice()
                .unwrap()
                .to_vec();
            fft.process(&mut buf);
            let buf: Array1<f64> = buf.iter().map(|x| (*x).re).collect();
            a.slice_mut(s![x, y, ..]).assign(&buf);
        }
    }
}

pub fn inverse(a: &mut Array3<f64>) {
    let (x_dim, y_dim, z_dim) = a.dim();

    // Should be a cube matrix
    assert!(x_dim == y_dim && y_dim == z_dim);

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(z_dim);

    for x in 0..x_dim {
        for y in 0..y_dim {
            let b = a.slice(s![x, y, ..]);
            let mut buf = b
                .map(|x| Complex { re: *x, im: 0.0 })
                .as_slice()
                .unwrap()
                .to_vec();
            fft.process(&mut buf);
            let buf: Array1<f64> = buf.iter().map(|x| (*x).re).collect();
            a.slice_mut(s![x, y, ..]).assign(&buf);
        }
    }
}

pub fn sample_freq(n: &usize) -> Vec<f64> {
    let len = match n % 2 {
        0 => n / 2,
        _ => (n - 1) / 2 + 1,
    };
    let mut res: Vec<f64> = (0..=len)
        .chain(0..(len - n % 2))
        .map(|x| (x as f64) / (*n as f64))
        .collect::<Vec<f64>>();

    res[len..].reverse();

    res[len..].iter_mut().for_each(|x| *x = -1. * (*x + 1.));

    res
}

// TODO use itertools
pub fn fourier_grid() -> Array3<f64> {
    let scale = 2. * PI;
    let binding = sample_freq(&N_CELLS);
    let samples: Vec<f64> = binding.iter().map(|x| scale * x).collect();

    let mg = Meshgrid3::new(&samples, &samples, &samples);
    let mg = (mg / 2.).sin().pow(2);

    let (kx, ky, kz) = mg.get();
    (kx + ky + kz).map(|x| x.max(DIV_BY_ZERO).recip())
}
#[cfg(test)]
mod tests {
    use super::sample_freq;

    #[test]
    fn fftfreq() {
        {
            let n: usize = 10;
            let ans = sample_freq(&n);
            let expected_result = vec![0.0, 0.1, 0.2, 0.3, 0.4, -1.4, -1.3, -1.2, -1.1, -1.0];
            ans.iter()
                .zip(expected_result)
                .for_each(|(a, b)| assert_eq!(*a, b));
        }
        {
            let n: usize = 11;
            let ans = sample_freq(&n);
            let expected_result = vec![
                0.0,
                0.09090909090909091,
                0.18181818181818182,
                0.2727272727272727,
                0.36363636363636365,
                0.45454545454545453,
                -1.3636363636363638,
                -1.2727272727272727,
                -1.1818181818181819,
                -1.0909090909090908,
                -1.0,
            ];
            ans.iter()
                .zip(expected_result)
                .for_each(|(a, b)| assert_eq!(*a, b));
        }
    }
}
