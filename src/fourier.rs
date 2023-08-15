use std::f64::consts::PI;

use ndarray::{s, Array, Array3};

use crate::config::N_CELLS;

fn sample_freq(n: &usize) -> Vec<f64> {
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

pub fn fourier_grid() -> Array3<f64> {
    let scale = 2. * PI;
    let binding = sample_freq(&N_CELLS);
    let samples: Vec<f64> = binding.iter().map(|x| scale * x).collect();
    let n = samples.len();

    let mut fgrid: Array3<f64> = Array3::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let k_sq: f64 = [i, j, k]
                    .map(|x| f64::sin(samples[x] / 2.).powi(2))
                    .iter()
                    .sum();
                fgrid.slice_mut(s![i, j, k,]).fill(k_sq.recip());
            }
        }
    }
    fgrid
}
#[cfg(test)]
mod tests {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::sync::Arc;

    use crate::fourier::sample_freq;

    #[test]
    fn test() {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(4);

        let mut buffer = vec![
            Complex {
                re: 8.0f32,
                im: 0.0,
            },
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex {
                re: 8.0f32,
                im: 0.0,
            },
            Complex {
                re: 8.0f32,
                im: 0.0,
            },
        ];
        fft.process(&mut buffer);

        let fft_clone = Arc::clone(&fft);
        println!("{:?}", buffer.to_vec());
    }

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
