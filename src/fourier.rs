use std::f64::{consts::PI, INFINITY};

use crate::{
    config::{DIV_BY_ZERO, N_CELLS},
    meshgrid::Meshgrid3,
};
use ndarray::{s, Array, Array1, Array3};
use ndrustfft::{ndfft, ndifft, FftHandler};
use rustfft::{num_complex::Complex, FftPlanner};

pub fn forward(a: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
    let (nx, ny, nz) = a.dim();

    let mut vhat: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));

    let mut handler_ax0 = FftHandler::<f64>::new(nx);
    let mut handler_ax1 = FftHandler::<f64>::new(ny);
    let mut handler_ax2 = FftHandler::<f64>::new(nz);

    let mut work2: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));
    let mut work1: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));

    ndfft(&a, &mut work2, &mut handler_ax2, 2);
    ndfft(&work2, &mut work1, &mut handler_ax1, 1);
    ndfft(&work1, &mut vhat, &mut handler_ax0, 0);

    vhat
}

pub fn inverse(a: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
    let (nx, ny, nz) = a.dim();

    let mut vhat: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));

    let mut handler_ax0 = FftHandler::<f64>::new(nx);
    let mut handler_ax1 = FftHandler::<f64>::new(ny);
    let mut handler_ax2 = FftHandler::<f64>::new(nz);

    let mut work2: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));
    let mut work1: Array3<Complex<f64>> = Array3::zeros((nx, ny, nz));

    ndifft(&a, &mut work1, &mut handler_ax0, 0);
    ndifft(&work1, &mut work2, &mut handler_ax1, 1);
    ndifft(&work2, &mut vhat, &mut handler_ax2, 2);

    vhat
}

pub fn sample_freq(n: &usize) -> Vec<f64> {
    let len = match n % 2 {
        0 => n / 2,
        _ => (n - 1) / 2 + 1,
    };
    let mut res: Vec<f64> = (0..len)
        .chain(0..(len - n.rem_euclid(2)))
        .map(|x| x as f64)
        .collect::<Vec<f64>>();

    res[len..].reverse();
    res[len..].iter_mut().for_each(|x| *x = -1. * (*x + 1.));
    res.iter().map(|x| *x / (*n as f64)).collect::<Vec<f64>>()
}

pub fn fourier_grid() -> Meshgrid3 {
    let scale = 2. * PI;
    let binding = sample_freq(&N_CELLS);
    let samples: Vec<f64> = binding.iter().map(|x| scale * x).collect();

    Meshgrid3::new(&samples, &samples, &samples)
}

pub fn ksq_inv() -> Array3<f64> {
    let mg = fourier_grid();
    let mg = (mg / 2.).sin().pow(2);
    let [kx, ky, kz]: [Array3<f64>; 3] = mg.get();
    (kx + ky + kz).map(|x| if *x < DIV_BY_ZERO { 0. } else { x.recip() })
}
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use ndarray::{Array, Array3, Zip};
    use ndrustfft::Complex;
    use rustfft::num_complex::ComplexFloat;

    use crate::{
        config::{BOX_SIZE, N_PARTICLES},
        fourier::{inverse, ksq_inv},
        meshgrid::Meshgrid3,
    };

    use super::{forward, sample_freq};

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
    #[test]
    fn check_fftn() {
        let (nx, ny, nz) = (3, 3, 3);

        let v = (0..(nx * ny * nz))
            .map(|x| Complex::new(x as f64, 0.))
            .collect::<Vec<Complex<f64>>>();

        let x: Array3<Complex<f64>> = Array3::from_shape_vec((nx, ny, nz), v).unwrap();
        let y: Array3<Complex<f64>> = inverse(&forward(&x));

        Zip::from(&x).and(&y).for_each(|&x, &y| {
            if (x.re - y.re).abs() > 1e-4 || (x.im - y.im) > 1e-4 {
                panic!("Large difference in values, got {} expected {}", x, y);
            }
        })
    }
    #[test]
    fn get_fourier_grid() {
        println!("{:?}", ksq_inv());
    }
}
