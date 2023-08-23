use std::f64::{consts::PI, INFINITY};

use crate::{
    config::{DIV_BY_ZERO, N_CELLS},
    meshgrid::Meshgrid3,
};
use ndarray::{s, Array, Array1, Array3};
use rustfft::{
    num_complex::{Complex, Complex64},
    FftPlanner,
};

// TODO use ofuton
pub fn forward(a: &Array3<Complex64>) -> Array3<Complex64> {
    let (x_dim, y_dim, z_dim) = a.dim();

    // Should be a cube matrix
    assert!(x_dim == y_dim && y_dim == z_dim);

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(z_dim);
    let mut res: Array3<Complex64> =
        Array::from_shape_simple_fn((x_dim, y_dim, z_dim), || Complex { re: 0., im: 0. });

    for x in 0..x_dim {
        for y in 0..y_dim {
            let b = a.slice(s![x, y, ..]);
            let mut buf = b.as_slice().unwrap().to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![x, y, ..]).assign(&buf);
        }
    }

    for x in 0..x_dim {
        for z in 0..z_dim {
            let b = res.slice(s![x, .., z]);
            let mut buf = b.to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![x, .., z]).assign(&buf);
        }
    }

    for y in 0..y_dim {
        for z in 0..z_dim {
            let b = res.slice(s![.., y, z]);
            let mut buf = b.to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![.., y, z]).assign(&buf);
        }
    }
    res
}

pub fn inverse(a: &Array3<Complex64>) -> Array3<Complex64> {
    let (x_dim, y_dim, z_dim) = a.dim();

    // Should be a cube matrix
    assert!(x_dim == y_dim && y_dim == z_dim);

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_inverse(z_dim);
    let mut res: Array3<Complex64> =
        Array::from_shape_simple_fn((x_dim, y_dim, z_dim), || Complex { re: 0., im: 0. });

    for x in 0..x_dim {
        for y in 0..y_dim {
            let b = a.slice(s![x, y, ..]);
            let mut buf = b.as_slice().unwrap().to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![x, y, ..]).assign(&buf);
        }
    }

    for x in 0..x_dim {
        for z in 0..z_dim {
            let b = res.slice(s![x, .., z]);
            let mut buf = b.to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![x, .., z]).assign(&buf);
        }
    }

    for y in 0..y_dim {
        for z in 0..z_dim {
            let b = res.slice(s![.., y, z]);
            let mut buf = b.to_vec();
            fft.process(&mut buf);
            let buf: Array1<Complex<f64>> = buf.into();
            res.slice_mut(s![.., y, z]).assign(&buf);
        }
    }
    res
}

pub fn sample_freq(n: &usize) -> Vec<f64> {
    let len = match n % 2 {
        0 => n / 2,
        _ => (n - 1) / 2 + 1,
    };
    let mut res: Vec<f64> = (0..len)
        .chain(0..(len - n % 2))
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

    use crate::{
        config::{BOX_SIZE, N_PARTICLES},
        fourier::ksq_inv,
        meshgrid::Meshgrid3,
    };

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

    #[test]
    fn test() {
        let m = Meshgrid3::new(
            &(0..5).map(|x| x as f64).collect::<Vec<f64>>(),
            &(0..5).map(|x| x as f64).collect::<Vec<f64>>(),
            &(0..5).map(|x| x as f64).collect::<Vec<f64>>(),
        );
        println!("{:?}", m.grid);
    }
    #[test]
    fn check_3() {
        let s = sample_freq(&3);
        let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
        println!("{:?}", s.iter().map(|x| x * scale).collect::<Vec<f64>>());
    }
    #[test]
    fn get_fourier_grid() {
        println!("{:?}", ksq_inv());
    }
}
