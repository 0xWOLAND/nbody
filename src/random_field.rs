use std::f64::INFINITY;

use ndarray::{s, Array, Array3};
use rand_distr::{num_traits::Pow, Distribution, Normal};
use rustfft::num_complex::Complex64;

use crate::{
    config::N_CELLS,
    fourier::{forward, fourier_grid, inverse, sample_freq},
    meshgrid::Meshgrid3,
};

pub fn random_field(power: f64, amplitude: f64) -> Array3<f64> {
    let gaussian_dist = Normal::new(0.0, 1.0).unwrap();
    let mut seed = Array::from_shape_simple_fn((N_CELLS, N_CELLS, N_CELLS), || Complex64 {
        re: gaussian_dist.sample(&mut rand::thread_rng()),
        im: 0.0,
    });

    forward(&mut seed);

    let v_k = sample_freq(&N_CELLS)
        .iter()
        .map(|x| (N_CELLS as f64) * x)
        .collect();
    let mut knorms: Array3<f64> = Meshgrid3::new(&v_k, &v_k, &v_k)
        .pow(2)
        .sum()
        .map(|x| x.sqrt());

    if power > 0. {
        knorms.slice_mut(s![0, 0, 0]).fill(INFINITY);
    }

    if power < 3. {
        let norms_real_mid = (N_CELLS as f64 / 2.).floor() as usize;
        knorms
            .slice_mut(s![norms_real_mid, norms_real_mid, norms_real_mid])
            .fill(INFINITY);
    }

    let power_spectrum: Array3<f64> = knorms.map(|x| amplitude * x.pow(-power));
    let power_spectrum_sqrt: Array3<f64> = power_spectrum.map(|x| x.sqrt());

    let realization_k: Array3<Complex64> = Array3::from_shape_vec(
        (N_CELLS, N_CELLS, N_CELLS),
        Array::from_iter(seed.iter().zip(power_spectrum_sqrt).map(|(a, b)| a * b)).to_vec(),
    )
    .unwrap();
    let mut realization_real = realization_k.clone();
    inverse(&mut realization_real);
    realization_real.map(|x| x.re)
    // seed.map(|x| x.re)
}

#[cfg(test)]
mod tests {
    use image::RgbImage;
    use rand::Rng;

    use crate::config::N_CELLS;

    use super::random_field;

    #[test]
    fn generate_random_field() {
        let amp = 3.685;
        let power = 0.845;

        // println!("{:?}", random_field(power, amp));
        let image = RgbImage::from_raw(
            N_CELLS as u32,
            N_CELLS as u32,
            random_field(power, amp)
                .into_raw_vec()
                .iter()
                .map(|x| (*x * 100.) as u8)
                .collect(),
        )
        .unwrap();
        image.save("./out.png");
    }
}
