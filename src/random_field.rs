use std::f64::INFINITY;

use ndarray::{s, Array, Array3};
use rand_distr::{num_traits::Pow, Distribution, Normal};
use rustfft::num_complex::Complex64;

use crate::{
    config::{N_CELLS, N_PARTICLES},
    fourier::{forward, inverse, sample_freq},
    meshgrid::Meshgrid3,
};

pub fn gaussian_random_field(power: f64, amplitude: f64) -> Array3<f64> {
    let gaussian_dist = Normal::new(0.0, 1.0).unwrap();
    let mut seed =
        Array::from_shape_simple_fn((N_PARTICLES, N_PARTICLES, N_PARTICLES), || Complex64 {
            re: gaussian_dist.sample(&mut rand::thread_rng()),
            im: 0.0,
        });

    forward(&mut seed);

    let v_k = sample_freq(&N_PARTICLES)
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
        (N_PARTICLES, N_PARTICLES, N_PARTICLES),
        Array::from_iter(seed.iter().zip(power_spectrum_sqrt).map(|(a, b)| a * b)).to_vec(),
    )
    .unwrap();
    let realization_real: Array3<Complex64> = inverse(realization_k.clone());
    realization_real.map(|x| x.re)
    // seed.map(|x| x.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::array_3_to_image;

    #[test]
    fn generate_random_field() {
        let amp = 3.685;
        let power = 0.845;

        let image = array_3_to_image(
            gaussian_random_field(power, amp).map(|x| (*x * 100.) as u8),
            None,
            None,
        );
        image.save("./out.png");
    }
}
