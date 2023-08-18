use std::f64::consts::PI;

use ndarray::{Array, Array2, Array3};

use crate::{
    config::{BOX_SIZE, N_PARTICLES},
    fourier::{self, sample_freq},
};

pub fn initial_conditions(mut density: Array3<f64>) -> (Array2<f64>, Array2<f64>) {
    let positions: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));
    let velocities: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));

    let mut density_k: Array3<f64> = density.clone();
    fourier::forward(&mut density_k);

    for i in 0..=2 {}

    (positions, velocities)
}

fn potential_k(density_k: Array3<f64>) {
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let kx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ky = kx.clone();
    let kz = kx.clone();

    // let resother = Array::from_iter(kx.iter().cartesian_product(ky).cartesian_product(kz));
}
