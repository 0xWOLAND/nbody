use std::{f64::consts::PI, ops::Div};

use ndarray::{Array, Array2, Array3};

use crate::{
    config::{BOX_SIZE, DIV_BY_ZERO, N_PARTICLES},
    fourier::{self, sample_freq},
    meshgrid::Meshgrid3,
};

pub fn initial_conditions(mut density: Array3<f64>) -> (Array2<f64>, Array2<f64>) {
    let positions: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));
    let velocities: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));

    let mut density_k: Array3<f64> = density.clone();
    fourier::forward(&mut density_k);

    for i in 0..=2 {}

    (positions, velocities)
}

fn potential_k(density_k: Array3<f64>) -> Array3<f64> {
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let kx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ky = kx.clone();
    let kz = kx.clone();

    let (lx, ly, lz) = Meshgrid3::new(&kx, &ky, &kz).pow(2).get();
    let laplace: Array3<f64> = -(lx + ly + lz);

    density_k.div(laplace).map(|x| x.min(DIV_BY_ZERO.recip()))
}
