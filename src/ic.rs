use std::{f64::consts::PI, ops::Div};

use ndarray::{Array, Array2, Array3};
use rustfft::num_complex::Complex;

use crate::{
    config::{BOX_SIZE, DIV_BY_ZERO, N_CELLS, N_PARTICLES},
    fourier::{self, sample_freq},
    meshgrid::Meshgrid3,
};

pub fn initial_conditions(density: Array3<f64>) -> (Array2<f64>, Array2<f64>) {
    let positions: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));
    let velocities: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));

    let density_k: Array3<f64> = density.clone();
    fourier::forward(&mut density_k.map(|x| Complex { re: *x, im: 0. }));

    for i in 0..=2 {
        let potential = potential_k(&density_k);
    }

    (positions, velocities)
}

fn potential_k(density_k: &Array3<f64>) -> Array3<f64> {
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let kx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ky = kx.clone();
    let kz = kx.clone();

    let [lx, ly, lz] = Meshgrid3::new(&kx, &ky, &kz).pow(2).get();
    let laplace: Array3<f64> = -(lx + ly + lz);

    density_k.div(laplace).map(|x| x.min(DIV_BY_ZERO.recip()))
}

fn displacement_field_k(potential_k: Array3<f64>, direction: usize) -> Array3<Complex<f64>> {
    let resolution = N_CELLS as f64 / N_PARTICLES as f64;
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let lx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ly = lx.clone();
    let lz = lx.clone();

    let mesh = Meshgrid3::new(&lx, &ly, &lz).get();
    let l_direction = mesh.get(direction).unwrap();
    (l_direction * potential_k).map(|x| -1. * resolution * *x * Complex::new(0., 1.))
}

fn displacement_field_real(potential_k: Array3<f64>, direction: usize) {
    let force_resolution = N_CELLS as f64 / BOX_SIZE as f64;
    let df_k = displacement_field_k(potential_k, direction);
}
