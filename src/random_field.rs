use ndarray::Array1;

use crate::{
    config::{A_INIT, N_PARTICLES},
    cosmology::D_t,
};

pub fn random_field() {
    let linear_growth_factor = D_t(A_INIT);
}

pub fn gaussian_random_numbers() {
    let x: Array1<f64> = Array1::zeros(N_PARTICLES.pow(3));
    let y: Array1<f64> = Array1::zeros(N_PARTICLES.pow(3));
}
