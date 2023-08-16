use ndarray::Array3;

use crate::config::{N_CELLS, OMEGA_M0};
use crate::fourier::*;

pub fn potential(density: Array3<f64>, fgrid: Array3<f64>, t: f64) -> Array3<f64> {
    let grid = density_k(density);
    let grid = potential_k(grid, fgrid, t);
    potential_real(grid)
}

fn density_k(mut density: Array3<f64>) -> Array3<f64> {
    forward(&mut density);
    density
}

fn potential_k(density: Array3<f64>, fgrid: Array3<f64>, t: f64) -> Array3<f64> {
    (-3. * OMEGA_M0 / 8.0) as f64 / t * density * fgrid
}

fn potential_real(mut potential_k: Array3<f64>) -> Array3<f64> {
    inverse(&mut potential_k);
    potential_k
}
