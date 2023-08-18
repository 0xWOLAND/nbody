use ndarray::Array3;
use rustfft::num_complex::Complex;

use crate::config::OMEGA_M0;
use crate::fourier::*;

pub fn potential(density: Array3<f64>, fgrid: Array3<f64>, t: f64) -> Array3<f64> {
    let grid = density_k(density);
    let grid = potential_k(grid, fgrid, t);
    potential_real(grid)
}

fn density_k(density: Array3<f64>) -> Array3<f64> {
    forward(&mut density.map(|x| Complex { re: *x, im: 0. }));
    density
}

pub fn potential_k(density: Array3<f64>, fgrid: Array3<f64>, t: f64) -> Array3<f64> {
    (-3. * OMEGA_M0 / 8.0) as f64 / t * density * fgrid
}

fn potential_real(potential_k: Array3<f64>) -> Array3<f64> {
    inverse(&mut potential_k.map(|x| Complex { re: *x, im: 0. }));
    potential_k
}
