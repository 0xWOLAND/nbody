use ndarray::{s, Array, Array1, Array2, Array3};
use rand::Rng;
use rustfft::num_complex::{Complex, Complex64, ComplexFloat};
use std::{
    f64::{consts::PI, MIN_POSITIVE},
    ops::Div,
};

use crate::{
    config::{A_INIT, BOX_SIZE, DIV_BY_ZERO, N_CELLS, N_PARTICLES},
    cosmology::*,
    fourier::{self, inverse, sample_freq},
    meshgrid::Meshgrid3,
};

pub fn initial_conditions(density: Array3<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut positions: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));
    let mut velocities: Array2<f64> = Array2::zeros((3, N_PARTICLES.pow(3)));

    let density_k: Array3<Complex64> = density.clone().map(|x| Complex { re: *x, im: 0. });
    let density_k: Array3<Complex64> = fourier::forward(&density_k);

    for i in 0..=2 {
        let potential: Array3<Complex64> = potential_k(&density_k);
        let (_positions, _velocities): (Array1<f64>, Array1<f64>) = zeldovich(potential, i);
        positions.slice_mut(s![i, ..]).assign(&_positions);
        velocities.slice_mut(s![i, ..]).assign(&_velocities);
    }

    (positions, velocities)
}

fn potential_k(density_k: &Array3<Complex64>) -> Array3<Complex64> {
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let kx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ky = kx.clone();
    let kz = kx.clone();

    let [lx, ly, lz]: [Array3<f64>; 3] = Meshgrid3::new(&kx, &ky, &kz).pow(2).get();
    let laplace: Array3<Complex64> = (-(lx + ly + lz)).map(|x| Complex {
        re: {
            if (*x).abs() > 0. {
                *x
            } else {
                1e-10
            }
        },
        im: 0.,
    });

    Array3::from_shape_vec(
        (N_PARTICLES, N_PARTICLES, N_PARTICLES),
        Array::from_iter(density_k.iter().zip(laplace).map(|(a, b)| *a / b)).to_vec(),
    )
    .unwrap()
}

fn zeldovich(potential_k: Array3<Complex64>, direction: usize) -> (Array1<f64>, Array1<f64>) {
    let df = displacement_field_real(potential_k, direction);
    (
        approximate_positions(&df, direction),
        approximate_velocity(&df),
    )
}

// TODO use last block hash/contribution
fn approximate_positions(displacement_field: &Array1<f64>, direction: usize) -> Array1<f64> {
    let linear_growth_factor = D_t(A_INIT);
    let mass_resolution = N_CELLS as f64 / N_PARTICLES as f64;
    let xs = (Array::linspace(0., N_CELLS as f64 - mass_resolution, N_PARTICLES) + 0.5).to_vec();
    let ys = (Array::linspace(0., N_CELLS as f64 - mass_resolution, N_PARTICLES) + 0.5).to_vec();
    let zs = (Array::linspace(0., N_CELLS as f64 - mass_resolution, N_PARTICLES) + 0.5).to_vec();

    let grid = Meshgrid3::new(&xs, &ys, &zs).get();
    let positions = Array::from_iter(grid.get(direction).unwrap())
        .map(|x| *x + rand::thread_rng().gen_range((-2.)..(2.)))
        .to_vec();

    // Zel'dovich Approximation causes a perturbation
    let positions: Array1<f64> = Array::from_iter(
        positions
            .iter()
            .zip(displacement_field)
            .map(|(a, b)| (a + linear_growth_factor * b).rem_euclid(N_CELLS as f64)),
    );
    assert!(
        positions.len() == N_PARTICLES.pow(3),
        "{} != {}",
        positions.len(),
        N_PARTICLES.pow(3)
    );

    positions
}

fn approximate_velocity(displacement_field: &Array1<f64>) -> Array1<f64> {
    let dt_0 = D_t(A_INIT);
    let h = hubble_constant(A_INIT);
    let f = expansion_factor(A_INIT);
    displacement_field.map(|x| A_INIT * dt_0 * h * f * x)
}

fn displacement_field_k(potential_k: Array3<Complex64>, direction: usize) -> Array3<Complex<f64>> {
    let resolution = N_CELLS as f64 / N_PARTICLES as f64;
    let scale = 2. * PI * (N_PARTICLES as f64 / BOX_SIZE as f64);
    let lx: Vec<f64> = sample_freq(&N_PARTICLES)
        .iter()
        .map(|x| x * scale)
        .collect();
    let ly = lx.clone();
    let lz = lx.clone();

    let mesh = Meshgrid3::new(&lx, &ly, &lz).get();
    let l_direction: &Array3<f64> = mesh.get(direction).unwrap();
    (l_direction * potential_k).map(|x| -1. * resolution * *x * Complex::new(1., 1.))
}

fn displacement_field_real(potential_k: Array3<Complex64>, direction: usize) -> Array1<f64> {
    let force_resolution = N_CELLS as f64 / BOX_SIZE as f64;
    let df_k: Array3<Complex64> = displacement_field_k(potential_k, direction);
    let df_real: Array3<Complex64> = inverse(&df_k);
    Array::from_iter(df_real.map(|x| x.re * force_resolution))
}
