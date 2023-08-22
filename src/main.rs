use config::*;
use nbody::{
    density::density, fourier::ksq_inv, ic::initial_conditions, integrate::update,
    random_field::gaussian_random_field, utils::array_2_to_image,
};
use ndarray::Array2;

mod config;
mod particle_mesh;

fn main() {
    // let particle_mass =
    // 1.32 * 1e5 * (OMEGA_M0 * H0.powi(2)) * ((BOX_SIZE / (N_PARTICLES / 128)).pow(3) as f64);
    let average_density = (N_CELLS / N_PARTICLES).pow(3) as f64;
    let dt = (A_END - A_INIT) / STEPS;
    let dt_plot = (A_END - A_INIT) / N_PLOTS;
    let mut t_current = A_INIT;

    let mut n_plots = 0.;
    let rho = gaussian_random_field(POWER, AMPLITUDE);
    let (mut positions, mut velocities): (Array2<f64>, Array2<f64>) =
        initial_conditions(rho.clone());
    let ksq_inverse = ksq_inv();
    let mut idx = 0;
    while t_current < A_END - dt {
        let rho = density(&positions, average_density);
        (positions, velocities) = update(rho, positions, velocities, &ksq_inverse, t_current, dt);
        t_current += dt;
        idx += 1;

        if t_current >= A_INIT + n_plots * dt_plot {
            let img = array_2_to_image(positions.clone(), N_CELLS);
            let _ = img.save(format!("./img/positions{}.png", idx));
            n_plots += 1.;
        }
    }
}
