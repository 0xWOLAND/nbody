pub const N_PARTICLES: usize = 10;
pub const N_CELLS: usize = 16;
pub const BOX_SIZE: usize = 5;

// pub const N_PARTICLES: usize = 6;
// pub const N_CELLS: usize = 32;
// pub const BOX_SIZE: usize = 8;

pub const MASS: f64 = 4.0;
pub const OMEGA_M0: f64 = 0.31;
pub const OMEGA_B0: f64 = 0.04;
pub const OMEGA_K0: f64 = 0.00;
pub const OMEGA_LAMBDA0: f64 = 0.69;
pub const H0: f64 = 0.68;
pub const A_INIT: f64 = 0.01; // Initial Growth Factor
pub const A_END: f64 = 100.00; // Final scale factor
pub const STEPS: f64 = 1000.; // Number of timesteps
pub const N_PLOTS: f64 = 100.; // Number of timesteps

pub const DIV_BY_ZERO: f64 = 1e-34;
pub const IMG_WIDTH: usize = N_CELLS.pow(2);

pub const POWER: f64 = 0.845;
pub const AMPLITUDE: f64 = 3.685;
