use crate::config::{OMEGA_K0, OMEGA_LAMBDA0, OMEGA_M0};

pub fn expansion_factor(t: f64) -> f64 {
    1. / ((OMEGA_M0 + OMEGA_K0 * t + OMEGA_LAMBDA0 * f64::powi(t, 3)) / t).sqrt()
}
