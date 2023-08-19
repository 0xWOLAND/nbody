use crate::config::{H0, OMEGA_K0, OMEGA_LAMBDA0, OMEGA_M0};

pub fn expansion_factor(t: f64) -> f64 {
    1. / ((OMEGA_M0 + OMEGA_K0 * t + OMEGA_LAMBDA0 * f64::powi(t, 3)) / t).sqrt()
}

pub fn D_t(t: f64) -> f64 {
    5. / 2.
        / OMEGA_M0
        / (OMEGA_M0.powf(4. / 7.) - OMEGA_LAMBDA0
            + (1. + OMEGA_M0 / 2.) * (1. + OMEGA_LAMBDA0 / 70.))
        * t
}

pub fn hubble_constant(a: f64) -> f64 {
    H0 * (OMEGA_LAMBDA0 * a.powi(2) + OMEGA_K0 + OMEGA_M0 / a).sqrt()
}
