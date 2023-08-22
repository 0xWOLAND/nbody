pub fn test() -> String {
    "".to_string()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use crate::config::{N_CELLS, N_PARTICLES};
    use crate::{ic::initial_conditions, random_field::gaussian_random_field};

    #[test]
    pub fn simulate() {
        let average_density = (N_CELLS / N_PARTICLES).pow(3);
        let rho: Array3<f64> = gaussian_random_field(0.845, 3.685);
        let (positions, velocities): (Array2<f64>, Array2<f64>) = initial_conditions(rho);
    }
}
