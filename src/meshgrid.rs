use std::ops;

use ndarray::{Array, Array2, Array3, ShapeBuilder};

#[derive(Debug)]
pub struct Meshgrid2 {
    grid: Vec<(f64, f64)>,
    dim: (usize, usize),
}

impl Meshgrid2 {
    pub fn new(xs: &Vec<f64>, ys: &Vec<f64>) -> Meshgrid2 {
        let grid: Vec<(f64, f64)> = ys
            .clone()
            .iter()
            .flat_map(move |y| xs.clone().iter().map(move |x| (*x, *y)).collect::<Vec<_>>())
            .collect();
        Meshgrid2 {
            grid,
            dim: (xs.len(), ys.len()),
        }
    }

    pub fn x(&self) -> Array2<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.0).collect()).unwrap()
    }

    pub fn y(&self) -> Array2<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.1).collect()).unwrap()
    }

    pub fn get(&self) -> (Array2<f64>, Array2<f64>) {
        (self.x(), self.y())
    }

    pub fn pow(&self, n: i32) -> Meshgrid2 {
        Meshgrid2 {
            grid: self
                .grid
                .iter()
                .map(|x| (x.0.powi(n), x.1.powi(n)))
                .collect(),
            dim: self.dim,
        }
    }

    pub fn sin(&self) -> Meshgrid2 {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (x.1.sin(), x.1.sin())).collect(),
            dim: self.dim,
        }
    }

    pub fn cos(&self) -> Meshgrid2 {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (x.1.cos(), x.1.cos())).collect(),
            dim: self.dim,
        }
    }

    pub fn tan(&self) -> Meshgrid2 {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (x.1.tan(), x.1.tan())).collect(),
            dim: self.dim,
        }
    }
}

impl ops::Mul<f64> for Meshgrid2 {
    type Output = Meshgrid2;

    fn mul(self, rhs: f64) -> Self::Output {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (rhs * x.0, rhs * x.1)).collect(),
            dim: self.dim,
        }
    }
}

impl ops::Mul<Meshgrid2> for Meshgrid2 {
    type Output = Meshgrid2;

    fn mul(self, rhs: Meshgrid2) -> Self::Output {
        Meshgrid2 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        rhs.grid.get(i).unwrap().0 * x.0,
                        rhs.grid.get(i).unwrap().1 * x.1,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Add<f64> for Meshgrid2 {
    type Output = Meshgrid2;

    fn add(self, rhs: f64) -> Self::Output {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (rhs + x.0, rhs + x.1)).collect(),
            dim: self.dim,
        }
    }
}

impl ops::Add<Meshgrid2> for Meshgrid2 {
    type Output = Meshgrid2;

    fn add(self, rhs: Meshgrid2) -> Self::Output {
        Meshgrid2 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        rhs.grid.get(i).unwrap().0 + x.0,
                        rhs.grid.get(i).unwrap().1 + x.1,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Div<f64> for Meshgrid2 {
    type Output = Meshgrid2;

    fn div(self, rhs: f64) -> Self::Output {
        Meshgrid2 {
            grid: self.grid.iter().map(|x| (x.0 / rhs, x.1 / rhs)).collect(),
            dim: self.dim,
        }
    }
}

impl ops::Div<Meshgrid2> for Meshgrid2 {
    type Output = Meshgrid2;

    fn div(self, rhs: Meshgrid2) -> Self::Output {
        Meshgrid2 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        x.0 / rhs.grid.get(i).unwrap().0,
                        x.1 / rhs.grid.get(i).unwrap().1,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

#[derive(Debug)]
pub struct Meshgrid3 {
    grid: Vec<(f64, f64, f64)>,
    dim: (usize, usize, usize),
}

impl Meshgrid3 {
    pub fn new(xs: &Vec<f64>, ys: &Vec<f64>, zs: &Vec<f64>) -> Meshgrid3 {
        let grid: Vec<(f64, f64, f64)> = zs
            .clone()
            .iter()
            .flat_map(move |z| {
                ys.clone()
                    .iter()
                    .flat_map(move |y| {
                        xs.clone()
                            .iter()
                            .map(move |x| (*x, *y, *z))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Meshgrid3 {
            grid,
            dim: (xs.len(), ys.len(), zs.len()),
        }
    }

    pub fn x(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.0).collect()).unwrap()
    }

    pub fn y(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.1).collect()).unwrap()
    }

    pub fn z(&self) -> Array3<f64> {
        Array::from_shape_vec(self.dim.f(), self.grid.iter().map(|x| x.2).collect()).unwrap()
    }

    pub fn get(&self) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        (self.x(), self.y(), self.z())
    }

    pub fn pow(&self, n: i32) -> Meshgrid3 {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (x.0.powi(n), x.1.powi(n), x.2.powi(n)))
                .collect(),
            dim: self.dim,
        }
    }

    pub fn sin(&self) -> Meshgrid3 {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (x.0.sin(), x.1.sin(), x.2.sin()))
                .collect(),
            dim: self.dim,
        }
    }

    pub fn cos(&self) -> Meshgrid3 {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (x.0.cos(), x.1.cos(), x.2.cos()))
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Mul<f64> for Meshgrid3 {
    type Output = Meshgrid3;

    fn mul(self, rhs: f64) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (rhs * x.0, rhs * x.1, rhs * x.2))
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Mul<Meshgrid3> for Meshgrid3 {
    type Output = Meshgrid3;

    fn mul(self, rhs: Meshgrid3) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        rhs.grid.get(i).unwrap().0 * x.0,
                        rhs.grid.get(i).unwrap().1 * x.1,
                        rhs.grid.get(i).unwrap().2 * x.2,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Add<f64> for Meshgrid3 {
    type Output = Meshgrid3;

    fn add(self, rhs: f64) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (rhs + x.0, rhs + x.1, rhs + x.2))
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Add<Meshgrid3> for Meshgrid3 {
    type Output = Meshgrid3;

    fn add(self, rhs: Meshgrid3) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        rhs.grid.get(i).unwrap().0 + x.0,
                        rhs.grid.get(i).unwrap().1 + x.1,
                        rhs.grid.get(i).unwrap().2 + x.2,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Div<f64> for Meshgrid3 {
    type Output = Meshgrid3;

    fn div(self, rhs: f64) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .map(|x| (x.0 / rhs, x.1 / rhs, x.2 / rhs))
                .collect(),
            dim: self.dim,
        }
    }
}

impl ops::Div<Meshgrid3> for Meshgrid3 {
    type Output = Meshgrid3;

    fn div(self, rhs: Meshgrid3) -> Self::Output {
        Meshgrid3 {
            grid: self
                .grid
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        x.0 / rhs.grid.get(i).unwrap().0,
                        x.1 / rhs.grid.get(i).unwrap().1,
                        x.2 / rhs.grid.get(i).unwrap().1,
                    )
                })
                .collect(),
            dim: self.dim,
        }
    }
}

#[cfg(test)]
mod Tests {
    use super::Meshgrid2;

    #[test]
    fn test() {
        let m = Meshgrid2::new(&vec![1., 2.], &vec![3., 4.]);
        println!("{:?}", m);
        println!("{:?}", m.x());
        println!("{:?}", m.y());
    }
}
