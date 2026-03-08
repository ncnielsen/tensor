#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod tensor;
pub mod ops;
pub mod christoffel;
pub mod christoffel_derivative;
pub mod solver;

pub use tensor::Tensor;
pub use christoffel::Christoffel;
pub use christoffel_derivative::ChristoffelDerivative;
pub use ops::outer::outer;
pub use ops::contract::contract;
pub use ops::covariant_derivative::covariant_derivative;
pub use ops::riemann::riemann;
pub use ops::ricci_tensor::ricci_tensor;
pub use ops::ricci_scalar::ricci_scalar;
pub use ops::einstein_tensor::einstein_tensor;
pub use ops::faraday::faraday;
pub use ops::em_stress_energy::em_stress_energy;
pub use ops::partial_deriv::partial_deriv;
pub use ops::partial_deriv::christoffel_partial_deriv;
pub use ops::einstein_residual::einstein_residual;
pub use ops::newton_step::newton_step;
pub use solver::{solve_1d, invert_matrix, SolveResult};
