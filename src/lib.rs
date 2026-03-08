#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod tensor;
pub mod ops;
pub mod christoffel;
pub mod christoffel_derivative;
pub mod solver;
pub mod adm;
pub mod adm_rhs;

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
pub use solver::{solve_1d, solve_3d, invert_matrix, SolveResult, SolveResult3D};
pub use ops::em_source::em_t_grid;
pub use adm::{AdmState, ExtrinsicCurvature};
pub use adm_rhs::{adm_rhs_geodesic, adm_rhs_vacuum, hamiltonian_constraint, k_squared, momentum_constraint, AdmRhs};
