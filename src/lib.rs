#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod tensor;
pub mod ops;
pub mod christoffel;
pub mod christoffel_derivative;

pub use tensor::Tensor;
pub use christoffel::Christoffel;
pub use christoffel_derivative::ChristoffelDerivative;
pub use ops::outer::outer;
pub use ops::contract::contract;
pub use ops::covariant_derivative::covariant_derivative;
pub use ops::riemann::riemann;
pub use ops::ricci_tensor::ricci_tensor;
