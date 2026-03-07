#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod tensor;
pub mod ops;
pub mod christoffel;

pub use tensor::Tensor;
pub use christoffel::Christoffel;
pub use ops::outer::outer;
pub use ops::contract::contract;
pub use ops::covariant_derivative::covariant_derivative;
