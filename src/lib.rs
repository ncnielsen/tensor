#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod tensor;
pub mod ops;

pub use tensor::Tensor;
pub use ops::outer::outer;
pub use ops::contract::contract;
