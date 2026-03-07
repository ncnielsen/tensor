use std::ops::Add;

use crate::tensor::Tensor;

/// Component-wise addition of two tensors of the same rank and dimension.
/// The type system enforces that both rank (M, N) and dimension match.
impl<const M: usize, const N: usize> Add for Tensor<M, N> {
    type Output = Tensor<M, N>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dim, rhs.dim,
            "Dimension mismatch in tensor addition: {} vs {}",
            self.dim, rhs.dim
        );
        let components = self
            .components
            .into_iter()
            .zip(rhs.components)
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(self.dim, components)
    }
}
