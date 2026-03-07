use aad::number::Number;
use std::fmt;

/// A tensor of contravariant rank M and covariant rank N.
///
/// Components are stored flattened in row-major order with upper indices
/// preceding lower indices. For a tensor T^{i1...iM}_{j1...jN} in d dimensions,
/// the component at multi-index [i1,...,iM, j1,...,jN] lives at flat position:
///   i1*d^(M+N-1) + i2*d^(M+N-2) + ... + jN*d^0
///
/// Components are of type Number so every arithmetic operation is recorded on
/// the AAD tape, giving differentiation of all tensor expressions for free.
#[derive(Debug, Clone)]
pub struct Tensor<const M: usize, const N: usize> {
    pub components: Vec<Number>,
    pub dim: usize,
}

/// Encode a multi-index into a flat position using row-major (Horner) order.
pub fn flat_index(indices: &[usize], dim: usize) -> usize {
    indices.iter().fold(0, |acc, &i| acc * dim + i)
}

/// Decode a flat position back into a multi-index of the given rank.
pub fn decode_flat_index(mut flat: usize, dim: usize, rank: usize) -> Vec<usize> {
    let mut indices = vec![0usize; rank];
    for i in (0..rank).rev() {
        indices[i] = flat % dim;
        flat /= dim;
    }
    indices
}

impl<const M: usize, const N: usize> Tensor<M, N> {
    /// Construct a tensor from a pre-built component vector.
    /// Panics if the number of components does not match dim^(M+N).
    pub fn new(dim: usize, components: Vec<Number>) -> Self {
        let expected = if M + N == 0 { 1 } else { dim.pow((M + N) as u32) };
        assert_eq!(
            components.len(),
            expected,
            "Expected {} components for Tensor<{},{}> in dim {}, got {}",
            expected, M, N, dim, components.len()
        );
        Tensor { components, dim }
    }

    /// Convenience constructor: wraps plain f64 values as leaf Number nodes.
    pub fn from_f64(dim: usize, values: Vec<f64>) -> Self {
        let components = values.into_iter().map(Number::new).collect();
        Self::new(dim, components)
    }

    /// Return the component at the given multi-index.
    /// Index order: upper indices first, then lower indices.
    pub fn component(&self, indices: &[usize]) -> Number {
        assert_eq!(
            indices.len(),
            M + N,
            "Expected {} indices for Tensor<{},{}>, got {}",
            M + N, M, N, indices.len()
        );
        self.components[flat_index(indices, self.dim)]
    }
}

impl<const M: usize, const N: usize> fmt::Display for Tensor<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{},{}>(dim={}, [", M, N, self.dim)?;
        for (i, c) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", c.result)?;
        }
        write!(f, "])")
    }
}
