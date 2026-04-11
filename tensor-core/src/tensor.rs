/// Rank-(M,N) tensor over an `dim`-dimensional manifold.
///
/// - `M` upper (contravariant) indices, `N` lower (covariant) indices.
/// - Flat `Vec<f64>` storage, row-major, upper indices first.
/// - Total components: `dim^(M+N)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<const M: usize, const N: usize> {
    dim: usize,
    data: Vec<f64>,
}

impl<const M: usize, const N: usize> Tensor<M, N> {
    /// Total number of indices (rank).
    pub const RANK: usize = M + N;

    /// Create a zero tensor for a manifold of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        let len = dim.pow(Self::RANK as u32);
        Self {
            dim,
            data: vec![0.0; len],
        }
    }

    /// Create a tensor from a flat data vector.
    ///
    /// # Panics
    /// Panics if `data.len() != dim^(M+N)`.
    pub fn from_vec(dim: usize, data: Vec<f64>) -> Self {
        let expected = dim.pow(Self::RANK as u32);
        assert_eq!(
            data.len(),
            expected,
            "expected {} components for dim={} rank=({},{}), got {}",
            expected,
            dim,
            M,
            N,
            data.len()
        );
        Self { dim, data }
    }

    /// Manifold dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of upper (contravariant) indices.
    pub fn contravariant_rank(&self) -> usize {
        M
    }

    /// Number of lower (covariant) indices.
    pub fn covariant_rank(&self) -> usize {
        N
    }

    /// Total number of stored components.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Raw data slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable raw data slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Convert a multi-index `[i_0, i_1, ..., i_{M+N-1}]` to a flat offset.
    ///
    /// Row-major: the leftmost index varies slowest.
    ///
    /// # Panics
    /// Panics if the slice length doesn't match the rank, or any index >= dim.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            Self::RANK,
            "expected {} indices, got {}",
            Self::RANK,
            indices.len()
        );
        let mut offset = 0;
        for &idx in indices {
            assert!(idx < self.dim, "index {} out of range for dim={}", idx, self.dim);
            offset = offset * self.dim + idx;
        }
        offset
    }

    /// Decode a flat offset back to a multi-index.
    pub fn decode_flat_index(&self, mut flat: usize) -> Vec<usize> {
        let mut indices = vec![0usize; Self::RANK];
        for i in (0..Self::RANK).rev() {
            indices[i] = flat % self.dim;
            flat /= self.dim;
        }
        indices
    }

    /// Get a component by multi-index.
    pub fn component(&self, indices: &[usize]) -> f64 {
        let i = self.flat_index(indices);
        self.data[i]
    }

    /// Set a component by multi-index.
    pub fn set_component(&mut self, indices: &[usize], value: f64) {
        let i = self.flat_index(indices);
        self.data[i] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_tensor() {
        let t: Tensor<0, 2> = Tensor::new(4);
        assert_eq!(t.dim(), 4);
        assert_eq!(t.len(), 16);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_vec_and_access() {
        // Rank-(1,0) vector in 3D
        let v: Tensor<1, 0> = Tensor::from_vec(3, vec![1.0, 2.0, 3.0]);
        assert_eq!(v.component(&[0]), 1.0);
        assert_eq!(v.component(&[1]), 2.0);
        assert_eq!(v.component(&[2]), 3.0);
    }

    #[test]
    fn set_component() {
        let mut g: Tensor<0, 2> = Tensor::new(2);
        g.set_component(&[0, 0], -1.0);
        g.set_component(&[1, 1], 1.0);
        assert_eq!(g.component(&[0, 0]), -1.0);
        assert_eq!(g.component(&[1, 1]), 1.0);
        assert_eq!(g.component(&[0, 1]), 0.0);
    }

    #[test]
    fn flat_index_roundtrip() {
        let t: Tensor<1, 2> = Tensor::new(4);
        // Rank-3 tensor in 4D: 64 components
        assert_eq!(t.len(), 64);
        for flat in 0..64 {
            let indices = t.decode_flat_index(flat);
            assert_eq!(indices.len(), 3);
            assert_eq!(t.flat_index(&indices), flat);
        }
    }

    #[test]
    fn rank_queries() {
        let t: Tensor<2, 1> = Tensor::new(4);
        assert_eq!(t.contravariant_rank(), 2);
        assert_eq!(t.covariant_rank(), 1);
        assert_eq!(Tensor::<2, 1>::RANK, 3);
    }

    #[test]
    fn scalar_is_single_element() {
        let s: Tensor<0, 0> = Tensor::new(4);
        assert_eq!(s.len(), 1);
        assert_eq!(s.component(&[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "expected 16 components")]
    fn from_vec_wrong_size() {
        let _: Tensor<0, 2> = Tensor::from_vec(4, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "index 4 out of range")]
    fn index_out_of_range() {
        let t: Tensor<1, 0> = Tensor::new(4);
        t.component(&[4]);
    }

    #[test]
    #[should_panic(expected = "expected 2 indices")]
    fn wrong_number_of_indices() {
        let t: Tensor<0, 2> = Tensor::new(4);
        t.component(&[0]);
    }

    #[test]
    fn minkowski_metric() {
        // η = diag(-1, 1, 1, 1)
        let mut eta: Tensor<0, 2> = Tensor::new(4);
        eta.set_component(&[0, 0], -1.0);
        eta.set_component(&[1, 1], 1.0);
        eta.set_component(&[2, 2], 1.0);
        eta.set_component(&[3, 3], 1.0);

        // Check diagonal
        assert_eq!(eta.component(&[0, 0]), -1.0);
        for i in 1..4 {
            assert_eq!(eta.component(&[i, i]), 1.0);
        }
        // Check off-diagonal is zero
        assert_eq!(eta.component(&[0, 1]), 0.0);
        assert_eq!(eta.component(&[1, 2]), 0.0);
    }
}
