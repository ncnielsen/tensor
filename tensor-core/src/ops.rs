use std::ops::{Add, Sub, Mul, Neg};

use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Add (element-wise, same rank)
// ---------------------------------------------------------------------------

impl<const M: usize, const N: usize> Add for &Tensor<M, N> {
    type Output = Tensor<M, N>;

    fn add(self, rhs: Self) -> Tensor<M, N> {
        assert_eq!(self.dim(), rhs.dim(), "dimension mismatch");
        let data: Vec<f64> = self.as_slice().iter()
            .zip(rhs.as_slice())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_vec(self.dim(), data)
    }
}

impl<const M: usize, const N: usize> Add for Tensor<M, N> {
    type Output = Tensor<M, N>;
    fn add(self, rhs: Self) -> Tensor<M, N> { &self + &rhs }
}

// ---------------------------------------------------------------------------
// Sub (element-wise, same rank)
// ---------------------------------------------------------------------------

impl<const M: usize, const N: usize> Sub for &Tensor<M, N> {
    type Output = Tensor<M, N>;

    fn sub(self, rhs: Self) -> Tensor<M, N> {
        assert_eq!(self.dim(), rhs.dim(), "dimension mismatch");
        let data: Vec<f64> = self.as_slice().iter()
            .zip(rhs.as_slice())
            .map(|(a, b)| a - b)
            .collect();
        Tensor::from_vec(self.dim(), data)
    }
}

impl<const M: usize, const N: usize> Sub for Tensor<M, N> {
    type Output = Tensor<M, N>;
    fn sub(self, rhs: Self) -> Tensor<M, N> { &self - &rhs }
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

impl<const M: usize, const N: usize> Neg for &Tensor<M, N> {
    type Output = Tensor<M, N>;

    fn neg(self) -> Tensor<M, N> {
        let data: Vec<f64> = self.as_slice().iter().map(|a| -a).collect();
        Tensor::from_vec(self.dim(), data)
    }
}

impl<const M: usize, const N: usize> Neg for Tensor<M, N> {
    type Output = Tensor<M, N>;
    fn neg(self) -> Tensor<M, N> { -&self }
}

// ---------------------------------------------------------------------------
// Scalar multiply: Tensor * f64, f64 * Tensor
// ---------------------------------------------------------------------------

impl<const M: usize, const N: usize> Mul<f64> for &Tensor<M, N> {
    type Output = Tensor<M, N>;

    fn mul(self, scalar: f64) -> Tensor<M, N> {
        let data: Vec<f64> = self.as_slice().iter().map(|a| a * scalar).collect();
        Tensor::from_vec(self.dim(), data)
    }
}

impl<const M: usize, const N: usize> Mul<f64> for Tensor<M, N> {
    type Output = Tensor<M, N>;
    fn mul(self, scalar: f64) -> Tensor<M, N> { &self * scalar }
}

impl<const M: usize, const N: usize> Mul<&Tensor<M, N>> for f64 {
    type Output = Tensor<M, N>;
    fn mul(self, tensor: &Tensor<M, N>) -> Tensor<M, N> { tensor * self }
}

impl<const M: usize, const N: usize> Mul<Tensor<M, N>> for f64 {
    type Output = Tensor<M, N>;
    fn mul(self, tensor: Tensor<M, N>) -> Tensor<M, N> { &tensor * self }
}

// ---------------------------------------------------------------------------
// Outer (tensor) product
// ---------------------------------------------------------------------------

/// Outer (tensor) product: `a ⊗ b`.
///
/// Result index layout (upper indices first, then lower):
///   `[a_upper... | b_upper... | a_lower... | b_lower...]`
///
/// The output rank must be specified by the caller — Rust const generics
/// cannot express `M1+M2` directly. Type inference from the binding works:
///
/// ```ignore
/// let v: Tensor<1, 0> = /* vector */;
/// let w: Tensor<0, 1> = /* covector */;
/// let t: Tensor<1, 1> = outer(&v, &w);
/// ```
///
/// # Panics
/// - Dimension mismatch between `a` and `b`
/// - Output rank mismatch (`MO != M1+M2` or `NO != N1+N2`)
pub fn outer<
    const M1: usize, const N1: usize,
    const M2: usize, const N2: usize,
    const MO: usize, const NO: usize,
>(
    a: &Tensor<M1, N1>,
    b: &Tensor<M2, N2>,
) -> Tensor<MO, NO> {
    assert_eq!(a.dim(), b.dim(), "dimension mismatch");
    assert_eq!(MO, M1 + M2, "output upper rank mismatch: MO={} != M1+M2={}", MO, M1 + M2);
    assert_eq!(NO, N1 + N2, "output lower rank mismatch: NO={} != N1+N2={}", NO, N1 + N2);

    let dim = a.dim();
    let mut result = Tensor::<MO, NO>::new(dim);

    for flat in 0..result.len() {
        let idx = result.decode_flat_index(flat);

        // Split result multi-index into a's and b's parts:
        //   idx = [a_upper(M1) | b_upper(M2) | a_lower(N1) | b_lower(N2)]
        let mut a_idx = Vec::with_capacity(M1 + N1);
        a_idx.extend_from_slice(&idx[..M1]);
        a_idx.extend_from_slice(&idx[MO..MO + N1]);

        let mut b_idx = Vec::with_capacity(M2 + N2);
        b_idx.extend_from_slice(&idx[M1..MO]);
        b_idx.extend_from_slice(&idx[MO + N1..]);

        result.as_mut_slice()[flat] = a.component(&a_idx) * b.component(&b_idx);
    }

    result
}

// ---------------------------------------------------------------------------
// Contraction
// ---------------------------------------------------------------------------

/// Contract one upper index with one lower index, summing over the shared
/// dimension.
///
/// `upper_idx`: which upper index to contract (0-based among upper indices)
/// `lower_idx`: which lower index to contract (0-based among lower indices)
///
/// The output rank must be specified by the caller:
///
/// ```ignore
/// let t: Tensor<1, 1> = /* mixed tensor */;
/// let s: Tensor<0, 0> = contract(&t, 0, 0); // trace
/// ```
///
/// # Panics
/// - Output rank mismatch (`MO != M-1` or `NO != N-1`)
/// - `upper_idx >= M` or `lower_idx >= N`
pub fn contract<
    const M: usize, const N: usize,
    const MO: usize, const NO: usize,
>(
    t: &Tensor<M, N>,
    upper_idx: usize,
    lower_idx: usize,
) -> Tensor<MO, NO> {
    assert_eq!(MO, M - 1, "output upper rank mismatch: MO={} != M-1={}", MO, M - 1);
    assert_eq!(NO, N - 1, "output lower rank mismatch: NO={} != N-1={}", NO, N - 1);
    assert!(upper_idx < M, "upper_idx {} out of range for M={}", upper_idx, M);
    assert!(lower_idx < N, "lower_idx {} out of range for N={}", lower_idx, N);

    let dim = t.dim();
    let mut result = Tensor::<MO, NO>::new(dim);

    for flat in 0..result.len() {
        let out_idx = result.decode_flat_index(flat);

        let mut sum = 0.0;
        for k in 0..dim {
            // Build t's full multi-index: M upper then N lower.
            // Insert k at position upper_idx among upper indices,
            // and k at position lower_idx among lower indices.
            let mut t_idx = Vec::with_capacity(M + N);

            // Upper indices
            for i in 0..M {
                if i == upper_idx {
                    t_idx.push(k);
                } else {
                    let reduced = if i < upper_idx { i } else { i - 1 };
                    t_idx.push(out_idx[reduced]);
                }
            }

            // Lower indices
            for j in 0..N {
                if j == lower_idx {
                    t_idx.push(k);
                } else {
                    let reduced = if j < lower_idx { j } else { j - 1 };
                    t_idx.push(out_idx[MO + reduced]);
                }
            }

            sum += t.component(&t_idx);
        }

        result.as_mut_slice()[flat] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Add / Sub -----------------------------------------------------------

    #[test]
    fn add_rank_02() {
        let a = Tensor::<0, 2>::from_vec(2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::<0, 2>::from_vec(2, vec![10.0, 20.0, 30.0, 40.0]);
        let c = &a + &b;
        assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sub_rank_02() {
        let a = Tensor::<0, 2>::from_vec(2, vec![10.0, 20.0, 30.0, 40.0]);
        let b = Tensor::<0, 2>::from_vec(2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a - &b;
        assert_eq!(c.as_slice(), &[9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn neg() {
        let a = Tensor::<1, 0>::from_vec(3, vec![1.0, -2.0, 3.0]);
        let b = -&a;
        assert_eq!(b.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn add_owned() {
        let a = Tensor::<1, 0>::from_vec(2, vec![1.0, 2.0]);
        let b = Tensor::<1, 0>::from_vec(2, vec![3.0, 4.0]);
        let c = a + b; // consumes both
        assert_eq!(c.as_slice(), &[4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn add_dim_mismatch() {
        let a = Tensor::<1, 0>::from_vec(2, vec![1.0, 2.0]);
        let b = Tensor::<1, 0>::from_vec(3, vec![1.0, 2.0, 3.0]);
        let _ = &a + &b;
    }

    // -- Scalar multiply -----------------------------------------------------

    #[test]
    fn scalar_mul() {
        let a = Tensor::<0, 2>::from_vec(2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = &a * 3.0;
        assert_eq!(b.as_slice(), &[3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn scalar_mul_commutative() {
        let a = Tensor::<1, 0>::from_vec(2, vec![5.0, 7.0]);
        let b = 2.0 * &a;
        let c = &a * 2.0;
        assert_eq!(b.as_slice(), c.as_slice());
    }

    // -- Outer product -------------------------------------------------------

    #[test]
    fn outer_vector_covector() {
        // v^i = [1, 2, 3], w_j = [4, 5, 6]
        // (v ⊗ w)^i_j = v^i * w_j
        let v = Tensor::<1, 0>::from_vec(3, vec![1.0, 2.0, 3.0]);
        let w = Tensor::<0, 1>::from_vec(3, vec![4.0, 5.0, 6.0]);
        let t: Tensor<1, 1> = outer(&v, &w);

        assert_eq!(t.dim(), 3);
        assert_eq!(t.len(), 9);

        // t^i_j = v^i * w_j
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    t.component(&[i, j]),
                    v.component(&[i]) * w.component(&[j]),
                );
            }
        }
    }

    #[test]
    fn outer_two_covectors() {
        // a_i = [1, 2], b_j = [3, 4]
        // (a ⊗ b)_{ij} = a_i * b_j
        let a = Tensor::<0, 1>::from_vec(2, vec![1.0, 2.0]);
        let b = Tensor::<0, 1>::from_vec(2, vec![3.0, 4.0]);
        let t: Tensor<0, 2> = outer(&a, &b);

        assert_eq!(t.component(&[0, 0]), 3.0);  // 1*3
        assert_eq!(t.component(&[0, 1]), 4.0);  // 1*4
        assert_eq!(t.component(&[1, 0]), 6.0);  // 2*3
        assert_eq!(t.component(&[1, 1]), 8.0);  // 2*4
    }

    #[test]
    fn outer_two_vectors() {
        // a^i = [2, 3], b^j = [5, 7]
        // (a ⊗ b)^{ij}
        let a = Tensor::<1, 0>::from_vec(2, vec![2.0, 3.0]);
        let b = Tensor::<1, 0>::from_vec(2, vec![5.0, 7.0]);
        let t: Tensor<2, 0> = outer(&a, &b);

        assert_eq!(t.component(&[0, 0]), 10.0); // 2*5
        assert_eq!(t.component(&[0, 1]), 14.0); // 2*7
        assert_eq!(t.component(&[1, 0]), 15.0); // 3*5
        assert_eq!(t.component(&[1, 1]), 21.0); // 3*7
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn outer_dim_mismatch() {
        let a = Tensor::<1, 0>::from_vec(2, vec![1.0, 2.0]);
        let b = Tensor::<0, 1>::from_vec(3, vec![1.0, 2.0, 3.0]);
        let _: Tensor<1, 1> = outer(&a, &b);
    }

    // -- Contraction ---------------------------------------------------------

    #[test]
    fn contract_identity_is_dim() {
        // δ^μ_ν in 4D: trace = 4
        let mut delta = Tensor::<1, 1>::new(4);
        for i in 0..4 {
            delta.set_component(&[i, i], 1.0);
        }
        let tr: Tensor<0, 0> = contract(&delta, 0, 0);
        assert_eq!(tr.component(&[]), 4.0);
    }

    #[test]
    fn contract_trace_manual() {
        // T^μ_ν with known values, verify trace = Σ_μ T^μ_μ
        let mut t = Tensor::<1, 1>::new(3);
        t.set_component(&[0, 0], 2.0);
        t.set_component(&[0, 1], 5.0);
        t.set_component(&[1, 0], 7.0);
        t.set_component(&[1, 1], 3.0);
        t.set_component(&[2, 2], 11.0);

        let tr: Tensor<0, 0> = contract(&t, 0, 0);
        // trace = T^0_0 + T^1_1 + T^2_2 = 2 + 3 + 11 = 16
        assert_eq!(tr.component(&[]), 16.0);
    }

    #[test]
    fn contract_rank_21_to_10() {
        // T^{ab}_c in 2D, contract upper index 1 (b) with lower index 0 (c)
        // result: R^a = Σ_k T^{ak}_k
        let mut t = Tensor::<2, 1>::new(2);
        // T^{00}_0 = 1, T^{00}_1 = 2, T^{01}_0 = 3, T^{01}_1 = 4
        // T^{10}_0 = 5, T^{10}_1 = 6, T^{11}_0 = 7, T^{11}_1 = 8
        for (i, val) in (1..=8).enumerate() {
            t.as_mut_slice()[i] = val as f64;
        }

        // Contract upper_idx=1 (b) with lower_idx=0 (c): sum over b=c=k
        // R^a = Σ_k T^{ak}_k
        // R^0 = T^{00}_0 + T^{01}_1 = 1 + 4 = 5
        // R^1 = T^{10}_0 + T^{11}_1 = 5 + 8 = 13
        let r: Tensor<1, 0> = contract(&t, 1, 0);
        assert_eq!(r.component(&[0]), 5.0);
        assert_eq!(r.component(&[1]), 13.0);
    }

    #[test]
    fn outer_then_contract_is_dot() {
        // v^i w_i = Σ_i v^i * w_i (inner product via outer + contract)
        let v = Tensor::<1, 0>::from_vec(3, vec![1.0, 2.0, 3.0]);
        let w = Tensor::<0, 1>::from_vec(3, vec![4.0, 5.0, 6.0]);

        let vw: Tensor<1, 1> = outer(&v, &w);
        let dot: Tensor<0, 0> = contract(&vw, 0, 0);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot.component(&[]), 32.0);
    }
}
