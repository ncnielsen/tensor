use crate::tensor::Tensor;

/// Enforce symmetry on a rank-(0,2) tensor: g_{μν} ← (g_{μν} + g_{νμ}) / 2.
pub fn enforce_symmetry(g: &mut Tensor<0, 2>) {
    let dim = g.dim();
    for mu in 0..dim {
        for nu in (mu + 1)..dim {
            let avg = (g.component(&[mu, nu]) + g.component(&[nu, mu])) / 2.0;
            g.set_component(&[mu, nu], avg);
            g.set_component(&[nu, mu], avg);
        }
    }
}

/// Invert a metric tensor g_{μν} → g^{μν} via Gaussian elimination with
/// partial pivoting.
///
/// # Panics
/// If the metric is singular (zero pivot).
pub fn invert_metric(g: &Tensor<0, 2>) -> Tensor<2, 0> {
    let dim = g.dim();

    // Augmented matrix [g | I]
    let mut aug = vec![vec![0.0; 2 * dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            aug[i][j] = g.component(&[i, j]);
        }
        aug[i][dim + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..dim {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..dim {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        assert!(max_val > 1e-14, "singular metric: zero pivot at column {}", col);

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..dim {
            let factor = aug[row][col] / pivot;
            for j in col..(2 * dim) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    for col in (0..dim).rev() {
        let pivot = aug[col][col];
        for j in 0..(2 * dim) {
            aug[col][j] /= pivot;
        }
        for row in 0..col {
            let factor = aug[row][col];
            for j in 0..(2 * dim) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Tensor::<2, 0>::new(dim);
    for i in 0..dim {
        for j in 0..dim {
            inv.set_component(&[i, j], aug[i][dim + j]);
        }
    }

    inv
}

/// Lower one upper index using the metric.
///
/// Lowers the `upper_idx`-th upper index of `t`. The lowered index becomes
/// the **last** lower index of the result.
///
/// Result_{...}_{...μ} = g_{μα} T^{...α...}_{...}
///
/// Output type specified by caller (Rust can't express M-1, N+1 in const generics):
/// ```ignore
/// let v_lower: Tensor<0, 1> = lower_index(&v_upper, &g, 0);
/// ```
pub fn lower_index<
    const M: usize, const N: usize,
    const MO: usize, const NO: usize,
>(
    t: &Tensor<M, N>,
    g: &Tensor<0, 2>,
    upper_idx: usize,
) -> Tensor<MO, NO> {
    assert_eq!(MO, M - 1, "output upper rank: expected {}, got {}", M - 1, MO);
    assert_eq!(NO, N + 1, "output lower rank: expected {}, got {}", N + 1, NO);
    assert!(upper_idx < M, "upper_idx {} out of range for M={}", upper_idx, M);
    assert_eq!(t.dim(), g.dim(), "dimension mismatch");

    let dim = t.dim();
    let mut result = Tensor::<MO, NO>::new(dim);

    for flat in 0..result.len() {
        let out = result.decode_flat_index(flat);
        // out = [upper_0..upper_{MO-1}, lower_0..lower_{NO-1}]
        // Last lower index is the newly lowered one (μ)
        let mu = out[MO + NO - 1];

        let mut sum = 0.0;
        for alpha in 0..dim {
            let mut t_idx = Vec::with_capacity(M + N);

            // Upper indices of t: re-insert alpha at upper_idx
            for i in 0..M {
                if i == upper_idx {
                    t_idx.push(alpha);
                } else {
                    let reduced = if i < upper_idx { i } else { i - 1 };
                    t_idx.push(out[reduced]);
                }
            }

            // Lower indices of t: first N of result's lower part
            for j in 0..N {
                t_idx.push(out[MO + j]);
            }

            sum += g.component(&[mu, alpha]) * t.component(&t_idx);
        }

        result.as_mut_slice()[flat] = sum;
    }

    result
}

/// Raise one lower index using the inverse metric.
///
/// Raises the `lower_idx`-th lower index of `t`. The raised index becomes
/// the **last** upper index of the result.
///
/// Result^{...μ}_{...} = g^{μα} T^{...}_{...α...}
///
/// Output type specified by caller:
/// ```ignore
/// let v_upper: Tensor<1, 0> = raise_index(&v_lower, &g_inv, 0);
/// ```
pub fn raise_index<
    const M: usize, const N: usize,
    const MO: usize, const NO: usize,
>(
    t: &Tensor<M, N>,
    g_inv: &Tensor<2, 0>,
    lower_idx: usize,
) -> Tensor<MO, NO> {
    assert_eq!(MO, M + 1, "output upper rank: expected {}, got {}", M + 1, MO);
    assert_eq!(NO, N - 1, "output lower rank: expected {}, got {}", N - 1, NO);
    assert!(lower_idx < N, "lower_idx {} out of range for N={}", lower_idx, N);
    assert_eq!(t.dim(), g_inv.dim(), "dimension mismatch");

    let dim = t.dim();
    let mut result = Tensor::<MO, NO>::new(dim);

    for flat in 0..result.len() {
        let out = result.decode_flat_index(flat);
        // out = [upper_0..upper_{MO-1}, lower_0..lower_{NO-1}]
        // Last upper index is the newly raised one (μ)
        let mu = out[MO - 1];

        let mut sum = 0.0;
        for alpha in 0..dim {
            let mut t_idx = Vec::with_capacity(M + N);

            // Upper indices of t: first M of result's upper part
            for i in 0..M {
                t_idx.push(out[i]);
            }

            // Lower indices of t: re-insert alpha at lower_idx
            for j in 0..N {
                if j == lower_idx {
                    t_idx.push(alpha);
                } else {
                    let reduced = if j < lower_idx { j } else { j - 1 };
                    t_idx.push(out[MO + reduced]);
                }
            }

            sum += g_inv.component(&[mu, alpha]) * t.component(&t_idx);
        }

        result.as_mut_slice()[flat] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{contract, outer};

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "{} != {} (diff={})", a, b, (a - b).abs());
    }

    // -- Symmetry ------------------------------------------------------------

    #[test]
    fn enforce_symmetry_averages() {
        let mut g = Tensor::<0, 2>::new(3);
        g.set_component(&[0, 1], 4.0);
        g.set_component(&[1, 0], 6.0);
        enforce_symmetry(&mut g);
        assert_eq!(g.component(&[0, 1]), 5.0);
        assert_eq!(g.component(&[1, 0]), 5.0);
    }

    #[test]
    fn enforce_symmetry_noop_on_symmetric() {
        let mut g = Tensor::<0, 2>::new(2);
        g.set_component(&[0, 0], 1.0);
        g.set_component(&[0, 1], 3.0);
        g.set_component(&[1, 0], 3.0);
        g.set_component(&[1, 1], 2.0);
        let before: Vec<f64> = g.as_slice().to_vec();
        enforce_symmetry(&mut g);
        assert_eq!(g.as_slice(), &before[..]);
    }

    // -- Inversion -----------------------------------------------------------

    fn minkowski() -> Tensor<0, 2> {
        let mut eta = Tensor::<0, 2>::new(4);
        eta.set_component(&[0, 0], -1.0);
        for i in 1..4 {
            eta.set_component(&[i, i], 1.0);
        }
        eta
    }

    #[test]
    fn invert_minkowski() {
        let eta = minkowski();
        let eta_inv = invert_metric(&eta);
        // Minkowski is self-inverse: η^{μν} = diag(-1,1,1,1)
        approx_eq(eta_inv.component(&[0, 0]), -1.0);
        for i in 1..4 {
            approx_eq(eta_inv.component(&[i, i]), 1.0);
        }
        // Off-diagonal zero
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    approx_eq(eta_inv.component(&[i, j]), 0.0);
                }
            }
        }
    }

    #[test]
    fn invert_schwarzschild() {
        // Schwarzschild at r=10, M=1, θ=π/2
        // g = diag(-(1-2/r), 1/(1-2/r), r², r²sin²θ)
        //   = diag(-0.8, 1.25, 100, 100)
        let r = 10.0_f64;
        let f = 1.0 - 2.0 / r; // 0.8
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r); // sin²(π/2) = 1

        let g_inv = invert_metric(&g);

        approx_eq(g_inv.component(&[0, 0]), -1.0 / f);   // -1.25
        approx_eq(g_inv.component(&[1, 1]), f);            // 0.8
        approx_eq(g_inv.component(&[2, 2]), 1.0 / (r * r)); // 0.01
        approx_eq(g_inv.component(&[3, 3]), 1.0 / (r * r)); // 0.01
    }

    #[test]
    fn inverse_symmetry_preserved() {
        // Non-diagonal symmetric metric
        let mut g = Tensor::<0, 2>::new(3);
        g.set_component(&[0, 0], 2.0);
        g.set_component(&[0, 1], 0.5);
        g.set_component(&[1, 0], 0.5);
        g.set_component(&[1, 1], 3.0);
        g.set_component(&[2, 2], 1.0);

        let g_inv = invert_metric(&g);

        for i in 0..3 {
            for j in 0..3 {
                approx_eq(g_inv.component(&[i, j]), g_inv.component(&[j, i]));
            }
        }
    }

    // -- g^{μα} g_{αν} = δ^μ_ν ----------------------------------------------

    #[test]
    fn metric_times_inverse_is_identity_minkowski() {
        let g = minkowski();
        let g_inv = invert_metric(&g);
        assert_metric_inverse_identity(&g, &g_inv);
    }

    #[test]
    fn metric_times_inverse_is_identity_schwarzschild() {
        let r = 10.0_f64;
        let f = 1.0 - 2.0 / r;
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r);

        let g_inv = invert_metric(&g);
        assert_metric_inverse_identity(&g, &g_inv);
    }

    #[test]
    fn metric_times_inverse_is_identity_offdiagonal() {
        let mut g = Tensor::<0, 2>::new(3);
        g.set_component(&[0, 0], 2.0);
        g.set_component(&[0, 1], 0.5);
        g.set_component(&[1, 0], 0.5);
        g.set_component(&[1, 1], 3.0);
        g.set_component(&[2, 2], 1.0);

        let g_inv = invert_metric(&g);
        assert_metric_inverse_identity(&g, &g_inv);
    }

    fn assert_metric_inverse_identity(g: &Tensor<0, 2>, g_inv: &Tensor<2, 0>) {
        // g^{μα} g_{αν} = δ^μ_ν via outer product + contraction
        let product: Tensor<2, 2> = outer(g_inv, g);
        let delta: Tensor<1, 1> = contract(&product, 1, 0);

        let dim = g.dim();
        for mu in 0..dim {
            for nu in 0..dim {
                let expected = if mu == nu { 1.0 } else { 0.0 };
                approx_eq(delta.component(&[mu, nu]), expected);
            }
        }
    }

    // -- Raise / Lower -------------------------------------------------------

    #[test]
    fn lower_vector_minkowski() {
        let eta = minkowski();
        // v^μ = [1, 2, 3, 4]
        let v = Tensor::<1, 0>::from_vec(4, vec![1.0, 2.0, 3.0, 4.0]);
        // v_μ = η_{μα} v^α = [-1, 2, 3, 4]
        let v_low: Tensor<0, 1> = lower_index(&v, &eta, 0);
        approx_eq(v_low.component(&[0]), -1.0);
        approx_eq(v_low.component(&[1]), 2.0);
        approx_eq(v_low.component(&[2]), 3.0);
        approx_eq(v_low.component(&[3]), 4.0);
    }

    #[test]
    fn raise_covector_minkowski() {
        let eta = minkowski();
        let eta_inv = invert_metric(&eta);
        // w_μ = [-1, 2, 3, 4]
        let w = Tensor::<0, 1>::from_vec(4, vec![-1.0, 2.0, 3.0, 4.0]);
        // w^μ = η^{μα} w_α = [1, 2, 3, 4]
        let w_up: Tensor<1, 0> = raise_index(&w, &eta_inv, 0);
        approx_eq(w_up.component(&[0]), 1.0);
        approx_eq(w_up.component(&[1]), 2.0);
        approx_eq(w_up.component(&[2]), 3.0);
        approx_eq(w_up.component(&[3]), 4.0);
    }

    #[test]
    fn raise_then_lower_roundtrip() {
        let eta = minkowski();
        let eta_inv = invert_metric(&eta);
        let v = Tensor::<1, 0>::from_vec(4, vec![3.0, -1.0, 4.0, 1.5]);

        // lower then raise
        let v_low: Tensor<0, 1> = lower_index(&v, &eta, 0);
        let v_back: Tensor<1, 0> = raise_index(&v_low, &eta_inv, 0);

        for i in 0..4 {
            approx_eq(v_back.component(&[i]), v.component(&[i]));
        }
    }

    #[test]
    fn raise_then_lower_schwarzschild() {
        let r = 10.0_f64;
        let f = 1.0 - 2.0 / r;
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r);
        let g_inv = invert_metric(&g);

        let v = Tensor::<1, 0>::from_vec(4, vec![1.0, 2.0, 3.0, 4.0]);
        let v_low: Tensor<0, 1> = lower_index(&v, &g, 0);
        let v_back: Tensor<1, 0> = raise_index(&v_low, &g_inv, 0);

        for i in 0..4 {
            approx_eq(v_back.component(&[i]), v.component(&[i]));
        }
    }

    #[test]
    fn lower_rank_20_tensor() {
        // T^{μν} in 2D, lower index 0 → T_α^ν = g_{αμ} T^{μν}
        // Result is Tensor<1, 1> with upper=[ν], lower=[α]
        let eta = minkowski();
        let mut t = Tensor::<2, 0>::new(4);
        t.set_component(&[0, 0], 1.0);
        t.set_component(&[0, 1], 2.0);
        t.set_component(&[1, 0], 3.0);
        t.set_component(&[1, 1], 4.0);

        // Lower upper index 0: R^ν_α = g_{αμ} T^{μν}
        // R^0_0 = g_{00}T^{00} + g_{01}T^{10} + ... = -1*1 + 0 = -1
        // R^1_0 = g_{00}T^{01} + ... = -1*2 = -2
        // R^0_1 = g_{11}T^{10} + ... = 1*3 = 3
        // R^1_1 = g_{11}T^{11} + ... = 1*4 = 4
        let r: Tensor<1, 1> = lower_index(&t, &eta, 0);
        approx_eq(r.component(&[0, 0]), -1.0);
        approx_eq(r.component(&[1, 0]), -2.0);
        approx_eq(r.component(&[0, 1]), 3.0);
        approx_eq(r.component(&[1, 1]), 4.0);
    }
}
