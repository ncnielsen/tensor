use tensor_core::{
    christoffel::Christoffel,
    curvature::{riemann, ricci_scalar, ricci_tensor, ChristoffelDerivative},
    metric::invert_metric,
    tensor::Tensor,
};

/// Spatial dimension for 3+1 ADM decomposition.
const DIM: usize = 3;

// ---------------------------------------------------------------------------
// ExtrinsicCurvature
// ---------------------------------------------------------------------------

/// Extrinsic curvature K_{ij} — symmetric spatial rank-(0,2) tensor.
///
/// Stores `dim^2` components in flat row-major layout. `set_component` enforces
/// symmetry: setting K_{ij} also sets K_{ji}.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrinsicCurvature {
    dim: usize,
    data: Vec<f64>,
}

impl ExtrinsicCurvature {
    /// Zero extrinsic curvature for a manifold of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; dim * dim],
        }
    }

    /// Create from a flat data vector (dim^2 components, row-major).
    pub fn from_vec(dim: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), dim * dim, "expected {} components", dim * dim);
        Self { dim, data }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// K_{ij}
    pub fn component(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dim + j]
    }

    /// Set K_{ij} = value and enforce symmetry K_{ji} = value.
    pub fn set_component(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.dim + j] = value;
        self.data[j * self.dim + i] = value;
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// AdmState
// ---------------------------------------------------------------------------

/// ADM state at a single spatial point: 3-metric, extrinsic curvature,
/// lapse, and shift.
///
/// Represents the 3+1 dynamical variables (γ_{ij}, K_{ij}, α, β^i).
/// The inverse metric γ^{ij} is precomputed on construction.
#[derive(Debug, Clone)]
pub struct AdmState {
    /// 3-metric γ_{ij} (dim=3).
    pub gamma: Tensor<0, 2>,
    /// Inverse 3-metric γ^{ij}.
    pub gamma_inv: Tensor<2, 0>,
    /// Extrinsic curvature K_{ij}.
    pub k: ExtrinsicCurvature,
    /// Lapse α.
    pub alpha: f64,
    /// Shift β^i (upper index, contravariant).
    pub beta: [f64; 3],
}

impl AdmState {
    /// Construct from metric + extrinsic curvature + gauge.
    /// Computes γ^{ij} automatically via Gaussian elimination.
    pub fn new(
        gamma: Tensor<0, 2>,
        k: ExtrinsicCurvature,
        alpha: f64,
        beta: [f64; 3],
    ) -> Self {
        let gamma_inv = invert_metric(&gamma);
        Self {
            gamma,
            gamma_inv,
            k,
            alpha,
            beta,
        }
    }
}

// ---------------------------------------------------------------------------
// GaugeDeriv
// ---------------------------------------------------------------------------

/// Spatial derivatives of the gauge variables (lapse α and shift β^i).
///
/// Provided by the caller — computed by FD from neighboring grid values in the
/// grid regime (phase 3), or set analytically for known gauge fields in tests.
#[derive(Debug, Clone)]
pub struct GaugeDeriv {
    /// ∂_i α (i = 0,1,2).
    pub partial_alpha: [f64; 3],
    /// ∂_i ∂_j α: `partial2_alpha[i][j]`.
    pub partial2_alpha: [[f64; 3]; 3],
    /// Shift Jacobian: `partial_beta[k][j]` = ∂_j β^k.
    pub partial_beta: [[f64; 3]; 3],
}

impl GaugeDeriv {
    /// All-zero gauge derivatives (constant lapse, zero shift).
    pub fn zero() -> Self {
        Self {
            partial_alpha: [0.0; 3],
            partial2_alpha: [[0.0; 3]; 3],
            partial_beta: [[0.0; 3]; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// K = γ^{ij} K_{ij} (trace of extrinsic curvature).
fn k_trace(k: &ExtrinsicCurvature, gamma_inv: &Tensor<2, 0>) -> f64 {
    let dim = k.dim();
    let mut s = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            s += gamma_inv.component(&[i, j]) * k.component(i, j);
        }
    }
    s
}

/// K_{ij} K^{ij} = γ^{ik} γ^{jl} K_{ij} K_{kl}.
fn k_contracted_square(k: &ExtrinsicCurvature, gamma_inv: &Tensor<2, 0>) -> f64 {
    let dim = k.dim();
    let mut s = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            for ki in 0..dim {
                for l in 0..dim {
                    s += gamma_inv.component(&[i, ki])
                        * gamma_inv.component(&[j, l])
                        * k.component(i, j)
                        * k.component(ki, l);
                }
            }
        }
    }
    s
}

/// K_{im} K^m_j = K_{im} γ^{ml} K_{lj}, returned as flat `dim×dim` array.
///
/// Index layout: `result[i * dim + j]` = K_{im} K^m_j.
fn kk_product(k: &ExtrinsicCurvature, gamma_inv: &Tensor<2, 0>) -> Vec<f64> {
    let dim = k.dim();
    let mut kk = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut val = 0.0;
            for m in 0..dim {
                // K^m_j = γ^{ml} K_{lj}
                let km_j: f64 = (0..dim)
                    .map(|l| gamma_inv.component(&[m, l]) * k.component(l, j))
                    .sum();
                val += k.component(i, m) * km_j;
            }
            kk[i * dim + j] = val;
        }
    }
    kk
}

// ---------------------------------------------------------------------------
// ADM RHS — geodesic slicing (α = 1, β^i = 0)
// ---------------------------------------------------------------------------

/// ADM evolution equations for geodesic slicing (α = 1, β^i = 0).
///
/// Returns `(∂_t γ_{ij}, ∂_t K_{ij})`.
///
/// Evolution equations:
/// ```text
/// ∂_t γ_{ij} = −2 K_{ij}
/// ∂_t K_{ij} = R_{ij}^{(3)} + K K_{ij} − 2 K_{im} K^m_j
/// ```
///
/// `christoffel` and `christoffel_deriv` encode the 3D spatial geometry;
/// they are supplied by the caller (Enzyme in the function regime, FD in
/// the grid regime — phase 3).
pub fn adm_rhs_geodesic(
    state: &AdmState,
    christoffel: &Christoffel,
    christoffel_deriv: &ChristoffelDerivative,
) -> (Tensor<0, 2>, ExtrinsicCurvature) {
    let dim = state.gamma.dim();
    assert_eq!(dim, DIM, "ADM requires 3D spatial metric");

    // ∂_t γ_{ij} = -2 K_{ij}
    let mut gamma_dot = Tensor::<0, 2>::new(dim);
    for i in 0..dim {
        for j in 0..dim {
            gamma_dot.set_component(&[i, j], -2.0 * state.k.component(i, j));
        }
    }

    // 3D Ricci tensor R_{ij}^{(3)}
    let riem = riemann(christoffel, christoffel_deriv);
    let ricci = ricci_tensor(&riem);

    // K = γ^{ij} K_{ij}
    let k_tr = k_trace(&state.k, &state.gamma_inv);

    // K_{im} K^m_j
    let kk = kk_product(&state.k, &state.gamma_inv);

    // ∂_t K_{ij} = R_{ij} + K K_{ij} − 2 K_{im} K^m_j
    let mut k_dot = ExtrinsicCurvature::new(dim);
    for i in 0..dim {
        for j in i..dim {
            let val = ricci.component(&[i, j])
                + k_tr * state.k.component(i, j)
                - 2.0 * kk[i * dim + j];
            k_dot.set_component(i, j, val);
        }
    }

    (gamma_dot, k_dot)
}

// ---------------------------------------------------------------------------
// ADM RHS — vacuum (general lapse/shift)
// ---------------------------------------------------------------------------

/// ADM evolution equations for vacuum with general lapse α and shift β^i.
///
/// Returns `(∂_t γ_{ij}, ∂_t K_{ij})`.
///
/// Evolution equations:
/// ```text
/// ∂_t γ_{ij} = −2α K_{ij} + β^k ∂_k γ_{ij} + γ_{ik} ∂_j β^k + γ_{jk} ∂_i β^k
///
/// ∂_t K_{ij} = −D_i D_j α + α (R_{ij} + K K_{ij} − 2 K_{im} K^m_j)
///              + β^k ∂_k K_{ij} + K_{ik} ∂_j β^k + K_{kj} ∂_i β^k
/// ```
///
/// where `D_i D_j α = ∂_i ∂_j α − Γ^k_{ij} ∂_k α`.
///
/// `partial_gamma[k]` = ∂_k γ_{ij} and `partial_k[k]` = ∂_k K_{ij} (both
/// length-3 slices, one per spatial direction). These come from FD on
/// neighboring grid points in the grid regime (phase 3).
pub fn adm_rhs_vacuum(
    state: &AdmState,
    christoffel: &Christoffel,
    christoffel_deriv: &ChristoffelDerivative,
    partial_gamma: &[Tensor<0, 2>],
    partial_k: &[Tensor<0, 2>],
    gauge: &GaugeDeriv,
) -> (Tensor<0, 2>, ExtrinsicCurvature) {
    let dim = state.gamma.dim();
    assert_eq!(dim, DIM, "ADM requires 3D spatial metric");
    assert_eq!(partial_gamma.len(), dim);
    assert_eq!(partial_k.len(), dim);

    // 3D Ricci tensor R_{ij}^{(3)}
    let riem = riemann(christoffel, christoffel_deriv);
    let ricci = ricci_tensor(&riem);

    // K = γ^{ij} K_{ij}
    let k_tr = k_trace(&state.k, &state.gamma_inv);

    // K_{im} K^m_j
    let kk = kk_product(&state.k, &state.gamma_inv);

    let alpha = state.alpha;

    // --- ∂_t γ_{ij} = -2α K_{ij} + L_β γ_{ij} ---
    //
    // L_β γ_{ij} = β^k ∂_k γ_{ij} + γ_{ik} ∂_j β^k + γ_{jk} ∂_i β^k
    let mut gamma_dot = Tensor::<0, 2>::new(dim);
    for i in 0..dim {
        for j in i..dim {
            // Lie shift: β^k ∂_k γ_{ij}
            let lie: f64 = (0..dim)
                .map(|k| state.beta[k] * partial_gamma[k].component(&[i, j]))
                .sum();
            // γ_{ik} ∂_j β^k + γ_{jk} ∂_i β^k
            let sym: f64 = (0..dim)
                .map(|k| {
                    state.gamma.component(&[i, k]) * gauge.partial_beta[k][j]
                        + state.gamma.component(&[j, k]) * gauge.partial_beta[k][i]
                })
                .sum();
            let val = -2.0 * alpha * state.k.component(i, j) + lie + sym;
            gamma_dot.set_component(&[i, j], val);
            gamma_dot.set_component(&[j, i], val);
        }
    }

    // --- ∂_t K_{ij} ---
    //
    // = -D_i D_j α + α (R_{ij} + K K_{ij} - 2 K_{im} K^m_j) + L_β K_{ij}
    //
    // D_i D_j α = ∂_i ∂_j α - Γ^k_{ij} ∂_k α
    //
    // L_β K_{ij} = β^k ∂_k K_{ij} + K_{ik} ∂_j β^k + K_{kj} ∂_i β^k
    let mut k_dot = ExtrinsicCurvature::new(dim);
    for i in 0..dim {
        for j in i..dim {
            // Hessian of lapse
            let d2_alpha = gauge.partial2_alpha[i][j]
                - (0..dim)
                    .map(|k| christoffel.component(k, i, j) * gauge.partial_alpha[k])
                    .sum::<f64>();

            // Lie derivative of K along β
            let lie: f64 = (0..dim)
                .map(|k| state.beta[k] * partial_k[k].component(&[i, j]))
                .sum();
            let sym: f64 = (0..dim)
                .map(|k| {
                    state.k.component(i, k) * gauge.partial_beta[k][j]
                        + state.k.component(k, j) * gauge.partial_beta[k][i]
                })
                .sum();

            let val = -d2_alpha
                + alpha
                    * (ricci.component(&[i, j]) + k_tr * state.k.component(i, j)
                        - 2.0 * kk[i * dim + j])
                + lie
                + sym;
            k_dot.set_component(i, j, val);
        }
    }

    (gamma_dot, k_dot)
}

// ---------------------------------------------------------------------------
// Hamiltonian constraint
// ---------------------------------------------------------------------------

/// Hamiltonian (energy) constraint: H = R^{(3)} + K² − K_{ij} K^{ij}.
///
/// Vanishes for physically consistent initial data. Used to monitor
/// constraint violation during time evolution.
pub fn hamiltonian_constraint(
    state: &AdmState,
    christoffel: &Christoffel,
    christoffel_deriv: &ChristoffelDerivative,
) -> f64 {
    // 3D Ricci scalar R^{(3)}
    let riem = riemann(christoffel, christoffel_deriv);
    let ric = ricci_tensor(&riem);
    let r3 = ricci_scalar(&ric, &state.gamma_inv);

    // K = γ^{ij} K_{ij}
    let k_tr = k_trace(&state.k, &state.gamma_inv);

    // K_{ij} K^{ij} = γ^{ik} γ^{jl} K_{ij} K_{kl}
    let kk_sq = k_contracted_square(&state.k, &state.gamma_inv);

    r3 + k_tr * k_tr - kk_sq
}

// ---------------------------------------------------------------------------
// Momentum constraint
// ---------------------------------------------------------------------------

/// Momentum constraint: M_i = D_j K^j_i − D_i K.
///
/// Expanded in components:
/// ```text
/// M_i = ∂_j K^j_i + Γ^j_{jl} K^l_i − Γ^l_{ij} K^j_l − ∂_i K
/// ```
///
/// where K^j_i = γ^{jl} K_{li} and ∂_j K^j_i, ∂_i K are computed from
/// `partial_gamma` and `partial_k` provided by the caller.
///
/// `partial_gamma[k]` = ∂_k γ_{ij}, `partial_k[k]` = ∂_k K_{ij}.
pub fn momentum_constraint(
    state: &AdmState,
    christoffel: &Christoffel,
    partial_gamma: &[Tensor<0, 2>],
    partial_k: &[Tensor<0, 2>],
) -> [f64; 3] {
    let dim = state.gamma.dim();
    assert_eq!(dim, DIM);
    assert_eq!(partial_gamma.len(), dim);
    assert_eq!(partial_k.len(), dim);

    // K^j_i = γ^{jl} K_{li}
    let mut k_up = vec![0.0f64; dim * dim]; // k_up[j * dim + i] = K^j_i
    for j in 0..dim {
        for i in 0..dim {
            for l in 0..dim {
                k_up[j * dim + i] +=
                    state.gamma_inv.component(&[j, l]) * state.k.component(l, i);
            }
        }
    }

    let mut m = [0.0f64; 3];
    for i in 0..dim {
        // Term 1: ∂_j K^j_i = ∂_j(γ^{jl} K_{li})
        //       = sum_{j,l} [(∂_j γ^{jl}) K_{li} + γ^{jl} ∂_j K_{li}]
        //
        // ∂_j γ^{jl} = -sum_{a,b} γ^{ja} γ^{lb} ∂_j γ_{ab}
        let mut div_kup = 0.0f64;
        for j in 0..dim {
            for l in 0..dim {
                let mut d_ginv_jl = 0.0f64;
                for a in 0..dim {
                    for b in 0..dim {
                        d_ginv_jl -= state.gamma_inv.component(&[j, a])
                            * state.gamma_inv.component(&[l, b])
                            * partial_gamma[j].component(&[a, b]);
                    }
                }
                div_kup += d_ginv_jl * state.k.component(l, i);
                div_kup +=
                    state.gamma_inv.component(&[j, l]) * partial_k[j].component(&[l, i]);
            }
        }

        // Term 2: Γ^j_{jl} K^l_i
        let mut gamma_trace_k = 0.0f64;
        for j in 0..dim {
            for l in 0..dim {
                gamma_trace_k += christoffel.component(j, j, l) * k_up[l * dim + i];
            }
        }

        // Term 3: -Γ^l_{ij} K^j_l
        let mut gamma_k = 0.0f64;
        for j in 0..dim {
            for l in 0..dim {
                gamma_k += christoffel.component(l, i, j) * k_up[j * dim + l];
            }
        }

        // Term 4: -∂_i K = -∂_i(γ^{mn} K_{mn})
        //
        // ∂_i γ^{mn} = -sum_{a,b} γ^{ma} γ^{nb} ∂_i γ_{ab}
        let mut di_k = 0.0f64;
        for mn0 in 0..dim {
            for mn1 in 0..dim {
                let mut d_ginv_mn = 0.0f64;
                for a in 0..dim {
                    for b in 0..dim {
                        d_ginv_mn -= state.gamma_inv.component(&[mn0, a])
                            * state.gamma_inv.component(&[mn1, b])
                            * partial_gamma[i].component(&[a, b]);
                    }
                }
                di_k += d_ginv_mn * state.k.component(mn0, mn1);
                di_k += state.gamma_inv.component(&[mn0, mn1])
                    * partial_k[i].component(&[mn0, mn1]);
            }
        }

        m[i] = div_kup + gamma_trace_k - gamma_k - di_k;
    }

    m
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_core::{
        christoffel::Christoffel, curvature::ChristoffelDerivative, tensor::Tensor,
    };

    const TOL: f64 = 1e-12;

    // Helper: flat space (γ = I, K = 0, α = 1, β = 0, Γ = 0, ∂Γ = 0)
    fn flat_state() -> (AdmState, Christoffel, ChristoffelDerivative) {
        let dim = 3;
        let mut gamma = Tensor::<0, 2>::new(dim);
        for i in 0..dim {
            gamma.set_component(&[i, i], 1.0);
        }
        let k = ExtrinsicCurvature::new(dim);
        let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
        let christoffel = Christoffel::new(dim);
        let christoffel_deriv = ChristoffelDerivative::new(dim);
        (state, christoffel, christoffel_deriv)
    }

    // Helper: zero spatial partial derivatives
    fn zero_partials() -> (Vec<Tensor<0, 2>>, Vec<Tensor<0, 2>>) {
        let dim = 3;
        let pg: Vec<Tensor<0, 2>> = (0..dim).map(|_| Tensor::<0, 2>::new(dim)).collect();
        let pk: Vec<Tensor<0, 2>> = (0..dim).map(|_| Tensor::<0, 2>::new(dim)).collect();
        (pg, pk)
    }

    // -- Flat space: geodesic RHS = 0 ----------------------------------------

    #[test]
    fn flat_geodesic_rhs_zero() {
        let (state, christoffel, christoffel_deriv) = flat_state();
        let (gamma_dot, k_dot) = adm_rhs_geodesic(&state, &christoffel, &christoffel_deriv);

        for &v in gamma_dot.as_slice() {
            assert!(v.abs() < TOL, "γ_dot component {} ≠ 0", v);
        }
        for &v in k_dot.as_slice() {
            assert!(v.abs() < TOL, "K_dot component {} ≠ 0", v);
        }
    }

    // -- Flat space: vacuum RHS = 0 ------------------------------------------

    #[test]
    fn flat_vacuum_rhs_zero() {
        let (state, christoffel, christoffel_deriv) = flat_state();
        let (pg, pk) = zero_partials();
        let gauge = GaugeDeriv::zero();
        let (gamma_dot, k_dot) =
            adm_rhs_vacuum(&state, &christoffel, &christoffel_deriv, &pg, &pk, &gauge);

        for &v in gamma_dot.as_slice() {
            assert!(v.abs() < TOL, "γ_dot component {} ≠ 0", v);
        }
        for &v in k_dot.as_slice() {
            assert!(v.abs() < TOL, "K_dot component {} ≠ 0", v);
        }
    }

    // -- Flat space: Hamiltonian constraint = 0 ------------------------------

    #[test]
    fn flat_hamiltonian_constraint_zero() {
        let (state, christoffel, christoffel_deriv) = flat_state();
        let h = hamiltonian_constraint(&state, &christoffel, &christoffel_deriv);
        assert!(h.abs() < TOL, "H = {} ≠ 0 for flat space", h);
    }

    // -- Flat space: momentum constraint = 0 ---------------------------------

    #[test]
    fn flat_momentum_constraint_zero() {
        let (state, christoffel, _) = flat_state();
        let (pg, pk) = zero_partials();
        let m = momentum_constraint(&state, &christoffel, &pg, &pk);
        for (i, &mi) in m.iter().enumerate() {
            assert!(mi.abs() < TOL, "M_{} = {} ≠ 0 for flat space", i, mi);
        }
    }

    // -- K evolution sign: positive K → collapsing γ -------------------------
    //
    // K_{ij} = ε δ_{ij} (isotropic positive extrinsic curvature) in flat space
    // with geodesic slicing: ∂_t γ_{ij} = -2ε δ_{ij} < 0 (contracting).

    #[test]
    fn positive_k_gives_contracting_gamma_dot() {
        let dim = 3;
        let eps = 0.1f64;
        let mut gamma = Tensor::<0, 2>::new(dim);
        for i in 0..dim {
            gamma.set_component(&[i, i], 1.0);
        }
        let mut k = ExtrinsicCurvature::new(dim);
        for i in 0..dim {
            k.set_component(i, i, eps);
        }
        let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
        let christoffel = Christoffel::new(dim);
        let christoffel_deriv = ChristoffelDerivative::new(dim);

        let (gamma_dot, _) = adm_rhs_geodesic(&state, &christoffel, &christoffel_deriv);

        for i in 0..dim {
            let v = gamma_dot.component(&[i, i]);
            assert!(
                v < 0.0,
                "γ_dot_{}{} = {} should be negative (contracting)",
                i,
                i,
                v
            );
            assert!(
                (v + 2.0 * eps).abs() < TOL,
                "γ_dot_{}{} = {}, expected {}",
                i,
                i,
                v,
                -2.0 * eps
            );
        }
    }

    // -- Geodesic ↔ vacuum equivalence at α=1, β=0 ---------------------------
    //
    // With unit lapse and zero shift (and zero partial derivatives), both
    // formulations must produce identical RHS.

    #[test]
    fn geodesic_matches_vacuum_at_unit_lapse() {
        let dim = 3;
        let eps = 0.05f64;
        let mut gamma = Tensor::<0, 2>::new(dim);
        for i in 0..dim {
            gamma.set_component(&[i, i], 1.0);
        }
        let mut k = ExtrinsicCurvature::new(dim);
        k.set_component(0, 0, eps);
        k.set_component(1, 1, 2.0 * eps);
        k.set_component(0, 1, 0.5 * eps);

        let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
        let christoffel = Christoffel::new(dim);
        let christoffel_deriv = ChristoffelDerivative::new(dim);
        let (pg, pk) = zero_partials();
        let gauge = GaugeDeriv::zero();

        let (gd_geo, kd_geo) = adm_rhs_geodesic(&state, &christoffel, &christoffel_deriv);
        let (gd_vac, kd_vac) =
            adm_rhs_vacuum(&state, &christoffel, &christoffel_deriv, &pg, &pk, &gauge);

        for (a, b) in gd_geo.as_slice().iter().zip(gd_vac.as_slice()) {
            assert!((a - b).abs() < TOL, "γ_dot mismatch: {} vs {}", a, b);
        }
        for (a, b) in kd_geo.as_slice().iter().zip(kd_vac.as_slice()) {
            assert!((a - b).abs() < TOL, "K_dot mismatch: {} vs {}", a, b);
        }
    }

    // -- Hamiltonian with isotropic K: R + K² - K_ij K^{ij} ------------------
    //
    // For flat γ = I and K_{ij} = ε δ_{ij}: K = 3ε, K_{ij} K^{ij} = 3ε².
    // H = 0 + (3ε)² - 3ε² = 9ε² - 3ε² = 6ε².

    #[test]
    fn hamiltonian_isotropic_k_analytic() {
        let dim = 3;
        let eps = 0.2f64;
        let mut gamma = Tensor::<0, 2>::new(dim);
        for i in 0..dim {
            gamma.set_component(&[i, i], 1.0);
        }
        let mut k = ExtrinsicCurvature::new(dim);
        for i in 0..dim {
            k.set_component(i, i, eps);
        }
        let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
        let christoffel = Christoffel::new(dim);
        let christoffel_deriv = ChristoffelDerivative::new(dim);

        let h = hamiltonian_constraint(&state, &christoffel, &christoffel_deriv);
        let expected = 6.0 * eps * eps;
        assert!(
            (h - expected).abs() < TOL,
            "H = {}, expected {} = 6ε²",
            h,
            expected
        );
    }
}
