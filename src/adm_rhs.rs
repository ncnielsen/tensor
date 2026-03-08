use aad::number::Number;

use crate::adm::{AdmState, ExtrinsicCurvature};
use crate::christoffel::Christoffel;
use crate::tensor::{flat_index, Tensor};

// ── Output type ───────────────────────────────────────────────────────────────

/// Time derivatives of the ADM state variables at one grid point.
pub struct AdmRhs {
    /// ∂_t γ_{ij}
    pub dgamma_dt: ExtrinsicCurvature,
    /// ∂_t K_{ij}
    pub dk_dt: ExtrinsicCurvature,
}

// ── Geodesic-slicing vacuum RHS ───────────────────────────────────────────────

/// Vacuum ADM right-hand sides under geodesic slicing (α = 1, β^i = 0).
///
/// Equations:
///   ∂_t γ_{ij} = −2 K_{ij}
///   ∂_t K_{ij} = ³R_{ij} + K K_{ij} − 2 K_{ik} K^k_j
///
/// where K = γ^{ij} K_{ij} (scalar trace) and K^k_j = γ^{ki} K_{ij}.
///
/// # Arguments
/// - `state`     — current ADM state (γ, K); α and β are ignored
/// - `gamma_inv` — inverse spatial metric γ^{ij}  (Tensor<2,0>)
/// - `ricci_3d`  — spatial Ricci tensor ³R_{ij}   (Tensor<0,2>)
pub fn adm_rhs_geodesic(
    state: &AdmState,
    gamma_inv: &Tensor<2, 0>,
    ricci_3d: &Tensor<0, 2>,
) -> AdmRhs {
    let dim = state.gamma.dim;
    assert_eq!(dim, 3, "Geodesic ADM RHS expects dim = 3");
    assert_eq!(gamma_inv.dim, 3);
    assert_eq!(ricci_3d.dim, 3);

    let k_trace = state.k.trace(gamma_inv);
    let k_mixed = state.k.raise_first(gamma_inv);

    // ∂_t γ_{ij} = −2 K_{ij}
    let dgamma_components: Vec<Number> =
        state.k.components.iter().map(|&k_ij| k_ij * -2.0).collect();
    let dgamma_dt = ExtrinsicCurvature::new(dim, dgamma_components);

    // ∂_t K_{ij} = ³R_{ij} + K K_{ij} − 2 K_{ik} K^k_j
    let mut dk_components = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            let r_ij = ricci_3d.component(&[i, j]);
            let k_ij = state.k.component(i, j);

            // Σ_k K_{ik} K^k_j
            let mut k2_iter =
                (0..dim).map(|k| state.k.component(i, k) * k_mixed.component(&[k, j]));
            let k2_first = k2_iter.next().unwrap();
            let k2 = k2_iter.fold(k2_first, |acc, x| acc + x);

            dk_components.push(r_ij + k_trace * k_ij - k2 * 2.0);
        }
    }
    let dk_dt = ExtrinsicCurvature::new(dim, symmetrize_3x3(dk_components));

    AdmRhs { dgamma_dt, dk_dt }
}

// ── General vacuum RHS ────────────────────────────────────────────────────────

/// Vacuum ADM right-hand sides for arbitrary lapse α and shift β^i.
///
/// Full ADM evolution equations:
///
///   ∂_t γ_{ij} = −2α K_{ij} + ∇_i β_j + ∇_j β_i
///
///   ∂_t K_{ij} = −∇_i ∇_j α + α(³R_{ij} + K K_{ij} − 2 K_{ik} K^k_j)
///               + β^k ∂_k K_{ij} + K_{ik} ∂_j β^k + K_{kj} ∂_i β^k
///
/// where:
///   ∇_i β_j = ∂_i β_j − Γ^k_{ij} β_k   (β_j = γ_{jl} β^l)
///   ∇_i ∇_j α = ∂_i ∂_j α − Γ^k_{ij} ∂_k α
///
/// # Arguments
/// - `state`       — current ADM state (γ, K, α, β^i)
/// - `gamma_inv`   — γ^{ij}
/// - `ricci_3d`    — ³R_{ij}
/// - `christoffel` — 3D spatial Γ^k_{ij}
/// - `d_alpha`     — ∂_k α, index k = 0..2
/// - `d2_alpha`    — ∂_i ∂_j α, row-major [i][j]
/// - `d_beta`      — ∂_k β^i, layout [i][k] = ∂_k β^i
/// - `d_k`         — ∂_k K_{ij}, flat [i*9 + j*3 + k]
pub fn adm_rhs_vacuum(
    state: &AdmState,
    gamma_inv: &Tensor<2, 0>,
    ricci_3d: &Tensor<0, 2>,
    christoffel: &Christoffel,
    d_alpha: [f64; 3],
    d2_alpha: [[f64; 3]; 3],
    d_beta: [[f64; 3]; 3],
    d_k: &[f64],
) -> AdmRhs {
    let dim = state.gamma.dim;
    assert_eq!(dim, 3);
    assert_eq!(d_k.len(), dim.pow(3));

    let alpha = state.alpha;
    let beta = &state.beta;

    let k_trace = state.k.trace(gamma_inv);
    let k_mixed = state.k.raise_first(gamma_inv);

    // β_j = γ_{jl} β^l  (lowered shift)
    let mut beta_lower = Vec::with_capacity(dim);
    for j in 0..dim {
        let mut iter = (0..dim).map(|l| state.gamma.component(&[j, l]) * beta[l]);
        let first = iter.next().unwrap();
        beta_lower.push(iter.fold(first, |acc, x| acc + x));
    }

    // ∇_i β_j = ∂_i(γ_{jl} β^l) − Γ^k_{ij} β_k
    // We use ∂_i β_j ≈ Σ_l γ_{jl} ∂_i β^l  (primary ADM variables are β^i)
    let mut cov_beta = vec![vec![Number::new(0.0); dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut d_iter = (0..dim).map(|l| state.gamma.component(&[j, l]) * d_beta[l][i]);
            let d_first = d_iter.next().unwrap();
            let d_beta_j_i = d_iter.fold(d_first, |acc, x| acc + x);

            let mut g_iter = (0..dim).map(|k| christoffel.component(k, i, j) * beta_lower[k]);
            let g_first = g_iter.next().unwrap();
            let gamma_term = g_iter.fold(g_first, |acc, x| acc + x);

            cov_beta[i][j] = d_beta_j_i - gamma_term;
        }
    }

    // ∂_t γ_{ij} = −2α K_{ij} + ∇_i β_j + ∇_j β_i
    let mut dgamma_components = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            dgamma_components.push(
                state.k.component(i, j) * (-2.0 * alpha) + cov_beta[i][j] + cov_beta[j][i],
            );
        }
    }
    let dgamma_dt = ExtrinsicCurvature::new(dim, symmetrize_3x3(dgamma_components));

    // ∂_t K_{ij}
    let mut dk_components = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            let r_ij = ricci_3d.component(&[i, j]);
            let k_ij = state.k.component(i, j);

            // Σ_k K_{ik} K^k_j
            let mut k2_iter =
                (0..dim).map(|k| state.k.component(i, k) * k_mixed.component(&[k, j]));
            let k2_first = k2_iter.next().unwrap();
            let k2 = k2_iter.fold(k2_first, |acc, x| acc + x);

            // −∇_i ∇_j α = −∂_i ∂_j α + Γ^k_{ij} ∂_k α
            let gamma_da: f64 = (0..dim)
                .map(|k| christoffel.component(k, i, j).result * d_alpha[k])
                .sum();
            let neg_hess_alpha = Number::new(-d2_alpha[i][j] + gamma_da);

            // Lie derivative: β^k ∂_k K_{ij} + K_{ik} ∂_j β^k + K_{kj} ∂_i β^k
            let beta_dk: f64 = (0..dim)
                .map(|k| beta[k] * d_k[flat_index(&[i, j, k], dim)])
                .sum();
            let mut k_db_iter = (0..dim).map(|k| state.k.component(i, k) * d_beta[k][j]);
            let k_db_first = k_db_iter.next().unwrap();
            let k_db = k_db_iter.fold(k_db_first, |acc, x| acc + x);
            let mut kb_d_iter = (0..dim).map(|k| state.k.component(k, j) * d_beta[k][i]);
            let kb_d_first = kb_d_iter.next().unwrap();
            let kb_d = kb_d_iter.fold(kb_d_first, |acc, x| acc + x);
            let lie_k = Number::new(beta_dk) + k_db + kb_d;

            dk_components.push(
                neg_hess_alpha + (r_ij + k_trace * k_ij - k2 * 2.0) * alpha + lie_k,
            );
        }
    }
    let dk_dt = ExtrinsicCurvature::new(dim, symmetrize_3x3(dk_components));

    AdmRhs { dgamma_dt, dk_dt }
}

// ── ADM Constraints ───────────────────────────────────────────────────────────

/// Hamiltonian constraint: H = ³R + K² − K_{ij} K^{ij} − 16π ρ.
///
/// Should be zero for a valid data set; monitor during evolution to gauge numerical error.
pub fn hamiltonian_constraint(
    ricci_scalar_3d: Number,
    k_trace: Number,
    k_sq: Number,
    rho: f64,
) -> Number {
    ricci_scalar_3d + k_trace * k_trace - k_sq
        - Number::new(16.0 * std::f64::consts::PI * rho)
}

/// K_{ij} K^{ij} = Σ_{i,j} K_{ij} K^{ij}.
pub fn k_squared(k: &ExtrinsicCurvature, gamma_inv: &Tensor<2, 0>) -> Number {
    let dim = k.dim;
    let k_up = k.raise_first(gamma_inv);
    let mut sum: Option<Number> = None;
    for i in 0..dim {
        for j in 0..dim {
            let term = k.component(i, j) * k_up.component(&[j, i]);
            sum = Some(match sum {
                None => term,
                Some(acc) => acc + term,
            });
        }
    }
    sum.unwrap()
}

/// Momentum constraint: M^i = ∇_j (K^{ij} − γ^{ij} K) − 8π J^i.
///
/// Returns a 3-vector. Should be zero for valid data; monitor during evolution.
///
/// # Arguments
/// - `k`          — K_{ij}
/// - `gamma_inv`  — γ^{ij}
/// - `k_trace`    — K (scalar)
/// - `christoffel`— ³Γ^k_{ij}
/// - `d_k_up`     — ∂_j K^{ij}, layout [i][j]
/// - `d_k_trace`  — ∂_j K, index j = 0..2
/// - `j_vec`      — matter current J^i (zero for vacuum)
pub fn momentum_constraint(
    k: &ExtrinsicCurvature,
    gamma_inv: &Tensor<2, 0>,
    _k_trace: Number,
    christoffel: &Christoffel,
    d_k_up: &[[f64; 3]; 3],
    d_k_trace: [f64; 3],
    j_vec: [f64; 3],
) -> [Number; 3] {
    let dim = k.dim;
    let k_up = k.raise_first(gamma_inv);
    let mut result = [Number::new(0.0); 3];

    for i in 0..dim {
        // ∂_j K^{ij}
        let div_k_up: f64 = (0..dim).map(|j| d_k_up[i][j]).sum();

        // γ^{ij} ∂_j K
        let div_gamma_k: f64 = (0..dim)
            .map(|j| gamma_inv.component(&[i, j]).result * d_k_trace[j])
            .sum();

        // Christoffel corrections: Γ^i_{jk} K^{kj} + Γ^j_{jk} K^{ik}
        let mut gamma_corr: Option<Number> = None;
        for j in 0..dim {
            for k_idx in 0..dim {
                let g1 = christoffel.component(i, j, k_idx) * k_up.component(&[k_idx, j]);
                let g2 = christoffel.component(j, j, k_idx) * k_up.component(&[i, k_idx]);
                let term = g1 + g2;
                gamma_corr = Some(match gamma_corr {
                    None => term,
                    Some(acc) => acc + term,
                });
            }
        }

        result[i] = Number::new(div_k_up - div_gamma_k)
            + gamma_corr.unwrap()
            - Number::new(8.0 * std::f64::consts::PI * j_vec[i]);
    }

    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Symmetrize a 3×3 component vector in-place: A_{ij} ← (A_{ij} + A_{ji}) / 2.
fn symmetrize_3x3(mut v: Vec<Number>) -> Vec<Number> {
    const DIM: usize = 3;
    for i in 0..DIM {
        for j in i + 1..DIM {
            let a = v[flat_index(&[i, j], DIM)];
            let b = v[flat_index(&[j, i], DIM)];
            let sym = (a + b) * 0.5;
            v[flat_index(&[i, j], DIM)] = sym;
            v[flat_index(&[j, i], DIM)] = sym;
        }
    }
    v
}
