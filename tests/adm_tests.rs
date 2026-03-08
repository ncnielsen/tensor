#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::adm::{AdmState, ExtrinsicCurvature};
use tensor::adm_rhs::{adm_rhs_geodesic, hamiltonian_constraint, k_squared};
use tensor::Tensor;

// ── ExtrinsicCurvature ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_extrinsic_curvature_trace_flat() {
    // K = diag(1, 2, 3), γ^{ij} = δ^{ij} → trace = 6
    let k = ExtrinsicCurvature::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ]);
    let gamma_inv = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let tr = k.trace(&gamma_inv);
    assert!((tr.result - 6.0).abs() < 1e-14);
}

#[test]
#[serial]
fn test_extrinsic_curvature_raise_first() {
    // K_{ij} = δ_{ij}, γ^{ij} = δ^{ij} → K^i_j = δ^i_j
    let k = ExtrinsicCurvature::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let k_up = k.raise_first(&gamma_inv);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (k_up.component(&[i, j]).result - expected).abs() < 1e-14,
                "K^{}_{} = {} ≠ {}",
                i, j, k_up.component(&[i, j]).result, expected
            );
        }
    }
}

// ── AdmState ──────────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_adm_state_flat() {
    let s = AdmState::flat();
    // γ_{ij} = δ_{ij}
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(s.gamma.component(&[i, j]).result, expected);
        }
    }
    // K_{ij} = 0
    for c in &s.k.components {
        assert_eq!(c.result, 0.0);
    }
    assert_eq!(s.alpha, 1.0);
    assert_eq!(s.beta, [0.0; 3]);
}

#[test]
#[serial]
fn test_gamma_inv_flat() {
    let s = AdmState::flat();
    let g_inv = s.gamma_inv().expect("flat metric is invertible");
    // Should recover δ^{ij}
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g_inv.component(&[i, j]).result - expected).abs() < 1e-12,
                "γ^{}_{} = {} ≠ {}",
                i, j, g_inv.component(&[i, j]).result, expected
            );
        }
    }
}

// ── Geodesic RHS — flat initial data ─────────────────────────────────────────

/// Flat Minkowski: K = 0, R_{ij} = 0.
/// Geodesic RHS should give ∂_t γ = 0, ∂_t K = 0.
#[test]
#[serial]
fn test_geodesic_rhs_flat() {
    let state = AdmState::flat();
    let gamma_inv = state.gamma_inv().unwrap();
    let ricci_zero: Tensor<0, 2> = Tensor::from_f64(3, vec![0.0; 9]);

    let rhs = adm_rhs_geodesic(&state, &gamma_inv, &ricci_zero);

    for c in &rhs.dgamma_dt.components {
        assert_eq!(c.result, 0.0, "∂_t γ should be zero for flat K=0 data");
    }
    for c in &rhs.dk_dt.components {
        assert_eq!(c.result, 0.0, "∂_t K should be zero for flat data");
    }
}

/// Non-trivial K: K_{ij} = ε δ_{ij}, R_{ij} = 0 (flat space).
/// ∂_t γ_{ij} = -2 K_{ij} = -2ε δ_{ij}
/// ∂_t K_{ij} = K K_{ij} - 2 K_{ik} K^k_j = (3ε)(ε δ_{ij}) - 2 ε² δ_{ij} = ε² δ_{ij}
#[test]
#[serial]
fn test_geodesic_rhs_isotropic_k() {
    let eps = 0.1_f64;
    let mut state = AdmState::flat();
    state.k = ExtrinsicCurvature::from_f64(3, vec![
        eps, 0.0, 0.0,
        0.0, eps, 0.0,
        0.0, 0.0, eps,
    ]);
    let gamma_inv = state.gamma_inv().unwrap();
    let ricci_zero: Tensor<0, 2> = Tensor::from_f64(3, vec![0.0; 9]);

    let rhs = adm_rhs_geodesic(&state, &gamma_inv, &ricci_zero);

    // ∂_t γ_{ij} = -2ε δ_{ij}
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { -2.0 * eps } else { 0.0 };
            assert!(
                (rhs.dgamma_dt.component(i, j).result - expected).abs() < 1e-14,
                "∂_t γ_{{{i},{j}}} = {} ≠ {expected}",
                rhs.dgamma_dt.component(i, j).result
            );
        }
    }

    // ∂_t K_{ij} = ε² δ_{ij}
    // K = 3ε, K_{ij} = ε δ_{ij}, K^k_j = ε δ^k_j
    // ∂_t K_{ii} = 3ε·ε - 2·ε·ε = ε²
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { eps * eps } else { 0.0 };
            assert!(
                (rhs.dk_dt.component(i, j).result - expected).abs() < 1e-14,
                "∂_t K_{{{i},{j}}} = {} ≠ {expected}",
                rhs.dk_dt.component(i, j).result
            );
        }
    }
}

// ── Hamiltonian constraint — flat vacuum ──────────────────────────────────────

/// For flat Minkowski: ³R = 0, K = 0, K_{ij} K^{ij} = 0, ρ = 0.
/// H = 0 + 0 - 0 - 0 = 0.
#[test]
#[serial]
fn test_hamiltonian_constraint_flat() {
    use aad::number::Number;
    let h = hamiltonian_constraint(
        Number::new(0.0),
        Number::new(0.0),
        Number::new(0.0),
        0.0,
    );
    assert_eq!(h.result, 0.0);
}

/// Isotropic K_{ij} = ε δ_{ij}, flat space (³R = 0, ρ = 0).
/// K = 3ε, K² = 9ε², K_{ij}K^{ij} = 3ε².
/// H = 0 + 9ε² − 3ε² = 6ε².
#[test]
#[serial]
fn test_hamiltonian_constraint_isotropic_k() {
    use aad::number::Number;
    let eps = 0.1_f64;
    let k = ExtrinsicCurvature::from_f64(3, vec![
        eps, 0.0, 0.0,
        0.0, eps, 0.0,
        0.0, 0.0, eps,
    ]);
    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let k_tr = k.trace(&gamma_inv);        // 3ε
    let k_sq = k_squared(&k, &gamma_inv);  // 3ε²

    let h = hamiltonian_constraint(Number::new(0.0), k_tr, k_sq, 0.0);
    let expected = 6.0 * eps * eps;
    assert!(
        (h.result - expected).abs() < 1e-14,
        "H = {} ≠ {expected}",
        h.result
    );
}
