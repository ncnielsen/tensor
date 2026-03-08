#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use tensor::adm::ExtrinsicCurvature;
use tensor::AdmGrid;
use tensor::adm_step::{adm_step_rk4, geodesic_rhs, hamiltonian_l2};
use tensor::AdmState;

// ── Flat-space stability ───────────────────────────────────────────────────────

// Minimum 5×5×5 grid gives a single interior point at (2,2,2), keeping tests fast.
const N: usize = 5;

/// Flat Minkowski data is an exact solution: γ = δ, K = 0, all derivatives zero.
/// The RHS should be exactly zero everywhere (no truncation error on flat space).
#[test]
fn test_geodesic_rhs_flat_is_zero() {
    let grid = AdmGrid::flat(N, N, N, 0.1, 0.1, 0.1);
    let rhs = geodesic_rhs(&grid);

    for (i, &v) in rhs.iter().enumerate() {
        assert!(
            v.abs() < 1e-12,
            "rhs[{i}] = {v:.3e} should be zero for flat space"
        );
    }
}

/// After one RK4 step on flat data, fields should remain unchanged.
#[test]
fn test_rk4_flat_no_drift() {
    let grid = AdmGrid::flat(N, N, N, 0.1, 0.1, 0.1);
    let stepped = adm_step_rk4(&grid, 0.05);

    for (a, b) in grid.fields.iter().zip(stepped.fields.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "flat space should not drift after RK4 step: Δ = {:.3e}",
            (a - b).abs()
        );
    }
}

/// Hamiltonian constraint on flat data should be zero (within numerical precision).
#[test]
fn test_hamiltonian_l2_flat() {
    let grid = AdmGrid::flat(N, N, N, 0.1, 0.1, 0.1);
    let h = hamiltonian_l2(&grid);
    assert!(h < 1e-10, "Hamiltonian constraint L2 = {h:.3e} on flat data");
}

// ── Non-trivial K — pure-trace isotropic ──────────────────────────────────────

/// Isotropic initial data: γ = δ_{ij}, K_{ij} = ε δ_{ij}.
/// This is not a solution of the constraint equations (H = 6ε² ≠ 0 for ε ≠ 0),
/// but we can verify that the RK4 stepper produces a non-trivial evolution and
/// that γ changes at the expected rate ∂_t γ = -2K = -2ε δ.
///
/// After one small step dt: γ_{ii} ≈ 1 - 2ε dt  (off-diagonals stay ≈ 0).
#[test]
fn test_rk4_isotropic_k_gamma_evolution() {
    let eps = 0.01_f64;
    let dt = 0.001_f64;

    let grid = AdmGrid::new(N, N, N, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, |_, _, _| {
        let mut s = AdmState::flat();
        s.k = ExtrinsicCurvature::from_f64(3, vec![
            eps, 0.0, 0.0,
            0.0, eps, 0.0,
            0.0, 0.0, eps,
        ]);
        s
    });

    let stepped = adm_step_rk4(&grid, dt);

    // Single interior point in a 5×5×5 grid is (2, 2, 2)
    let g_old = grid.gamma_flat(2, 2, 2);
    let g_new = stepped.gamma_flat(2, 2, 2);

    for i in 0..3 {
        let diag_old = g_old[i * 3 + i];
        let diag_new = g_new[i * 3 + i];
        let expected = diag_old - 2.0 * eps * dt;
        assert!(
            (diag_new - expected).abs() < 1e-8,
            "γ_{{{i}{i}}} after dt: got {diag_new:.8}, expected {expected:.8}"
        );

        for j in 0..3 {
            if i != j {
                let off = g_new[i * 3 + j];
                assert!(off.abs() < 1e-12, "γ_{{{i}{j}}} should stay zero, got {off:.3e}");
            }
        }
    }
}

/// Verify that boundary cells (ix < 2 or ix ≥ nx-2) are never modified by RK4.
#[test]
fn test_rk4_boundary_fixed() {
    let eps = 0.05_f64;

    let grid = AdmGrid::new(N, N, N, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, |_, _, _| {
        let mut s = AdmState::flat();
        s.k = ExtrinsicCurvature::from_f64(3, vec![
            eps, 0.0, 0.0,
            0.0, eps, 0.0,
            0.0, 0.0, eps,
        ]);
        s
    });

    let stepped = adm_step_rk4(&grid, 0.05);

    // Boundary points: ix ∈ {0, 1, N-2, N-1}
    for ix in [0, 1, N - 2, N - 1] {
        for iy in 0..N {
            for iz in 0..N {
                let g_old = grid.gamma_flat(ix, iy, iz);
                let g_new = stepped.gamma_flat(ix, iy, iz);
                let k_old = grid.k_flat(ix, iy, iz);
                let k_new = stepped.k_flat(ix, iy, iz);
                for f in 0..9 {
                    assert_eq!(g_old[f], g_new[f],
                        "boundary ({ix},{iy},{iz}) gamma[{f}] was modified");
                    assert_eq!(k_old[f], k_new[f],
                        "boundary ({ix},{iy},{iz}) K[{f}] was modified");
                }
            }
        }
    }
}

// ── Multi-step stability on flat space ────────────────────────────────────────

/// Run 100 RK4 steps on flat space. Fields should remain machine-precision flat.
/// Slow in debug mode (~2 min); run with: cargo test -- --ignored
#[test]
#[ignore]
fn test_rk4_flat_100_steps() {
    let mut grid = AdmGrid::flat(7, 7, 7, 0.1, 0.1, 0.1);
    for _ in 0..100 {
        grid = adm_step_rk4(&grid, 0.01);
    }

    // All gamma should still be δ_{ij}, all K should be 0 (single interior pt at (2,2,2))
    for ix in [2] {
        for iy in [2] {
            for iz in [2] {
                let g = grid.gamma_flat(ix, iy, iz);
                let k = grid.k_flat(ix, iy, iz);
                for i in 0..3 {
                    for j in 0..3 {
                        let expected_g = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (g[i * 3 + j] - expected_g).abs() < 1e-10,
                            "γ_{{{i}{j}}} at ({ix},{iy},{iz}) after 100 steps: {:.6e}",
                            g[i * 3 + j] - expected_g
                        );
                        assert!(
                            k[i * 3 + j].abs() < 1e-10,
                            "K_{{{i}{j}}} at ({ix},{iy},{iz}) after 100 steps: {:.6e}",
                            k[i * 3 + j]
                        );
                    }
                }
            }
        }
    }
}
