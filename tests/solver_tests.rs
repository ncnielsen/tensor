#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::{invert_matrix, solve_1d, Tensor};

// ─── invert_matrix ────────────────────────────────────────────────────────────

/// Identity matrix inverts to itself.
#[test]
#[serial]
fn test_invert_identity_2x2() {
    let id = vec![1.0, 0.0, 0.0, 1.0];
    let inv = invert_matrix(&id, 2).expect("identity is invertible");
    for (a, b) in id.iter().zip(inv.iter()) {
        assert!((a - b).abs() < 1e-12, "inv[i] = {} (expected {})", b, a);
    }
}

/// Known 2×2 inverse: [[2,1],[5,3]]⁻¹ = [[3,-1],[-5,2]].
#[test]
#[serial]
fn test_invert_2x2_known() {
    let a = vec![2.0, 1.0, 5.0, 3.0];
    let expected = vec![3.0, -1.0, -5.0, 2.0];
    let inv = invert_matrix(&a, 2).expect("invertible");
    for (e, got) in expected.iter().zip(inv.iter()) {
        assert!((e - got).abs() < 1e-10, "inv = {} (expected {})", got, e);
    }
}

/// Diagonal 3×3: [[a,0,0],[0,b,0],[0,0,c]]⁻¹ = [[1/a,0,0],[0,1/b,0],[0,0,1/c]].
#[test]
#[serial]
fn test_invert_diagonal_3x3() {
    let (a, b, c) = (2.0_f64, 4.0_f64, 0.5_f64);
    let mat = vec![
        a, 0.0, 0.0,
        0.0, b, 0.0,
        0.0, 0.0, c,
    ];
    let inv = invert_matrix(&mat, 3).expect("invertible");
    let expected = vec![
        1.0 / a, 0.0, 0.0,
        0.0, 1.0 / b, 0.0,
        0.0, 0.0, 1.0 / c,
    ];
    for (e, got) in expected.iter().zip(inv.iter()) {
        assert!((e - got).abs() < 1e-12, "inv = {} (expected {})", got, e);
    }
}

/// Singular matrix returns None.
#[test]
#[serial]
fn test_invert_singular_returns_none() {
    let singular = vec![1.0, 2.0, 2.0, 4.0]; // rows are linearly dependent
    assert!(invert_matrix(&singular, 2).is_none());
}

// ─── solve_1d ─────────────────────────────────────────────────────────────────

/// Build a flat 2D identity metric tensor (dim=2, g = diag(1,1)).
fn flat_2d() -> Vec<f64> {
    vec![1.0, 0.0, 0.0, 1.0]
}

/// Build a zero Tensor<0,2> of dimension `dim`.
fn zero_t(dim: usize) -> Tensor<0, 2> {
    Tensor::from_f64(dim, vec![0.0; dim * dim])
}

/// Flat vacuum (G = 0, T = 0) on a 3-point grid.
///
/// The residual is identically zero for flat spacetime, so `solve_1d` should
/// declare convergence after 0 Newton iterations.
#[test]
#[serial]
fn test_solve_1d_flat_vacuum_converges_immediately() {
    let n = 3;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| flat_2d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t(2)).collect();

    let result = solve_1d(&g_grid, &t_grid, 1.0, 1.0, 1e-8, 10, 1e-5);

    assert!(
        result.converged,
        "Flat vacuum should converge immediately (residual = {})",
        result.residual_norm
    );
    assert_eq!(result.iterations, 0, "No NR steps needed for flat vacuum");
    assert!(
        result.residual_norm < 1e-8,
        "Residual = {}",
        result.residual_norm
    );
}

/// Flat vacuum on a 5-point grid — boundary conditions are flat, interior is flat.
#[test]
#[serial]
fn test_solve_1d_flat_vacuum_5_points() {
    let n = 5;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| flat_2d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t(2)).collect();

    let result = solve_1d(&g_grid, &t_grid, 0.5, 1.0, 1e-8, 20, 1e-5);

    assert!(
        result.converged,
        "Flat vacuum 5-pt should converge; residual = {}",
        result.residual_norm
    );
    assert!(result.residual_norm < 1e-8);

    // Verify all interior metrics are still close to identity.
    let dim2 = 4usize;
    let id = flat_2d();
    for i in 1..n - 1 {
        for (k, (&got, &exp)) in result.g_grid[i].iter().zip(id.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "g_grid[{}][{}] = {} (expected {})",
                i, k, got, exp
            );
        }
    }
    let _ = dim2; // suppress unused warning
}

/// solve_1d preserves boundary conditions exactly.
#[test]
#[serial]
fn test_solve_1d_boundaries_unchanged() {
    let n = 4;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| flat_2d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t(2)).collect();

    let result = solve_1d(&g_grid, &t_grid, 1.0, 1.0, 1e-6, 5, 1e-5);

    assert_eq!(result.g_grid[0], flat_2d(), "Left boundary must not change");
    assert_eq!(
        result.g_grid[n - 1],
        flat_2d(),
        "Right boundary must not change"
    );
}
