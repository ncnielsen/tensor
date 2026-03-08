#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::{invert_matrix, solve_1d, solve_3d, Tensor};

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

// ─── solve_3d ─────────────────────────────────────────────────────────────────

/// Flat Minkowski metric: diag(−1, 1, 1, 1).
fn minkowski_4d() -> Vec<f64> {
    vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]
}

/// Zero stress-energy tensor of dimension `dim`.
fn zero_t4() -> Tensor<0, 2> {
    Tensor::from_f64(4, vec![0.0; 16])
}

/// Flat Minkowski vacuum on a 3×3×3 grid (dim=4).
///
/// G = 0 everywhere, T = 0 everywhere → residual is identically zero.
/// The solver must declare convergence after 0 NR steps.
#[test]
#[serial]
fn test_solve_3d_flat_vacuum_converges_immediately() {
    let (nx, ny, nz) = (3, 3, 3);
    let n = nx * ny * nz;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| minkowski_4d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t4()).collect();

    let result = solve_3d(&g_grid, &t_grid, nx, ny, nz, 1.0, 1.0, 1e-7, 5, 1e-5);

    assert!(
        result.converged,
        "Flat Minkowski vacuum should converge immediately; residual = {}",
        result.residual_norm
    );
    assert_eq!(result.iterations, 0, "No NR steps needed for flat vacuum");
    assert!(result.residual_norm < 1e-7);
}

/// solve_3d preserves all 6 boundary faces.
#[test]
#[serial]
fn test_solve_3d_boundaries_unchanged() {
    let (nx, ny, nz) = (4, 4, 4);
    let n = nx * ny * nz;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| minkowski_4d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t4()).collect();

    let result = solve_3d(&g_grid, &t_grid, nx, ny, nz, 1.0, 1.0, 1e-7, 5, 1e-5);

    let flat = |ix: usize, iy: usize, iz: usize| ix * ny * nz + iy * nz + iz;
    let expected = minkowski_4d();

    // Spot-check each face.
    for iy in 0..ny {
        for iz in 0..nz {
            assert_eq!(result.g_grid[flat(0, iy, iz)], expected, "x-left face");
            assert_eq!(result.g_grid[flat(nx - 1, iy, iz)], expected, "x-right face");
        }
    }
    for ix in 0..nx {
        for iz in 0..nz {
            assert_eq!(result.g_grid[flat(ix, 0, iz)], expected, "y-left face");
            assert_eq!(result.g_grid[flat(ix, ny - 1, iz)], expected, "y-right face");
        }
    }
    for ix in 0..nx {
        for iy in 0..ny {
            assert_eq!(result.g_grid[flat(ix, iy, 0)], expected, "z-left face");
            assert_eq!(result.g_grid[flat(ix, iy, nz - 1)], expected, "z-right face");
        }
    }
}

/// A 5×5×5 flat vacuum: all 27 interior points should converge immediately.
#[test]
#[serial]
fn test_solve_3d_flat_vacuum_5x5x5() {
    let (nx, ny, nz) = (5, 5, 5);
    let n = nx * ny * nz;
    let g_grid: Vec<Vec<f64>> = (0..n).map(|_| minkowski_4d()).collect();
    let t_grid: Vec<Tensor<0, 2>> = (0..n).map(|_| zero_t4()).collect();

    let result = solve_3d(&g_grid, &t_grid, nx, ny, nz, 1.0, 1.0, 1e-7, 5, 1e-5);

    assert!(
        result.converged,
        "5×5×5 flat vacuum should converge; residual = {}",
        result.residual_norm
    );
    assert!(result.residual_norm < 1e-7);

    // Every interior metric should still be Minkowski.
    let expected = minkowski_4d();
    let flat = |ix: usize, iy: usize, iz: usize| ix * ny * nz + iy * nz + iz;
    for ix in 1..nx - 1 {
        for iy in 1..ny - 1 {
            for iz in 1..nz - 1 {
                for (k, (&got, &exp)) in result.g_grid[flat(ix, iy, iz)]
                    .iter()
                    .zip(expected.iter())
                    .enumerate()
                {
                    assert!(
                        (got - exp).abs() < 1e-6,
                        "g[{},{},{}][{}] = {} (expected {})",
                        ix, iy, iz, k, got, exp
                    );
                }
            }
        }
    }
}
