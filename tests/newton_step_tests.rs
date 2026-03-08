#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::newton_step;

/// Linear scalar equation F(x) = x − 3.
///
/// NR converges in exactly one step for affine F.
#[test]
#[serial]
fn test_newton_step_linear_scalar() {
    let f = |x: &[f64]| vec![x[0] - 3.0];
    let x_new = newton_step(&f, &[1.0], 1e-5);
    assert!((x_new[0] - 3.0).abs() < 1e-8, "x_new = {} (expected 3)", x_new[0]);
}

/// Quadratic scalar F(x) = x² − 4, solution x = 2.
///
/// NR from x = 1 converges, but slowly at first (we start far from the root).
/// After enough steps the root is tight.
#[test]
#[serial]
fn test_newton_step_quadratic_scalar() {
    let f = |x: &[f64]| vec![x[0] * x[0] - 4.0];

    let mut x = vec![1.0_f64];
    for _ in 0..10 {
        x = newton_step(&f, &x, 1e-5);
    }
    assert!((x[0] - 2.0).abs() < 1e-7, "converged to {} (expected 2.0)", x[0]);
}

/// Linear 2×2 system — exact solution in one step.
///
///   F(x, y) = [ 2x +  y − 5 ]    solution: x = 2, y = 1
///             [  x −  y − 1 ]
///
/// J = [[2, 1], [1, −1]],  det = −3.
/// From (0, 0): δ = J⁻¹ [5, 1] = [2, 1].
#[test]
#[serial]
fn test_newton_step_linear_2x2() {
    let f = |x: &[f64]| vec![
        2.0 * x[0] + x[1] - 5.0,
        x[0] - x[1] - 1.0,
    ];
    let x_new = newton_step(&f, &[0.0, 0.0], 1e-5);
    assert!((x_new[0] - 2.0).abs() < 1e-8, "x = {} (expected 2)", x_new[0]);
    assert!((x_new[1] - 1.0).abs() < 1e-8, "y = {} (expected 1)", x_new[1]);
}

/// Nonlinear 2×2 system — Newton steps converge to the root.
///
///   F(x, y) = [ x² + y² − 5 ]   root: (2, 1) and (−2, 1), etc.
///             [ x  − y  − 1 ]
///
/// Start close to the root so convergence is fast.
#[test]
#[serial]
fn test_newton_step_nonlinear_2x2() {
    let f = |x: &[f64]| vec![
        x[0] * x[0] + x[1] * x[1] - 5.0,
        x[0] - x[1] - 1.0,
    ];

    // Starting from (2.1, 1.1), close to the root (2, 1).
    let x1 = newton_step(&f, &[2.1, 1.1], 1e-5);
    let x2 = newton_step(&f, &x1, 1e-5);
    let x3 = newton_step(&f, &x2, 1e-5);

    // Three steps with quadratic convergence → very tight residual.
    let r = f(&x3);
    let res_norm: f64 = r.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "|F| = {} after 3 NR steps (expected < 1e-8)", res_norm);
}

/// Singular Jacobian — partial update, no panic.
///
/// F(x, y) = [ x + y − 1 ]  (one equation, two unknowns: rank-1 Jacobian)
///           [ x + y − 1 ]
///
/// J = [[1,1],[1,1]]: singular.  The solver skips the zero pivot and returns
/// a partial step rather than crashing.
#[test]
#[serial]
fn test_newton_step_singular_jacobian() {
    let f = |x: &[f64]| vec![x[0] + x[1] - 1.0, x[0] + x[1] - 1.0];
    // Should not panic.
    let x_new = newton_step(&f, &[0.0, 0.0], 1e-5);
    // Partial update: one pivot is solved, one is skipped → x+y ≈ 1.
    let sum = x_new[0] + x_new[1];
    assert!((sum - 1.0).abs() < 1e-6, "x+y = {} (expected 1)", sum);
}
