#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::einstein_residual;
use tensor::Tensor;

/// Flat Minkowski vacuum: G = 0, T = 0 → residual = 0.
#[test]
#[serial]
fn test_einstein_residual_flat_vacuum() {
    let g_fn = |_: &[f64]| {
        Tensor::<0, 2>::from_f64(4, vec![
            -1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
        ])
    };
    let g_inv_fn = |_: &[f64]| {
        Tensor::<2, 0>::from_f64(4, vec![
            -1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
        ])
    };
    let t = Tensor::<0, 2>::from_f64(4, vec![0.0; 16]);
    let point = vec![0.0, 0.0, 0.0, 0.0];

    let residual = einstein_residual(&g_fn, &g_inv_fn, &t, &point, 1e-5, 1.0);

    for c in &residual.components {
        assert!(
            c.result.abs() < 1e-7,
            "Flat vacuum residual should be 0, got {}",
            c.result
        );
    }
}

/// Flat Minkowski with static electric field E = 2 in x-direction, μ₀ = 1.
///
/// G_{μν} = 0 (flat geometry), so residual = −κ T_{μν}.
///
/// T_{00} =  2,  T_{11} = −2,  T_{22} = 2,  T_{33} = 2  (off-diag = 0).
/// With κ = 1: residual_{00} = −2, residual_{11} = 2, residual_{22} = −2, etc.
#[test]
#[serial]
fn test_einstein_residual_flat_em_source() {
    let g_fn = |_: &[f64]| {
        Tensor::<0, 2>::from_f64(4, vec![
            -1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
        ])
    };
    let g_inv_fn = |_: &[f64]| {
        Tensor::<2, 0>::from_f64(4, vec![
            -1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
        ])
    };

    // T_{μν} from EM stress-energy test: E=2 in x-direction, μ₀=1.
    let mut t_vals = vec![0.0f64; 16];
    t_vals[0]  =  2.0;   // T_{00} = E²/(2μ₀)
    t_vals[5]  = -2.0;   // T_{11}
    t_vals[10] =  2.0;   // T_{22}
    t_vals[15] =  2.0;   // T_{33}
    let t = Tensor::<0, 2>::from_f64(4, t_vals);

    let point = vec![0.0, 0.0, 0.0, 0.0];
    let kappa = 1.0;

    let residual = einstein_residual(&g_fn, &g_inv_fn, &t, &point, 1e-5, kappa);

    // G = 0 (flat), so residual = -κ T.
    assert!((residual.components[0].result  - (-2.0)).abs() < 1e-7); // −T_{00}
    assert!((residual.components[5].result  -   2.0 ).abs() < 1e-7); // −T_{11}
    assert!((residual.components[10].result - (-2.0)).abs() < 1e-7); // −T_{22}
    assert!((residual.components[15].result - (-2.0)).abs() < 1e-7); // −T_{33}

    // Off-diagonal components: G=0, T=0 → residual=0.
    for (i, c) in residual.components.iter().enumerate() {
        if ![0, 5, 10, 15].contains(&i) {
            assert!(
                c.result.abs() < 1e-7,
                "residual[{}] should be 0, got {}",
                i, c.result
            );
        }
    }
}

/// Unit 2-sphere vacuum: G_{μν} = 0 identically in 2D, T = 0 → residual = 0.
///
/// Metric: g = diag(1, sin²θ) at (θ, φ) = (π/4, 0).
/// G_{μν} = 0 is the fundamental theorem for 2D (proven analytically in
/// einstein_tensor_tests). This test verifies that the fully numerical
/// pipeline also gives ≈ 0.
#[test]
#[serial]
fn test_einstein_residual_2d_sphere_vacuum() {
    let theta = std::f64::consts::FRAC_PI_4; // π/4, away from pole singularity
    let point = vec![theta, 0.0];

    let g_fn = |x: &[f64]| {
        let s = x[0].sin();
        Tensor::<0, 2>::from_f64(2, vec![
            1.0, 0.0,
            0.0, s * s,
        ])
    };
    let g_inv_fn = |x: &[f64]| {
        let s = x[0].sin();
        Tensor::<2, 0>::from_f64(2, vec![
            1.0, 0.0,
            0.0, 1.0 / (s * s),
        ])
    };
    let t = Tensor::<0, 2>::from_f64(2, vec![0.0; 4]);

    let residual = einstein_residual(&g_fn, &g_inv_fn, &t, &point, 1e-5, 1.0);

    for c in &residual.components {
        assert!(
            c.result.abs() < 1e-6,
            "2-sphere vacuum residual should be 0, got {}",
            c.result
        );
    }
}
