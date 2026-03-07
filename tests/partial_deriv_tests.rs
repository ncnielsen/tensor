#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::christoffel_partial_deriv;
use tensor::partial_deriv;
use tensor::Christoffel;
use tensor::Tensor;

/// Constant covector field → all partial derivatives are zero.
#[test]
#[serial]
fn test_partial_deriv_constant_field() {
    let f = |_: &[f64]| Tensor::<0, 1>::from_f64(2, vec![3.0, 7.0]);
    let dp = partial_deriv(&f, &[1.0, 2.0], 1e-5);
    // Output: Tensor<0,2>, 4 components, all zero.
    for c in &dp.components {
        assert!(
            c.result.abs() < 1e-9,
            "Expected 0, got {}",
            c.result
        );
    }
}

/// Linear covector field f_μ(x) = a_{μν} x^ν → ∂_κ f_μ = a_{μκ} (exact).
///
///   f_0(x) = 2·x0 + 3·x1,  f_1(x) = 5·x0 − x1
///   ∂_0 f_0 = 2,  ∂_1 f_0 = 3
///   ∂_0 f_1 = 5,  ∂_1 f_1 = −1
///
/// Output layout [μ, κ]: component at flat index μ·2 + κ.
#[test]
#[serial]
fn test_partial_deriv_linear_field() {
    let f = |x: &[f64]| {
        Tensor::<0, 1>::from_f64(2, vec![
            2.0 * x[0] + 3.0 * x[1],
            5.0 * x[0] - x[1],
        ])
    };
    let dp = partial_deriv(&f, &[1.0, 2.0], 1e-5);

    assert!((dp.components[0].result - 2.0).abs() < 1e-8);  // ∂_0 f_0
    assert!((dp.components[1].result - 3.0).abs() < 1e-8);  // ∂_1 f_0
    assert!((dp.components[2].result - 5.0).abs() < 1e-8);  // ∂_0 f_1
    assert!((dp.components[3].result - (-1.0)).abs() < 1e-8); // ∂_1 f_1
}

/// Flat Minkowski metric: all partial derivatives vanish, so Christoffel = 0.
///
/// This is an integration test: partial_deriv → Christoffel::from_metric.
#[test]
#[serial]
fn test_partial_deriv_flat_metric_zero_christoffel() {
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

    let point = vec![0.0, 0.0, 0.0, 0.0];
    let partial_g = partial_deriv(&g_fn, &point, 1e-5);
    let g = g_fn(&point);
    let g_inv = g_inv_fn(&point);
    let gamma = Christoffel::from_metric(&g, &g_inv, &partial_g);

    for c in &gamma.components {
        assert!(
            c.result.abs() < 1e-8,
            "Expected Γ = 0 for flat metric, got {}",
            c.result
        );
    }
}

/// Polar coordinate metric g = diag(1, r²) in 2D.
///
/// g_{rr} = 1, g_{φφ} = r², g_{rφ} = 0.
/// Only non-zero metric derivative: ∂_r g_{φφ} = 2r.
///
/// Christoffel symbols at (r=2, φ=0.5):
///   Γ^r_{φφ} = −r = −2
///   Γ^φ_{rφ} = Γ^φ_{φr} = 1/r = 0.5
///   all others = 0
#[test]
#[serial]
fn test_partial_deriv_polar_metric_christoffel() {
    let r = 2.0_f64;
    let phi = 0.5_f64;
    let point = vec![r, phi];

    let g_fn = |x: &[f64]| {
        Tensor::<0, 2>::from_f64(2, vec![
            1.0, 0.0,
            0.0, x[0] * x[0],
        ])
    };
    let g_inv_fn = |x: &[f64]| {
        Tensor::<2, 0>::from_f64(2, vec![
            1.0, 0.0,
            0.0, 1.0 / (x[0] * x[0]),
        ])
    };

    let partial_g = partial_deriv(&g_fn, &point, 1e-5);
    let g = g_fn(&point);
    let g_inv = g_inv_fn(&point);
    let gamma = Christoffel::from_metric(&g, &g_inv, &partial_g);

    assert!((gamma.component(0, 1, 1).result - (-r)).abs() < 1e-6,
        "Γ^r_{{φφ}} = {} (expected {})", gamma.component(0, 1, 1).result, -r);
    assert!((gamma.component(1, 0, 1).result - 1.0 / r).abs() < 1e-6,
        "Γ^φ_{{rφ}} = {} (expected {})", gamma.component(1, 0, 1).result, 1.0 / r);
    assert!(gamma.component(0, 0, 0).result.abs() < 1e-6);
    assert!(gamma.component(0, 0, 1).result.abs() < 1e-6);
    assert!(gamma.component(1, 1, 1).result.abs() < 1e-6);
    assert!(gamma.component(1, 0, 0).result.abs() < 1e-6);
}

/// Polar coordinate Christoffel symbols → christoffel_partial_deriv.
///
/// Γ^r_{φφ} = −r  ⟹  ∂_r Γ^r_{φφ} = −1,  ∂_φ Γ^r_{φφ} = 0
/// Γ^φ_{rφ} = 1/r ⟹  ∂_r Γ^φ_{rφ} = −1/r²,  ∂_φ Γ^φ_{rφ} = 0
/// All other Γ = 0 ⟹  all their derivatives = 0
#[test]
#[serial]
fn test_christoffel_partial_deriv_polar() {
    let r = 2.0_f64;
    let phi = 0.5_f64;
    let point = vec![r, phi];

    // Christoffel symbols for 2D polar as a function of position.
    // Layout [k, i, j]:
    //   flat 0:(0,0,0)=Γ^r_{rr}=0   flat 1:(0,0,1)=Γ^r_{rφ}=0
    //   flat 2:(0,1,0)=Γ^r_{φr}=0   flat 3:(0,1,1)=Γ^r_{φφ}=−r
    //   flat 4:(1,0,0)=Γ^φ_{rr}=0   flat 5:(1,0,1)=Γ^φ_{rφ}=1/r
    //   flat 6:(1,1,0)=Γ^φ_{φr}=1/r flat 7:(1,1,1)=Γ^φ_{φφ}=0
    let gamma_fn = |x: &[f64]| {
        let rv = x[0];
        Christoffel::from_f64(2, vec![
            0.0, 0.0, 0.0, -rv,
            0.0, 1.0 / rv, 1.0 / rv, 0.0,
        ])
    };

    let dgamma = christoffel_partial_deriv(&gamma_fn, &point, 1e-5);

    // ∂_r Γ^r_{φφ}: (ρ=0, κ=1, μ=1, ν=0)
    assert!(
        (dgamma.component(0, 1, 1, 0).result - (-1.0)).abs() < 1e-6,
        "∂_r Γ^r_{{φφ}} = {} (expected -1)", dgamma.component(0, 1, 1, 0).result
    );

    // ∂_r Γ^φ_{rφ}: (ρ=1, κ=0, μ=1, ν=0) = -1/r²
    let expected = -1.0 / (r * r);
    assert!(
        (dgamma.component(1, 0, 1, 0).result - expected).abs() < 1e-6,
        "∂_r Γ^φ_{{rφ}} = {} (expected {})", dgamma.component(1, 0, 1, 0).result, expected
    );

    // φ-derivatives should all be zero (no φ dependence).
    for rho in 0..2 {
        for kappa in 0..2 {
            for mu in 0..2 {
                let v = dgamma.component(rho, kappa, mu, 1).result;
                assert!(v.abs() < 1e-6,
                    "∂_φ Γ^{}_{{{}{}}} = {} (expected 0)", rho, kappa, mu, v);
            }
        }
    }
}
