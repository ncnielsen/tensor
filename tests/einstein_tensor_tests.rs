#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::einstein_tensor;
use tensor::ricci_scalar;
use tensor::ricci_tensor;
use tensor::riemann;
use tensor::Christoffel;
use tensor::ChristoffelDerivative;
use tensor::Tensor;

/// Flat 2D space: G_{μν} = 0.
#[test]
#[serial]
fn test_einstein_tensor_flat_space() {
    let g: Tensor<0, 2> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let g_inv: Tensor<2, 0> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let gamma = Christoffel::from_f64(2, vec![0.0; 8]);
    let partial_gamma = ChristoffelDerivative::from_f64(2, vec![0.0; 16]);

    let r_tensor = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r_tensor);
    let scalar = ricci_scalar(&g_inv, &ric);
    let g_einstein = einstein_tensor(&ric, &g, &scalar);

    for c in &g_einstein.components {
        assert_eq!(c.result, 0.0);
    }
}

/// Unit 2-sphere at θ = π/2.
///
/// R_{μν} = diag(1, 1),  R = 2,  g_{μν} = diag(1, 1)
///
/// G_{μν} = R_{μν} − ½ g_{μν} R
///         = diag(1,1) − ½ · diag(1,1) · 2
///         = diag(1,1) − diag(1,1)
///         = 0
///
/// This is a fundamental result: in 2D the Einstein tensor always vanishes
/// identically, a consequence of the 2D Bianchi identity.
#[test]
#[serial]
fn test_einstein_tensor_unit_2_sphere() {
    // Metric and inverse metric at θ = π/2, unit sphere
    let g: Tensor<0, 2> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let g_inv: Tensor<2, 0> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);

    let gamma = Christoffel::from_f64(2, vec![0.0; 8]);

    let mut pg = vec![0.0f64; 16];
    pg[6]  =  1.0;   // ∂_θ Γ^θ_{φφ} =  1
    pg[10] = -1.0;   // ∂_θ Γ^φ_{θφ} = -1
    pg[12] = -1.0;   // ∂_θ Γ^φ_{φθ} = -1
    let partial_gamma = ChristoffelDerivative::from_f64(2, pg);

    let r_tensor = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r_tensor);
    let scalar = ricci_scalar(&g_inv, &ric);
    let g_einstein = einstein_tensor(&ric, &g, &scalar);

    for c in &g_einstein.components {
        assert!(
            c.result.abs() < 1e-12,
            "Expected G_μν = 0 in 2D, got {}",
            c.result
        );
    }
}
