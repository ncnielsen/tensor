#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::ricci_scalar;
use tensor::ricci_tensor;
use tensor::riemann;
use tensor::Christoffel;
use tensor::ChristoffelDerivative;
use tensor::Tensor;

/// Flat 2D space: Riemann = 0, Ricci = 0, scalar = 0.
#[test]
#[serial]
fn test_ricci_scalar_flat_space() {
    let g_inv: Tensor<2, 0> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let gamma = Christoffel::from_f64(2, vec![0.0; 8]);
    let partial_gamma = ChristoffelDerivative::from_f64(2, vec![0.0; 16]);

    let r_tensor = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r_tensor);
    let scalar = ricci_scalar(&g_inv, &ric);

    assert_eq!(scalar.components[0].result, 0.0);
}

/// Unit 2-sphere at θ = π/2.
///
/// Inverse metric at θ = π/2:  g^{θθ} = 1,  g^{φφ} = 1/sin²θ = 1,  off-diag = 0.
/// Ricci tensor:                R_{μν} = diag(1, 1)  (from ricci_tensor tests).
///
/// R = g^{μν} R_{μν} = 1·1 + 0 + 0 + 1·1 = 2
///
/// This matches the known result R = n(n−1)/r² = 2·1/1² = 2 for a unit 2-sphere.
#[test]
#[serial]
fn test_ricci_scalar_unit_2_sphere() {
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

    assert_eq!(scalar.components[0].result, 2.0);
}
