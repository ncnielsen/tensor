#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::ricci_tensor;
use tensor::riemann;
use tensor::Christoffel;
use tensor::ChristoffelDerivative;

/// Unit 2-sphere at θ = π/2.
///
/// Coordinates: x^0 = θ, x^1 = φ.
///
/// At θ = π/2 all Christoffel symbols vanish (Γ^θ_{φφ} = -sinθ cosθ = 0,
/// Γ^φ_{θφ} = cot θ = 0), so the quadratic Γ·Γ terms in the Riemann tensor
/// drop out.  The non-zero partial derivatives of Γ at this point are:
///
///   ∂_θ Γ^θ_{φφ} = -(cos²θ - sin²θ)|_{π/2} = 1   →  ∂_0 Γ^0_{11} = 1
///   ∂_θ Γ^φ_{θφ} = -1/sin²θ|_{π/2}           = -1  →  ∂_0 Γ^1_{01} = -1
///   ∂_θ Γ^φ_{φθ} = -1/sin²θ|_{π/2}           = -1  →  ∂_0 Γ^1_{10} = -1
///
/// Riemann tensor (only ∂Γ terms survive):
///   R^0_{101} =  ∂_0 Γ^0_{11} - ∂_1 Γ^0_{01} = 1 - 0 =  1
///   R^1_{001} =  ∂_0 Γ^1_{10} - ∂_1 Γ^1_{00} = -1 - 0 = -1
///   (and their antisymmetric partners; all others zero)
///
/// Ricci tensor R_{σν} = Σ_ρ R^ρ_{σρν}:
///   R_{00} = R^0_{000} + R^1_{010} = 0 + 1 = 1
///   R_{01} = R^0_{001} + R^1_{011} = 0 + 0 = 0
///   R_{10} = R^0_{100} + R^1_{110} = 0 + 0 = 0
///   R_{11} = R^0_{101} + R^1_{111} = 1 + 0 = 1
///
/// For a unit 2-sphere, R_{μν} = (n−1) g_{μν} = g_{μν} = diag(1, 1)  ✓
#[test]
#[serial]
fn test_ricci_tensor_unit_2_sphere() {
    // All Christoffel symbols are zero at θ = π/2
    let gamma = Christoffel::from_f64(2, vec![0.0; 8]);

    // Christoffel derivatives: layout [ρ, κ, μ, ν], flat = ρ*8 + κ*4 + μ*2 + ν
    // Non-zero entries:
    //   [0,1,1,0] = flat 6  → ∂_θ Γ^θ_{φφ} =  1
    //   [1,0,1,0] = flat 10 → ∂_θ Γ^φ_{θφ} = -1
    //   [1,1,0,0] = flat 12 → ∂_θ Γ^φ_{φθ} = -1
    let mut pg = vec![0.0f64; 16];
    pg[6]  =  1.0;
    pg[10] = -1.0;
    pg[12] = -1.0;
    let partial_gamma = ChristoffelDerivative::from_f64(2, pg);

    let r = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r);

    // R_{μν} = diag(1, 1)
    assert_eq!(ric.components[0].result, 1.0); // R_{00}
    assert_eq!(ric.components[1].result, 0.0); // R_{01}
    assert_eq!(ric.components[2].result, 0.0); // R_{10}
    assert_eq!(ric.components[3].result, 1.0); // R_{11}
}

/// Flat Cartesian 2D: Riemann = 0 so Ricci = 0.
#[test]
#[serial]
fn test_ricci_tensor_flat_space() {
    let gamma = Christoffel::from_f64(2, vec![0.0; 8]);
    let partial_gamma = ChristoffelDerivative::from_f64(2, vec![0.0; 16]);

    let r = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r);

    for c in &ric.components {
        assert_eq!(c.result, 0.0);
    }
}
