#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::covariant_derivative;
use tensor::Christoffel;
use tensor::Tensor;

// ── helpers ──────────────────────────────────────────────────────────────────

/// 2-D zero Christoffel symbols (flat / Cartesian connection).
fn zero_christoffel_2d() -> Christoffel {
    Christoffel::from_f64(2, vec![0.0; 8])
}

/// 2-D Christoffel with only Γ^0_{00} = 1, all other components zero.
///
/// Layout: component(i, j, k) = Γ^i_{jk}, row-major over [i, j, k] in 0..2.
/// Flat index 0 = Γ^0_{00} = 1, all others = 0.
fn christoffel_upper0_00() -> Christoffel {
    Christoffel::from_f64(2, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// In flat space (Γ = 0) the covariant derivative equals the partial derivative.
/// V = [3, 5] (Tensor<1,0>, dim 2).
/// ∂V layout (Tensor<1,1>, index [i, k]): [∂_0 V^0, ∂_1 V^0, ∂_0 V^1, ∂_1 V^1]
///         = [2, 7, 4, 9].
/// Expected ∇V = ∂V = [2, 7, 4, 9].
#[test]
#[serial]
fn test_vector_covariant_derivative_flat_space() {
    let v: Tensor<1, 0> = Tensor::from_f64(2, vec![3.0, 5.0]);
    let partial: Tensor<1, 1> = Tensor::from_f64(2, vec![2.0, 7.0, 4.0, 9.0]);
    let gamma = zero_christoffel_2d();

    let nabla_v = covariant_derivative(&v, &partial, &gamma);

    assert_eq!(nabla_v.components[0].result, 2.0); // ∇_0 V^0
    assert_eq!(nabla_v.components[1].result, 7.0); // ∇_1 V^0
    assert_eq!(nabla_v.components[2].result, 4.0); // ∇_0 V^1
    assert_eq!(nabla_v.components[3].result, 9.0); // ∇_1 V^1
}

/// Non-trivial connection: only Γ^0_{00} = 1.
/// V = [3, 5],  ∂V = [2, 7, 4, 9].
///
/// ∇_k V^i = ∂_k V^i + Γ^i_{kl} V^l
///
/// ∇_0 V^0 = 2 + Γ^0_{0,0}·V^0 + Γ^0_{0,1}·V^1 = 2 + 1·3 + 0·5 = 5
/// ∇_1 V^0 = 7 + Γ^0_{1,0}·V^0 + Γ^0_{1,1}·V^1 = 7 + 0   + 0   = 7
/// ∇_0 V^1 = 4 + Γ^1_{0,0}·V^0 + Γ^1_{0,1}·V^1 = 4 + 0   + 0   = 4
/// ∇_1 V^1 = 9 + Γ^1_{1,0}·V^0 + Γ^1_{1,1}·V^1 = 9 + 0   + 0   = 9
#[test]
#[serial]
fn test_vector_covariant_derivative_with_connection() {
    let v: Tensor<1, 0> = Tensor::from_f64(2, vec![3.0, 5.0]);
    let partial: Tensor<1, 1> = Tensor::from_f64(2, vec![2.0, 7.0, 4.0, 9.0]);
    let gamma = christoffel_upper0_00();

    let nabla_v = covariant_derivative(&v, &partial, &gamma);

    assert_eq!(nabla_v.components[0].result, 5.0); // ∇_0 V^0 = 2 + 3
    assert_eq!(nabla_v.components[1].result, 7.0); // ∇_1 V^0 unchanged
    assert_eq!(nabla_v.components[2].result, 4.0); // ∇_0 V^1 unchanged
    assert_eq!(nabla_v.components[3].result, 9.0); // ∇_1 V^1 unchanged
}

/// Non-trivial connection: only Γ^0_{00} = 1.
/// ω = [3, 5] (Tensor<0,1>, dim 2).
/// ∂ω layout (Tensor<0,2>, index [j, k]): [∂_0 ω_0, ∂_1 ω_0, ∂_0 ω_1, ∂_1 ω_1]
///           = [2, 7, 4, 9].
///
/// ∇_k ω_j = ∂_k ω_j − Γ^l_{kj} ω_l
///
/// ∇_0 ω_0 = 2 − (Γ^0_{0,0}·ω_0 + Γ^1_{0,0}·ω_1) = 2 − (1·3 + 0·5) = −1
/// ∇_1 ω_0 = 7 − (Γ^0_{1,0}·ω_0 + Γ^1_{1,0}·ω_1) = 7 − (0   + 0  ) =  7
/// ∇_0 ω_1 = 4 − (Γ^0_{0,1}·ω_0 + Γ^1_{0,1}·ω_1) = 4 − (0   + 0  ) =  4
/// ∇_1 ω_1 = 9 − (Γ^0_{1,1}·ω_0 + Γ^1_{1,1}·ω_1) = 9 − (0   + 0  ) =  9
#[test]
#[serial]
fn test_covector_covariant_derivative_with_connection() {
    let omega: Tensor<0, 1> = Tensor::from_f64(2, vec![3.0, 5.0]);
    let partial: Tensor<0, 2> = Tensor::from_f64(2, vec![2.0, 7.0, 4.0, 9.0]);
    let gamma = christoffel_upper0_00();

    let nabla_omega = covariant_derivative(&omega, &partial, &gamma);

    assert_eq!(nabla_omega.components[0].result, -1.0); // ∇_0 ω_0 = 2 − 3
    assert_eq!(nabla_omega.components[1].result, 7.0);  // ∇_1 ω_0 unchanged
    assert_eq!(nabla_omega.components[2].result, 4.0);  // ∇_0 ω_1 unchanged
    assert_eq!(nabla_omega.components[3].result, 9.0);  // ∇_1 ω_1 unchanged
}
