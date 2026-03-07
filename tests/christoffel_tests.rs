#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::Christoffel;
use tensor::Tensor;

/// 2D polar coordinates at r = 2.
///
/// Metric:           g = diag(1, r²)    →  g_00=1, g_11=4, off-diag=0
/// Inverse metric: g⁻¹ = diag(1, 1/r²) →  g⁰⁰=1, g¹¹=0.25, off-diag=0
///
/// Non-zero partial derivatives of g (layout [i,j,k] = ∂_k g_{ij}):
///   ∂_0 g_{11} = ∂_r(r²)|_{r=2} = 2r = 4  →  partial_g[1,1,0] = 4
///   all others = 0
///
/// Known Christoffel symbols for polar coords:
///   Γ^r_{θθ} = −r  =  Γ^0_{11} = −2
///   Γ^θ_{rθ} = Γ^θ_{θr} = 1/r  =  Γ^1_{01} = Γ^1_{10} = 0.5
///   all others = 0
#[test]
#[serial]
fn test_from_metric_polar_coordinates() {
    // g_{ij}: Tensor<0,2>, layout [i,j]
    let g: Tensor<0, 2> = Tensor::from_f64(2, vec![
        1.0, 0.0,  // g_00, g_01
        0.0, 4.0,  // g_10, g_11
    ]);

    // g^{ij}: Tensor<2,0>, layout [i,j]
    let g_inv: Tensor<2, 0> = Tensor::from_f64(2, vec![
        1.0,  0.0,   // g^00, g^01
        0.0,  0.25,  // g^10, g^11
    ]);

    // ∂_k g_{ij}: Tensor<0,3>, layout [i,j,k]
    // Only ∂_0 g_{11} = 4 is non-zero → flat index for [1,1,0] = 1*4+1*2+0 = 6
    let partial_g: Tensor<0, 3> = Tensor::from_f64(2, vec![
        0.0, 0.0,  // [0,0,0]=∂_0 g_00, [0,0,1]=∂_1 g_00
        0.0, 0.0,  // [0,1,0]=∂_0 g_01, [0,1,1]=∂_1 g_01
        0.0, 0.0,  // [1,0,0]=∂_0 g_10, [1,0,1]=∂_1 g_10
        4.0, 0.0,  // [1,1,0]=∂_0 g_11, [1,1,1]=∂_1 g_11
    ]);

    let gamma = Christoffel::from_metric(&g, &g_inv, &partial_g);

    // All-zero symbols
    assert_eq!(gamma.component(0, 0, 0).result,  0.0); // Γ^r_{rr}
    assert_eq!(gamma.component(0, 0, 1).result,  0.0); // Γ^r_{rθ}
    assert_eq!(gamma.component(0, 1, 0).result,  0.0); // Γ^r_{θr}
    assert_eq!(gamma.component(1, 0, 0).result,  0.0); // Γ^θ_{rr}
    assert_eq!(gamma.component(1, 1, 1).result,  0.0); // Γ^θ_{θθ}

    // Non-zero symbols
    assert_eq!(gamma.component(0, 1, 1).result, -2.0); // Γ^r_{θθ} = −r = −2
    assert_eq!(gamma.component(1, 0, 1).result,  0.5); // Γ^θ_{rθ} = 1/r = 0.5
    assert_eq!(gamma.component(1, 1, 0).result,  0.5); // Γ^θ_{θr} = 1/r = 0.5
}
