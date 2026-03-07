#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::faraday;
use tensor::Tensor;

/// Zero 4-potential: all partial derivatives zero → F_{μν} = 0.
#[test]
#[serial]
fn test_faraday_zero_potential() {
    let partial_a: Tensor<0, 2> = Tensor::from_f64(4, vec![0.0; 16]);
    let f = faraday(&partial_a);
    for c in &f.components {
        assert_eq!(c.result, 0.0);
    }
}

/// 2D antisymmetry check.
///
/// partial_a layout [ν, μ]: component(&[ν, μ]) = ∂_μ A_ν
///   ∂_0 A_0 = 2,  ∂_1 A_0 = 5
///   ∂_0 A_1 = 7,  ∂_1 A_1 = 3
///
/// F_{μν} = ∂_μ A_ν − ∂_ν A_μ:
///   F_{00} = ∂_0 A_0 − ∂_0 A_0 = 0
///   F_{01} = ∂_0 A_1 − ∂_1 A_0 = 7 − 5 =  2
///   F_{10} = ∂_1 A_0 − ∂_0 A_1 = 5 − 7 = −2
///   F_{11} = ∂_1 A_1 − ∂_1 A_1 = 0
#[test]
#[serial]
fn test_faraday_2d_antisymmetry() {
    // layout [ν, μ]: [∂_0 A_0, ∂_1 A_0, ∂_0 A_1, ∂_1 A_1]
    let partial_a: Tensor<0, 2> = Tensor::from_f64(2, vec![2.0, 5.0, 7.0, 3.0]);
    let f = faraday(&partial_a);

    assert_eq!(f.components[0].result,  0.0); // F_{00}
    assert_eq!(f.components[1].result,  2.0); // F_{01}
    assert_eq!(f.components[2].result, -2.0); // F_{10}
    assert_eq!(f.components[3].result,  0.0); // F_{11}

    // Antisymmetry: F_{01} = −F_{10}
    assert_eq!(
        f.components[1].result,
        -f.components[2].result
    );
}

/// 4D static uniform electric field in the x-direction.
///
/// 4-potential A_μ = (Ex · x¹, 0, 0, 0) with field strength E = 3.
/// Only non-zero derivative: ∂_1 A_0 = E = 3.
///
/// In partial_a layout [ν, μ] (dim=4), flat index [ν,μ] = ν*4 + μ:
///   [0, 1] = flat 1 → ∂_1 A_0 = 3,  all others = 0.
///
/// F_{μν} = ∂_μ A_ν − ∂_ν A_μ:
///   F_{01} = ∂_0 A_1 − ∂_1 A_0 = 0 − 3 = −3   (= −E_x in Gaussian units)
///   F_{10} = ∂_1 A_0 − ∂_0 A_1 = 3 − 0 =  3   (=  E_x)
///   all other components = 0
#[test]
#[serial]
fn test_faraday_4d_electric_field() {
    // 16 components, all zero except flat index 1 = ∂_1 A_0 = 3
    let mut vals = vec![0.0f64; 16];
    vals[1] = 3.0; // partial_a[0, 1] = ∂_1 A_0
    let partial_a: Tensor<0, 2> = Tensor::from_f64(4, vals);

    let f = faraday(&partial_a);

    // F_{01} = flat index 1 (0*4 + 1)
    assert_eq!(f.components[1].result, -3.0);  // F_{01} = −E_x
    // F_{10} = flat index 4 (1*4 + 0)
    assert_eq!(f.components[4].result,  3.0);  // F_{10} =  E_x

    // All other components zero
    for (i, c) in f.components.iter().enumerate() {
        if i != 1 && i != 4 {
            assert_eq!(c.result, 0.0, "Expected F[{}] = 0, got {}", i, c.result);
        }
    }
}
