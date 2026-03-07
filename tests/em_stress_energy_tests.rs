#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::em_stress_energy;
use tensor::Tensor;

/// Zero field → T = 0.
#[test]
#[serial]
fn test_em_stress_energy_zero_field() {
    let f: Tensor<0, 2> = Tensor::from_f64(4, vec![0.0; 16]);
    let g: Tensor<0, 2> = Tensor::from_f64(4, vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]);
    let g_inv: Tensor<2, 0> = Tensor::from_f64(4, vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]);

    let t = em_stress_energy(&f, &g, &g_inv, 1.0);
    for c in &t.components {
        assert_eq!(c.result, 0.0);
    }
}

/// 2D Euclidean space with F_{01} = B (magnetic-like), μ₀ = 1.
///
/// g = g_inv = I,  F_{01} = 3, F_{10} = -3.
///
/// Invariant:  I = F_{01}² + F_{10}² = 9 + 9 = 18
/// A_{00} = F_{01}² = 9,  A_{11} = F_{10}² = 9,  A_{01} = 0
///
/// T_{00} = T_{11} = 9 − ¼·18 = 9 − 4.5 = 4.5
/// T_{01} = T_{10} = 0
#[test]
#[serial]
fn test_em_stress_energy_2d_euclidean() {
    // F: layout [μ,ν], flat [0,0]=0,[0,1]=3,[1,0]=-3,[1,1]=0
    let f: Tensor<0, 2> = Tensor::from_f64(2, vec![0.0, 3.0, -3.0, 0.0]);
    let g: Tensor<0, 2>    = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let g_inv: Tensor<2, 0> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);

    let t = em_stress_energy(&f, &g, &g_inv, 1.0);

    assert_eq!(t.components[0].result, 4.5); // T_{00}
    assert_eq!(t.components[1].result, 0.0); // T_{01}
    assert_eq!(t.components[2].result, 0.0); // T_{10}
    assert_eq!(t.components[3].result, 4.5); // T_{11}
}

/// 4D Minkowski (−,+,+,+), static electric field E = 2 in x-direction, μ₀ = 1.
///
/// F_{01} = 2, F_{10} = -2, all others zero.
///
/// Invariant I = F_{01} g^{00} g^{11} F_{01} + F_{10} g^{11} g^{00} F_{10}
///             = (2)(−1)(1)(2) + (−2)(1)(−1)(−2) = −4 + (−4) = −8
///
/// A_{00} = F_{01} g^{11} F_{01} = (2)(1)(2)   =  4
/// A_{11} = F_{10} g^{00} F_{10} = (−2)(−1)(−2) = −4
/// A_{22} = A_{33} = 0
///
/// T_{00} = 4 − ¼(−1)(−8) = 4 − 2 =  2   = E²/(2μ₀)  ✓ (energy density)
/// T_{11} = −4 − ¼(1)(−8) = −4 + 2 = −2
/// T_{22} = 0 − ¼(1)(−8)  = 2
/// T_{33} = 2,   off-diagonal = 0
#[test]
#[serial]
fn test_em_stress_energy_4d_minkowski_electric() {
    let g: Tensor<0, 2> = Tensor::from_f64(4, vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]);
    let g_inv: Tensor<2, 0> = Tensor::from_f64(4, vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]);

    // F_{01} = 2 (flat index 1), F_{10} = -2 (flat index 4)
    let mut f_vals = vec![0.0f64; 16];
    f_vals[1] =  2.0;
    f_vals[4] = -2.0;
    let f: Tensor<0, 2> = Tensor::from_f64(4, f_vals);

    let t = em_stress_energy(&f, &g, &g_inv, 1.0);

    assert_eq!(t.components[0].result,  2.0);  // T_{00} = E²/(2μ₀)
    assert_eq!(t.components[5].result, -2.0);  // T_{11}
    assert_eq!(t.components[10].result, 2.0);  // T_{22}
    assert_eq!(t.components[15].result, 2.0);  // T_{33}

    // Off-diagonal components are zero
    for (i, c) in t.components.iter().enumerate() {
        if ![0, 5, 10, 15].contains(&i) {
            assert_eq!(c.result, 0.0, "T[{}] should be 0, got {}", i, c.result);
        }
    }
}
