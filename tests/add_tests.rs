#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use aad::automatic_differentiator::AutomaticDifferentiator;
use aad::number::Number;
use serial_test::serial;
use tensor::Tensor;

/// Flush any operations left on the global AAD tape by previous tests.
///
/// Non-adjoint tests perform Number arithmetic outside of derivatives(), which
/// registers operations on the global tape without ever calling global_clear().
/// Calling this at the start of any adjoint test guarantees a clean tape.
fn clear_tape() {
    let arg = Number::new(0.0);
    AutomaticDifferentiator::new().derivatives(|_| Number::new(0.0), &[arg]);
}

#[test]
#[serial]
fn test_vector_addition() {
    let a: Tensor<1, 0> = Tensor::from_f64(3, vec![1.0, 2.0, 3.0]);
    let b: Tensor<1, 0> = Tensor::from_f64(3, vec![4.0, 5.0, 6.0]);
    let c = a + b;

    assert_eq!(c.dim, 3);
    assert_eq!(c.components[0].result, 5.0);
    assert_eq!(c.components[1].result, 7.0);
    assert_eq!(c.components[2].result, 9.0);
}

#[test]
#[serial]
fn test_covector_addition() {
    let a: Tensor<0, 1> = Tensor::from_f64(3, vec![1.0, 0.0, 0.0]);
    let b: Tensor<0, 1> = Tensor::from_f64(3, vec![0.0, 1.0, 0.0]);
    let c = a + b;

    assert_eq!(c.components[0].result, 1.0);
    assert_eq!(c.components[1].result, 1.0);
    assert_eq!(c.components[2].result, 0.0);
}

#[test]
#[serial]
fn test_mixed_tensor_addition() {
    // T^i_j in 2D: 4 components stored as [T^0_0, T^0_1, T^1_0, T^1_1]
    let a: Tensor<1, 1> = Tensor::from_f64(2, vec![1.0, 2.0, 3.0, 4.0]);
    let b: Tensor<1, 1> = Tensor::from_f64(2, vec![5.0, 6.0, 7.0, 8.0]);
    let c = a + b;

    assert_eq!(c.components[0].result, 6.0);
    assert_eq!(c.components[1].result, 8.0);
    assert_eq!(c.components[2].result, 10.0);
    assert_eq!(c.components[3].result, 12.0);
}

#[test]
#[serial]
fn test_vector_addition_adjoint() {
    clear_tape();

    // f(V) = Σ_i (V + W)^i — sum all components to produce a scalar output.
    // The final Add is the last registered node, so adjoint propagation is correct.
    // df/dV^i = 1 for all i.
    let w_vals = vec![10.0, 20.0, 30.0];
    let mut ad = AutomaticDifferentiator::new();
    let v_args = [Number::new(1.0), Number::new(2.0), Number::new(3.0)];

    let eval = ad.derivatives(
        |args| {
            let v: Tensor<1, 0> = Tensor::new(3, args.to_vec());
            let w: Tensor<1, 0> = Tensor::from_f64(3, w_vals.clone());
            let s = v + w;
            s.components[0] + s.components[1] + s.components[2]
        },
        &v_args,
    );

    // (1+10) + (2+20) + (3+30) = 66
    assert_eq!(eval.result, 66.0);
    assert_eq!(eval.derivatives[0].derivative, 1.0);
    assert_eq!(eval.derivatives[1].derivative, 1.0);
    assert_eq!(eval.derivatives[2].derivative, 1.0);
}
