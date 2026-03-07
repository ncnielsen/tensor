#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use aad::automatic_differentiator::AutomaticDifferentiator;
use aad::number::Number;
use serial_test::serial;
use tensor::{outer, Tensor};

fn clear_tape() {
    let arg = Number::new(0.0);
    AutomaticDifferentiator::new().derivatives(|_| Number::new(0.0), &[arg]);
}

#[test]
#[serial]
fn test_outer_product_two_vectors() {
    // A^i ⊗ B^j = C^{ij}
    // A = [1, 2], B = [3, 4]
    // C^{00}=3, C^{01}=4, C^{10}=6, C^{11}=8
    let a: Tensor<1, 0> = Tensor::from_f64(2, vec![1.0, 2.0]);
    let b: Tensor<1, 0> = Tensor::from_f64(2, vec![3.0, 4.0]);
    let c: Tensor<2, 0> = outer(&a, &b);

    assert_eq!(c.dim, 2);
    assert_eq!(c.components.len(), 4);
    assert_eq!(c.component(&[0, 0]).result, 3.0); // 1 * 3
    assert_eq!(c.component(&[0, 1]).result, 4.0); // 1 * 4
    assert_eq!(c.component(&[1, 0]).result, 6.0); // 2 * 3
    assert_eq!(c.component(&[1, 1]).result, 8.0); // 2 * 4
}

#[test]
#[serial]
fn test_outer_product_vector_covector() {
    // A^i ⊗ B_j = T^i_j  (rank 1,1)
    // A = [2, 3], B = [5, 7]
    // T^0_0=10, T^0_1=14, T^1_0=15, T^1_1=21
    let a: Tensor<1, 0> = Tensor::from_f64(2, vec![2.0, 3.0]);
    let b: Tensor<0, 1> = Tensor::from_f64(2, vec![5.0, 7.0]);
    let t: Tensor<1, 1> = outer(&a, &b);

    assert_eq!(t.component(&[0, 0]).result, 10.0);
    assert_eq!(t.component(&[0, 1]).result, 14.0);
    assert_eq!(t.component(&[1, 0]).result, 15.0);
    assert_eq!(t.component(&[1, 1]).result, 21.0);
}

#[test]
#[serial]
fn test_outer_product_adjoint() {
    clear_tape();

    // f(A^i) = Σ_{ij} (A ⊗ B)^{ij} = (Σ_i A^i)(Σ_j B^j)
    // Sum all components so the final Add is the output node.
    // With B=[3,4]: df/dA^i = B^0 + B^1 = 7 for all i.
    let b_vals = vec![3.0, 4.0];
    let mut ad = AutomaticDifferentiator::new();
    let a_args = [Number::new(1.0), Number::new(2.0)];

    let eval = ad.derivatives(
        |args| {
            let a: Tensor<1, 0> = Tensor::new(2, args.to_vec());
            let b: Tensor<1, 0> = Tensor::from_f64(2, b_vals.clone());
            let c: Tensor<2, 0> = outer(&a, &b);
            // A^0*B^0 + A^0*B^1 + A^1*B^0 + A^1*B^1
            c.components[0] + c.components[1] + c.components[2] + c.components[3]
        },
        &a_args,
    );

    // (1*3 + 1*4 + 2*3 + 2*4) = 3+4+6+8 = 21
    assert_eq!(eval.result, 21.0);
    assert_eq!(eval.derivatives[0].derivative, 7.0);
    assert_eq!(eval.derivatives[1].derivative, 7.0);
}
