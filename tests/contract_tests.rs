#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use aad::automatic_differentiator::AutomaticDifferentiator;
use aad::number::Number;
use serial_test::serial;
use tensor::{contract, outer, Tensor};

fn clear_tape() {
    let arg = Number::new(0.0);
    AutomaticDifferentiator::new().derivatives(|_| Number::new(0.0), &[arg]);
}

#[test]
#[serial]
fn test_trace_of_mixed_tensor() {
    // 3D identity tensor T^i_j = δ^i_j, trace = 3.
    // Components stored as [T^0_0, T^0_1, T^0_2, T^1_0, T^1_1, T^1_2, T^2_0, T^2_1, T^2_2]
    let t: Tensor<1, 1> = Tensor::from_f64(
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let scalar: Tensor<0, 0> = contract(&t, 0, 0);

    assert_eq!(scalar.components.len(), 1);
    assert_eq!(scalar.components[0].result, 3.0);
}

#[test]
#[serial]
fn test_inner_product_via_outer_then_contract() {
    // V^i W_i = contract(outer(V, W), 0, 0)
    // V = [1, 2, 3], W = [4, 5, 6]  →  V·W = 4 + 10 + 18 = 32
    let v: Tensor<1, 0> = Tensor::from_f64(3, vec![1.0, 2.0, 3.0]);
    let w: Tensor<0, 1> = Tensor::from_f64(3, vec![4.0, 5.0, 6.0]);
    let vw: Tensor<1, 1> = outer(&v, &w);
    let scalar: Tensor<0, 0> = contract(&vw, 0, 0);

    assert_eq!(scalar.components[0].result, 32.0);
}

#[test]
#[serial]
fn test_inner_product_adjoint() {
    clear_tape();

    // f(V^i) = V^i W_i.  df/dV^j = W_j.
    // V = [1, 2, 3], W = [4, 5, 6]  →  f=32, df/dV = [4, 5, 6]
    //
    // The contraction naturally produces a scalar as the last node:
    //   contract sums Mul(V^0,W_0) + Mul(V^1,W_1) + Mul(V^2,W_2) via fold,
    //   making the final Add the returned (and highest-ID) node.
    let w_vals = vec![4.0, 5.0, 6.0];
    let mut ad = AutomaticDifferentiator::new();
    let v_args = [Number::new(1.0), Number::new(2.0), Number::new(3.0)];

    let eval = ad.derivatives(
        |args| {
            let v: Tensor<1, 0> = Tensor::new(3, args.to_vec());
            let w: Tensor<0, 1> = Tensor::from_f64(3, w_vals.clone());
            let vw: Tensor<1, 1> = outer(&v, &w);
            let scalar: Tensor<0, 0> = contract(&vw, 0, 0);
            scalar.components[0]
        },
        &v_args,
    );

    assert_eq!(eval.result, 32.0);
    assert_eq!(eval.derivatives[0].derivative, 4.0);
    assert_eq!(eval.derivatives[1].derivative, 5.0);
    assert_eq!(eval.derivatives[2].derivative, 6.0);
}

#[test]
#[serial]
fn test_matrix_multiplication_via_outer_then_contract() {
    // C^i_l = A^{ij} B_{jl} — matrix multiply via outer product + contraction.
    //
    // A = [[1,2],[3,4]] as Tensor<2,0>,  B = identity as Tensor<0,2>
    // Outer: T^{ij}_{kl} = A^{ij} * B_{kl}
    // Contract j with k (upper_idx=1, lower_idx=0): C^i_l = Σ_j A^{ij} B_{jl} = A (unchanged)
    let a: Tensor<2, 0> = Tensor::from_f64(2, vec![1.0, 2.0, 3.0, 4.0]);
    let b: Tensor<0, 2> = Tensor::from_f64(2, vec![1.0, 0.0, 0.0, 1.0]);
    let ab: Tensor<2, 2> = outer(&a, &b);
    let c: Tensor<1, 1> = contract(&ab, 1, 0);

    assert_eq!(c.component(&[0, 0]).result, 1.0);
    assert_eq!(c.component(&[0, 1]).result, 2.0);
    assert_eq!(c.component(&[1, 0]).result, 3.0);
    assert_eq!(c.component(&[1, 1]).result, 4.0);
}
