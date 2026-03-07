# Tensor

A Rust library for Automatic Adjoint Differentiation (AAD) on general Tensors of rank (m, n).

This crate provides a tensor type whose components are differentiable `Number` values, enabling automatic differentiation of arbitrary tensor expressions at no extra cost. Differentiation of any tensor expression is handled entirely by the companion [`aad`](../automatic_adjoint_differentiation) crate via reverse-mode adjoint propagation.

---

## Background

### Why Tensors?

The laws of physics do not depend on the coordinate system in which they are expressed. A mathematical object that represents a physical quantity should therefore be *invariant under coordinate transformation* — it should evaluate to the same result regardless of which coordinate system is chosen.

Tensors are precisely such objects. A tensor of rank (m, n) has `m` contravariant (upper) indices and `n` covariant (lower) indices. Under a change of coordinates described by Jacobian `∂x̄ⁱ/∂xʲ`, each upper index transforms by the Jacobian and each lower index transforms by its inverse. These two transformation laws cancel in any contraction of an upper index with a lower index, making the result coordinate-independent.

This has a practical consequence: if you build expressions using only tensors and the operations defined here, the result is guaranteed to be the same in all coordinate systems.

### Why AAD on Tensors?

Differentiating a tensor expression with respect to its inputs is a fundamental operation in simulation and optimisation. Finite-difference approximations introduce algorithmic error. Symbolic differentiation does not scale. Automatic Adjoint Differentiation (AAD) avoids both problems by decomposing the expression into elementary operations, applying the chain rule analytically at each step, and propagating adjoints backwards through the computation graph in a single reverse pass — regardless of the number of input parameters.

By making tensor *components* the differentiable primitive, this library inherits full AAD for free: every tensor operation reduces to scalar arithmetic on `Number` values, and the existing AAD engine handles the rest.

---

## Design

### `Tensor<const M: usize, const N: usize>`

The central type. `M` and `N` are the contravariant and covariant ranks respectively, encoded as const generics so that rank mismatches (e.g. adding a vector to a covector) are caught at **compile time**.

```rust
let v: Tensor<1, 0> = Tensor::from_f64(3, vec![1.0, 2.0, 3.0]); // contravariant vector
let w: Tensor<0, 1> = Tensor::from_f64(3, vec![4.0, 5.0, 6.0]); // covariant vector (one-form)
let g: Tensor<0, 2> = Tensor::from_f64(3, components);           // metric tensor
```

Components are stored in row-major order with upper indices preceding lower indices. For `T^{i₁...iₘ}_{j₁...jₙ}` in `d` dimensions, the component at multi-index `[i₁,...,iₘ, j₁,...,jₙ]` lives at flat position:

```
i₁·d^(m+n-1) + i₂·d^(m+n-2) + ... + jₙ·d⁰
```

### Operations

| Operation | Signature | Notes |
|---|---|---|
| Addition | `Tensor<M,N> + Tensor<M,N> → Tensor<M,N>` | Rank enforced at compile time |
| Outer product | `outer(Tensor<M1,N1>, Tensor<M2,N2>) → Tensor<M1+M2, N1+N2>` | Index layout: `[upper_a, upper_b, lower_a, lower_b]` |
| Contraction | `contract(Tensor<M,N>, upper_idx, lower_idx) → Tensor<M-1, N-1>` | Only legal contraction: upper + lower |

Contracting two upper or two lower indices is not a tensor operation. Attempting to call `contract` on a `Tensor<0, N>` is a **compile error** — the where clause `[(); M-1]:` does not hold when `M = 0`.

### Differentiation

All operations reduce to scalar arithmetic on `Number` values, which are recorded on the AAD tape. To differentiate a tensor expression, wrap it in the `AutomaticDifferentiator::derivatives` closure from the `aad` crate:

```rust
use aad::automatic_differentiator::AutomaticDifferentiator;
use aad::number::Number;
use tensor::{contract, outer, Tensor};

// Differentiate the inner product f(V) = V^i W_i with respect to V.
// The analytical result is df/dV^j = W_j.

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

assert_eq!(eval.result, 32.0);          // 1·4 + 2·5 + 3·6
assert_eq!(eval.derivatives[0].derivative, 4.0); // df/dV⁰ = W₀
assert_eq!(eval.derivatives[1].derivative, 5.0); // df/dV¹ = W₁
assert_eq!(eval.derivatives[2].derivative, 6.0); // df/dV² = W₂
```

Matrix multiplication is outer product followed by contraction:

```rust
// C^i_l = A^{ij} B_{jl}
let ab: Tensor<2, 2> = outer(&a, &b);
let c: Tensor<1, 1> = contract(&ab, 1, 0); // contract upper index 1 with lower index 0
```

### Important constraint on output nodes

The AAD engine identifies the *last registered node* (highest ID) on the tape as the output and assigns it adjoint 1. Tensor expressions should therefore produce a single scalar as their final operation — which is natural in tensor calculus, where physical quantities are always scalars formed by full contraction.

If an expression computes multiple components simultaneously (e.g. `v + w` produces three component additions), summing them into a scalar before returning satisfies this constraint:

```rust
// Correct: final Add is the output node
s.components[0] + s.components[1] + s.components[2]

// Incorrect: components[0] is not the last registered node
s.components[0]
```

Contractions naturally satisfy this constraint because the fold over the summation index makes the final Add the returned value.

---

## Building and Testing

This crate requires Rust nightly for the `generic_const_exprs` feature, which enables const arithmetic in type positions (`Tensor<{M+N}, ...>`). The `rust-toolchain.toml` file pins the toolchain automatically.

```sh
cargo test
```

Tests within each test binary access the same global AAD tape and must run sequentially. The `serial_test` crate enforces this. All tests that perform AAD also call `clear_tape()` at their start to flush any state left by preceding value tests.

---

## Project context

This crate is part of a broader effort to build scalable numerical solvers for Partial Differential Equations on coordinate-independent tensor expressions. It provides the differentiable tensor foundation upon which PDE solvers will be built.

Related projects:
- [`aad`](../automatic_adjoint_differentiation) — Automatic Adjoint Differentiation on scalars, the differentiation engine used by this crate

---

## License

MIT
