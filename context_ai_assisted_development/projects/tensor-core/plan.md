---
status: active
created: 2026-04-10
updated: 2026-04-11
---

# Project: Tensor Core

## Goal

Implement a tensor library for general relativity simulations. All arithmetic
is plain `f64` — derivatives are provided by Enzyme (LLVM-level autodiff via
Rust nightly `#[autodiff]`), not by a custom `Number` type or tape.

## Toolchain

Requires Rust nightly with Enzyme:
- `#![feature(autodiff)]` in crate root
- `RUSTFLAGS="-Zautodiff=Enable"` at build time
- `lto = "fat"` in `[profile.release]`
- `libEnzyme-22.so` in sysroot (downloaded from CI artifacts until rustup
  distributes it — re-download after each `rustup update nightly`)

## Implementation Plan

### Phase 1 — Core Tensor Type
Foundational data structure. Everything else builds on this.

- `Tensor<const M: usize, const N: usize>` — rank (M,N) with const generics
- Flat `Vec<f64>` storage, row-major, upper indices first
- `dim` field (e.g., 4 for spacetime, 3 for spatial)
- `from_vec` / `new` constructors
- `component(&[usize])` / `set_component` accessor
- `flat_index` / `decode_flat_index` encoding/decoding
- **Tests:** construction, component access, round-trip encode/decode, rank/dimension checks

### Phase 2 — Basic Operations
Element-wise and algebraic operations on tensors.

- `Add` / `Sub` (operator overload, same rank)
- `outer(a, b)` → `Tensor<{M1+M2}, {N1+N2}>`
- `contract(t, upper_idx, lower_idx)` → `Tensor<{M-1}, {N-1}>`
- Scalar multiply
- **Tests:** add two rank-(0,2) tensors, outer product dimensions, contraction trace
  equals manual sum, contract identity tensor = trace

### Phase 3 — Metric and Index Operations
The metric tensor and its role in raising/lowering.

- Metric symmetry enforcement (g_{μν} = g_{νμ})
- `invert_metric(g)` → g^{μν} (via Gaussian elimination)
- `raise_index` / `lower_index` using metric/inverse
- Verify g^{μα} g_{αν} = δ^μ_ν
- **Tests:** invert known metrics (Minkowski, Schwarzschild), raise then lower = identity,
  symmetry preserved after inversion

### Phase 4 — Derivative Infrastructure (Enzyme)
Enzyme differentiates compiled LLVM IR — no tape, no `Number` type. Functions
that compute from `f64` inputs are annotated with `#[autodiff_reverse]` and
Enzyme generates exact derivative code at the compiler level.

**How it works in this library:**

The metric is a plain `fn(&[f64]) -> Vec<f64>` (coordinates → flat metric
components). To get ∂_k g_{μν}, annotate the metric function:

```rust
#[autodiff_reverse(d_metric_fn, Duplicated, Duplicated)]
fn metric_fn(coords: &[f64; 4], out: &mut [f64; 16]) { ... }
```

Enzyme generates `d_metric_fn` which computes the Jacobian: how each output
component depends on each input coordinate. This gives exact ∂_k g_{μν}
with zero step-size tuning — it differentiates the compiled machine code.

**Two derivative regimes remain, but for different reasons:**

- **Function regime (solver, analytic metrics, testing):**
  The metric is a callable function. Enzyme provides exact derivatives by
  differentiating through the full pipeline (coords → g → Γ → R → G).
  No finite differences needed anywhere in this chain.

- **Grid regime (ADM evolution):**
  The metric γ_{ij} is stored as discrete values at grid points. There is
  no function connecting neighboring points — Enzyme has nothing to
  differentiate. Spatial derivatives use central FD: (γ[i+1]−γ[i−1])/2h.
  This is inherent to grid-based PDE solving, not a limitation of Enzyme.

**What Enzyme replaces:**
- No `Number` type — all computation is plain `f64`
- No global tape — no mutex contention, no `no_tape` hack
- No AAD library dependency
- Exact derivatives (machine-precision) through arbitrarily deep call chains
- Zero runtime overhead for code that doesn't need derivatives

**Enzyme annotations used:**
- `Active` — scalar input/output whose derivative is returned as a value
- `Duplicated` — slice/array input/output with a shadow buffer for gradients
- `Const` — input not differentiated (e.g., dimension, flags)

**Jacobian extraction (for Newton-Raphson):**
For a vector-valued residual `F: R^n → R^n`, compute each Jacobian row by
seeding the output shadow with a unit vector and calling the generated
reverse-mode function. n calls give the full Jacobian. (Alternatively,
forward mode via `#[autodiff_forward]` gives one column per call — choose
based on n_inputs vs n_outputs.)

**Tests:**
- Enzyme derivatives match analytic for Schwarzschild metric
- Enzyme vs FD agreement within FD truncation error
- Deep chain (g → Γ → R → G) differentiable for simple metrics
- Jacobian of vector function matches FD

### Phase 5 — Christoffel Symbols
Connection coefficients from the metric. Separate type (not a tensor).

- `Christoffel` struct with lower-index symmetry Γ^i_{jk} = Γ^i_{kj}
- `Christoffel::from_metric(g, g_inv, partial_g)` — the Christoffel formula
- `component(i, j, k)` accessor
- `from_flat` constructor for testing
- **Derivatives:** `partial_g` (∂_k g_{μν}) is an input to `from_metric`, computed by
  the caller using the method appropriate to the regime:
  - **Function regime:** Enzyme — annotate the metric function with
    `#[autodiff_reverse]`, call the generated derivative function to get
    exact ∂_k g_{μν}
  - **Grid regime:** FD — `partial_gamma_at` computes (γ[i+1]−γ[i−1])/2h
    from stored grid values (see solver project)
  - **Tests:** analytic closures for validation against known solutions
- **Tests:** flat metric → Γ = 0, Schwarzschild Christoffel against known values,
  symmetry Γ^i_{jk} = Γ^i_{kj} holds for arbitrary input

### Phase 6 — Covariant Derivative
Tensor differentiation on curved manifolds.

- `covariant_derivative(tensor, partial_deriv, christoffel)` → Tensor<M, N+1>
- General formula: +Γ per upper index, -Γ per lower index
- Partial derivative layout convention: derivative direction k appended last
- **Derivatives:** `partial_deriv` is a pre-computed input, not computed internally.
  The caller provides it using the appropriate method:
  - **Function regime:** Enzyme — differentiate the tensor-valued function
  - **Grid regime:** FD — central differences between stored grid values
  - The covariant derivative itself is purely algebraic (partial + Γ corrections),
    so it is regime-agnostic
- **Tests:** covariant derivative of scalar = partial derivative, ∇g = 0 (metric
  compatibility), covariant derivative of vector in flat space = partial derivative

### Phase 7 — Curvature Tensors
The full pipeline from Christoffel to Einstein tensor.

- `ChristoffelDerivative` — ∂_k Γ^i_{jl}
- `riemann(christoffel, christoffel_deriv)` → Tensor<1,3>
- `ricci_tensor(riemann, dim)` → Tensor<0,2> (contract Riemann)
- `ricci_scalar(ricci, g_inv)` → f64 (trace)
- `einstein_tensor(ricci, ricci_scalar, metric)` → Tensor<0,2>
- **Derivatives:** `ChristoffelDerivative` (∂_k Γ) is an input, computed by the caller:
  - **Function regime:** Enzyme — the entire chain (coords → g → Γ) is a plain
    `f64` function. Annotate it with `#[autodiff_reverse]` and Enzyme
    differentiates through the full chain. Machine-precision, no step-size tuning.
  - **Grid regime:** FD — `christoffel_deriv_at` computes (see solver project)
    (Γ[i+1]−Γ[i−1])/2h from Christoffel values at neighboring grid points.
  - The Riemann → Ricci → Einstein computation is purely algebraic
    (products and contractions of Γ, ∂Γ, g), so it is regime-agnostic.
- **Tests:** flat space → all curvature = 0, Schwarzschild Ricci = 0 (vacuum),
  Einstein tensor symmetry, Bianchi identity ∇^μ G_{μν} = 0 (numerical check)

### Derivative Strategy Summary

| What | Function regime | Grid regime |
|------|----------------|-------------|
| ∂_k g_{μν} | Enzyme (exact) | FD (O(h²)) |
| ∂_k Γ^i_{jl} | Enzyme (exact) | FD (O(h²)) |
| ∂F_i/∂x_j (Jacobian) | Enzyme (exact) | N/A |

This library provides the **function-based regime**: the metric is a callable
`fn(&[f64]) -> ...`, and all spatial derivatives come from Enzyme
differentiation of the compiled code. The grid-based regime (FD on stored
values) is in the solver project.

### Dependencies

```
Phase:  1 → 2 → 3 → 4 → 5 → 6 → 7
```

- Phase 4 (derivatives) is the foundation for all later computation
- Phases 1-7 are the math core (tensor algebra + curvature)
- The solver and simulation projects build on this library

## Decisions

- **Language:** Rust (nightly, for `#[autodiff]`)
- **Autodiff:** Enzyme via `#[autodiff]` — replaces custom AAD library.
  No `Number` type, no tape, no mutex. Plain `f64` everywhere.
  Verified working: nightly 1.96.0 (2026-04-10), libEnzyme-22, lto="fat".
- **Scalar type:** `f64` (not a custom `Number`). Enzyme differentiates
  compiled code, so the library never needs to know about autodiff.

## Open Questions

- Compile-time rank checking vs runtime?
- Storage format — dense flat array, sparse, or hybrid?
- How to handle Christoffel (not a tensor) cleanly?
- Symmetry exploitation — store only independent components, or full + enforce?
- Scope — just tensor algebra, or include solvers/grid/ADM from the start?
- Enzyme stability — nightly-only, not yet a rustup component. Fallback plan
  if Enzyme breaks on a future nightly?
