---
status: active
created: 2026-04-10
---

# Project: Research

## Goal

Research and evaluate approaches to implementing a tensor library for general
relativity simulations. Determine the right design choices before writing code.

## Implementation Plan

### Phase 1 — Core Tensor Type
Foundational data structure. Everything else builds on this.

- `Tensor<const M: usize, const N: usize>` — rank (M,N) with const generics
- Flat `Vec<Number>` storage, row-major, upper indices first
- `dim` field (e.g., 4 for spacetime, 3 for spatial)
- `from_f64` / `new` constructors
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

### Phase 4 — Derivative Infrastructure
Three derivative strategies for different contexts. All later phases consume these.

Two regimes with different derivative strategies depending on how the metric
is represented:

**Function-based regime** (solver, analytic metrics, testing):
The metric is a callable `Fn(&[Number]) -> Tensor` that builds a computation
graph from coordinate inputs. Derivatives come from:

- **AAD autodiff** — pass `Number` coordinates as tape leaves, evaluate the metric
  function, backpropagate for exact ∂_k g_{μν}. Machine-precision derivatives with
  no step-size tuning. Same approach chains through Christoffel and curvature —
  the entire pipeline g → Γ → R → G can be differentiated exactly.
- **Analytic closures** — hand-coded derivative functions for validation/testing.
  Used to verify AAD results against known solutions.

**Grid-based regime** (ADM evolution):
The metric γ_{ij} is stored as discrete values at grid points — no computation
graph connects neighboring points. Derivatives must use:

- **Finite differences** — central FD between neighboring grid points.
  Second-order accurate O(h²). The only option for grid-stored data.
  - `partial_gamma_at(grid, ix, iy, iz, h)` — ∂_k γ_{ij} from grid neighbors
  - `christoffel_deriv_at(grid, ix, iy, iz, h)` — ∂_k Γ from grid neighbors

**Shared infrastructure:**
- `partial_deriv(field_fn, point, h, dim)` — central FD operator, works in both
  regimes (on functions or grid lookups wrapped as closures)
- `christoffel_partial_deriv(christoffel_fn, point, h, dim)` — central FD for Christoffel
- Layout convention: derivative direction k appended as last index

**AAD for solver Jacobians:**
- Newton-Raphson Jacobian ∂F_i/∂x_j: the metric function is taped end-to-end
  (coordinates → metric → Christoffel → curvature → residual), then backpropagated
  to get exact Jacobian columns. No numerical FD Jacobian needed.
- Sensitivity analysis: how does the solution change with source parameters?
- ADM grid RHS always runs inside `aad::no_tape` — tape is never used in the
  time-stepping hot path.

**Tests:**
- AAD spatial derivatives match analytic for Schwarzschild metric
- FD spatial derivatives match analytic, second-order convergence (halve h, error drops 4x)
- AAD vs FD agreement within FD truncation error
- `no_tape` wrapping produces identical primal values to taped evaluation
- Full pipeline (g → Γ → R → G) differentiable via AAD for simple metrics

### Phase 5 — Christoffel Symbols
Connection coefficients from the metric. Separate type (not a tensor).

- `Christoffel` struct with lower-index symmetry Γ^i_{jk} = Γ^i_{kj}
- `Christoffel::from_metric(g, g_inv, partial_g)` — the Christoffel formula
- `component(i, j, k)` accessor
- `from_f64` constructor for testing
- **Derivatives:** `partial_g` (∂_k g_{μν}) is an input to `from_metric`, computed by
  the caller using the method appropriate to the regime:
  - **Function regime:** AAD — metric function is taped, backpropagated
    for exact ∂_k g_{μν}
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
  - **Function regime:** AAD — tape the tensor-valued function, backpropagate for
    exact partial derivatives
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
- `ricci_scalar(ricci, g_inv)` → Number (trace)
- `einstein_tensor(ricci, ricci_scalar, metric)` → Tensor<0,2>
- **Derivatives:** `ChristoffelDerivative` (∂_k Γ) is an input, computed by the caller:
  - **Function regime:** AAD — the Christoffel computation is part of the
    taped pipeline (coords → g → Γ), so ∂_k Γ comes from backpropagation through
    the full chain. Machine-precision, no step-size tuning.
  - **Grid regime:** FD — `christoffel_deriv_at` computes (see solver project)
    (Γ[i+1]−Γ[i−1])/2h from Christoffel values at neighboring grid points,
    which are themselves computed from FD metric derivatives.
  - The Riemann → Ricci → Einstein computation is purely algebraic
    (products and contractions of Γ, ∂Γ, g), so it is regime-agnostic.
- **Tests:** flat space → all curvature = 0, Schwarzschild Ricci = 0 (vacuum),
  Einstein tensor symmetry, Bianchi identity ∇^μ G_{μν} = 0 (numerical check)

### Derivative Strategy Summary

| What | Method |
|------|--------|
| ∂_k g_{μν} | AAD (exact) — metric function is taped |
| ∂_k Γ^i_{jl} | AAD (exact) — chain through taped g → Γ pipeline |
| Analytic ∂_k f | Closures (test/validation only) |

This library provides the **function-based regime**: the metric is a callable
`Fn(&[Number]) -> Tensor`, and all spatial derivatives come from AAD
backpropagation through the computation graph. The grid-based regime (FD on
stored values) is in the solver project.

### Dependencies

```
Phase:  1 → 2 → 3 → 4 → 5 → 6 → 7
```

- Phase 4 (derivatives) is the foundation for all later computation
- Phases 1-7 are the math core (tensor algebra + curvature)
- The solver and simulation projects build on this library

## Decisions

- **Language:** Rust

## Open Questions

- Generic over scalar type, or concrete f64?
- Compile-time rank checking vs runtime?
- Storage format — dense flat array, sparse, or hybrid?
- How to handle Christoffel (not a tensor) cleanly?
- Symmetry exploitation — store only independent components, or full + enforce?
- Autodiff integration — baked in, trait-based, or separate layer?
- Scope — just tensor algebra, or include solvers/grid/ADM from the start?
