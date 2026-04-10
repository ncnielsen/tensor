---
status: active
created: 2026-04-11
---

# Project: Solver

## Goal

Implement solvers for Einstein's field equations. Includes the function-based
Newton-Raphson solver (AAD) and the grid-based ADM 3+1 time evolution (FD).

## Dependencies

Depends on the tensor library (tensor-core project phases 1-7): core tensor type,
operations, metric, derivatives, Christoffel, covariant derivative, curvature.

## Implementation Plan

### Phase 1 — Einstein Residual + Newton-Raphson Solver
Root-finding for the field equations. **Function regime only — AAD throughout.**

- `einstein_residual(metric_fn, point, kappa)` — full G−κT pipeline
- `newton_step(F, x)` — Jacobian via AAD, Gaussian elimination
- `solve(initial_guess, F, tolerance, max_iter)` → metric solution
- **Derivatives — AAD exclusively (no FD):** the metric is a callable function,
  so the entire pipeline is a single taped computation graph:
  - **Spatial derivatives (∂_k g, ∂_k Γ):** coordinates are tape leaves, the
    metric function and Christoffel formula are recorded ops. Backpropagation
    gives exact ∂_k g_{μν} and ∂_k Γ^i_{jl} — no finite differences, no
    step-size parameter.
  - **Solver Jacobian (∂F_i/∂x_j):** the residual function (coordinates →
    metric → Christoffel → curvature → G−κT) is taped end-to-end.
    Backpropagation gives exact Jacobian columns for Newton-Raphson.
  - This is the regime where AAD provides maximum value: machine-precision
    derivatives through a deep computation chain, avoiding compounding FD error.
- **Tests:** flat vacuum has zero residual, Newton step on simple F(x)=0 converges,
  Schwarzschild metric satisfies vacuum residual ≈ 0

### Phase 2 — ADM Types and Evolution Equations
3+1 decomposition for time evolution. **Grid regime — FD for all spatial derivatives.**

- `ExtrinsicCurvature` — symmetric K_{ij}, dim=3
- `AdmState` — (γ_{ij}, K_{ij}, α, β^i)
- `adm_rhs_geodesic(state)` — ∂_t γ and ∂_t K with α=1, β=0
- `adm_rhs_vacuum(state)` — full lapse/shift evolution
- `hamiltonian_constraint(state)` / `momentum_constraint(state)`
- **Derivatives — FD only:** the 3-metric γ_{ij} and extrinsic curvature K_{ij}
  are stored values at grid points, not computed from coordinates. There is no
  computation graph connecting spatial neighbors, so AAD cannot compute spatial
  derivatives. All ∂_k γ_{ij} and ∂_k Γ^i_{jl} use central finite differences
  between neighboring grid values.
- **Performance:** all RHS evaluations run inside `aad::no_tape` to avoid tape
  overhead — since no AAD derivatives are needed, the tape is pure cost.
- **Tests:** flat space RHS = 0, constraint satisfaction on known initial data,
  K evolution sign correctness

### Phase 3 — ADM Grid and Time Stepping
Spatial grid with RK4 integration. **Grid regime — FD for all spatial derivatives.**

- `AdmGrid` — flat storage (22 f64/pt), N×N×N, 2-cell boundary band
- `partial_gamma_at`, `christoffel_at`, `christoffel_deriv_at` — FD on grid
- `geodesic_rhs(grid)` — full-grid RHS evaluation (wrapped in `no_tape`)
- `adm_step_rk4(grid, dt)` — 4th-order Runge-Kutta step, freeze boundary
- `hamiltonian_l2(grid)` — L2 norm of constraint violation
- **Derivatives — FD exclusively:** this phase implements the concrete FD operators
  that phase 2 depends on:
  - `partial_gamma_at(grid, ix, iy, iz, h)` — central FD: (γ[i+1]−γ[i−1])/2h
    for each spatial direction, giving ∂_k γ_{ij}
  - `christoffel_at` — calls `from_metric` with FD-computed ∂_k γ
  - `christoffel_deriv_at` — FD on the Christoffel values themselves:
    (Γ[i+1]−Γ[i−1])/2h, a second layer of finite differences
  - All O(h²) accuracy, requires minimum 5×5×5 grid (2-cell boundary band
    leaves 1 interior point at (2,2,2))
- **Performance:** entire grid sweep wrapped in `aad::no_tape` — no tape recording,
  no mutex overhead. Pure f64 arithmetic.
- **Tests:** flat space no drift (100 steps, H ≈ 0), boundary values unchanged,
  isotropic K evolution matches analytic, RK4 convergence order

### Derivative Strategy Summary

| What | Function regime (phase 1) | Grid regime (phases 2-3) |
|------|--------------------------|--------------------------|
| ∂_k g_{μν} | AAD (exact) | FD (O(h²)) |
| ∂_k Γ^i_{jl} | AAD (exact) | FD (O(h²)) |
| ∂F_i/∂x_j (Jacobian) | AAD (exact) | N/A |

**Why AAD cannot be used in the grid regime:** AAD computes derivatives by
backpropagating through a recorded computation graph — it answers "how does
this output depend on that input through computation." On the grid, metric
values at neighboring points are independent stored numbers with no arithmetic
relationship to each other. There is no computation graph connecting them, so
AAD has nothing to backpropagate through. Spatial derivatives must be
approximated by finite differences between stored values.

### Dependencies

```
Phase:  1
        ↓
        2 → 3
```

- Phase 1 (NR solver) is independent — function regime, AAD
- Phases 2-3 (ADM) are grid regime, FD
- Both depend on the tensor library (tensor-core project)

## Decisions

## Open Questions
