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
- `partial_g` provided via AAD (function regime), FD (grid regime), or analytic closure (tests)
- `component(i, j, k)` accessor
- `from_f64` constructor for testing
- **Tests:** flat metric → Γ = 0, Schwarzschild Christoffel against known values,
  symmetry Γ^i_{jk} = Γ^i_{kj} holds for arbitrary input

### Phase 6 — Covariant Derivative
Tensor differentiation on curved manifolds.

- `covariant_derivative(tensor, partial_deriv, christoffel)` → Tensor<M, N+1>
- `partial_deriv` is the pre-computed derivative of the tensor (from phase 4: AAD, FD, or analytic)
- General formula: +Γ per upper index, -Γ per lower index
- Partial derivative layout convention: derivative direction k appended last
- **Tests:** covariant derivative of scalar = partial derivative, ∇g = 0 (metric
  compatibility), covariant derivative of vector in flat space = partial derivative

### Phase 7 — Curvature Tensors
The full pipeline from Christoffel to Einstein tensor.

- `ChristoffelDerivative` — ∂_k Γ^i_{jl} via AAD (function regime) or FD (grid regime)
- `riemann(christoffel, christoffel_deriv)` → Tensor<1,3>
- `ricci_tensor(riemann, dim)` → Tensor<0,2> (contract Riemann)
- `ricci_scalar(ricci, g_inv)` → Number (trace)
- `einstein_tensor(ricci, ricci_scalar, metric)` → Tensor<0,2>
- **Tests:** flat space → all curvature = 0, Schwarzschild Ricci = 0 (vacuum),
  Einstein tensor symmetry, Bianchi identity ∇^μ G_{μν} = 0 (numerical check)

### Phase 8 — Electromagnetic Source
Faraday tensor and EM stress-energy for the tornado source.

- `faraday(A, point, h)` → Tensor<0,2> — F_{μν} = ∂_μ A_ν - ∂_ν A_μ via AAD or FD
- `em_stress_energy(F, g, g_inv, mu_0)` → Tensor<0,2>
- Verify antisymmetry of F, symmetry of T, trace-free T
- **Tests:** uniform B-field → known T_{μν}, point source E-field, F antisymmetric,
  T^{EM} trace = 0

### Phase 9 — Einstein Residual + Newton-Raphson Solver
Root-finding for the field equations. AAD provides Jacobians here.

- `einstein_residual(metric_fn, matter_fn, point, kappa)` — full G-κT pipeline
  (function-based regime: AAD for all spatial derivatives through the entire
  g → Γ → ∂Γ → R → G chain, machine-precision curvature)
- `newton_step(F, x)` — Jacobian via AAD tape end-to-end, Gaussian elimination
  (coordinates → metric → curvature → residual, all taped, backprop for exact Jacobian)
- `solve(initial_guess, F, tolerance, max_iter)` → metric solution
- **Derivative strategy:** the entire residual is a single taped computation graph.
  No FD needed in this regime — AAD provides both the spatial derivatives inside
  the residual and the Jacobian of the residual w.r.t. metric unknowns.
- **Tests:** flat vacuum has zero residual, Newton step on simple F(x)=0 converges,
  Schwarzschild metric satisfies vacuum residual ≈ 0

### Phase 10 — ADM Types and Evolution Equations
3+1 decomposition for time evolution.

- `ExtrinsicCurvature` — symmetric K_{ij}, dim=3
- `AdmState` — (γ_{ij}, K_{ij}, α, β^i)
- `adm_rhs_geodesic(state)` — ∂_t γ and ∂_t K with α=1, β=0
- `adm_rhs_vacuum(state)` — full lapse/shift evolution
- `hamiltonian_constraint(state)` / `momentum_constraint(state)`
- **Tests:** flat space RHS = 0, constraint satisfaction on known initial data,
  K evolution sign correctness

### Phase 11 — ADM Grid and Time Stepping
Spatial grid with RK4 integration.

- `AdmGrid` — flat storage (22 f64/pt), N×N×N, 2-cell boundary band
- `partial_gamma_at`, `christoffel_at`, `christoffel_deriv_at` — FD on grid
- `geodesic_rhs(grid)` — full-grid RHS evaluation (wrap in `no_tape`)
- `adm_step_rk4(grid, dt)` — 4th-order Runge-Kutta step, freeze boundary
- `hamiltonian_l2(grid)` — L2 norm of constraint violation
- **Tests:** flat space no drift (100 steps, H ≈ 0), boundary values unchanged,
  isotropic K evolution matches analytic, RK4 convergence order

### Phase 12 — Matter Coupling and Tornado Source
EM matter on the ADM grid.

- `AdmMatter` — {ρ, j_i, S_{ij}, S} projections of T_{μν}
- `AdmMatter::from_t4d(T, γ_inv)` — geodesic-slicing projection
- `EmSource` — single magnetic Gaussian flux tube
- `TornadoArray` — circular ring of emitters with rotating activation
- `tornado_matter_grid(array, grid, t)` — T_{μν} → AdmMatter at all points
- `adm_step_rk4_with_source(grid, matter, dt)` — matter-coupled evolution
- **Tests:** single vortex T_{μν} matches analytic, tornado activation rotates,
  matter-coupled flat space evolves (K grows from source), energy density positive

### Phase 13 — Simulation Runner
End-to-end tornado simulation with output.

- `TornadoConfig` — grid size, time step, duration, source parameters
- `TornadoSnapshot` / `TornadoResult` — diagnostics at each step
- `run_tornado(config)` → TornadoResult (flat IC, RK4 multi-step, snapshots)
- CSV output, print_summary, peak metric deviation
- Binary `tornado` with CLI args
- **Tests:** short run completes without NaN, constraint stays bounded,
  CSV output parseable, snapshot count matches expected

### Derivative Strategy Summary

| What | Function regime | Grid regime |
|------|----------------|-------------|
| ∂_k g_{μν} | AAD (exact) | FD (O(h²)) |
| ∂_k Γ^i_{jl} | AAD (exact) | FD (O(h²)) |
| ∂_μ A_ν | AAD (exact) | FD (O(h²)) |
| ∂F_i/∂x_j (Jacobian) | AAD (exact) | N/A |
| Analytic ∂_k f | Closures (test/validation) | N/A |

| Regime | Phases | Metric is... | Derivative method |
|--------|--------|-------------|-------------------|
| Function-based | 5-9 (solver) | a callable `Fn(&[Number]) -> Tensor` | AAD through computation graph |
| Grid-based | 10-13 (ADM) | stored f64 values at grid points | FD between neighbors |

### Dependencies

```
Phase:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 9
                          ↓           ↓
                          ├──→ 8 ───→ 12
                          ↓
                         10 → 11 → 12 → 13
```

- Phase 4 (derivatives) is the foundation for all later computation
- Phases 1-7 are the math core (tensor algebra + curvature)
- Phases 8-9 add EM source and solver
- Phases 10-13 build the ADM evolution + simulation

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
