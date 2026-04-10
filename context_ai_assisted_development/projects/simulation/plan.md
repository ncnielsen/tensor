---
status: active
created: 2026-04-11
---

# Project: Simulation

## Goal

Simulate a spacetime tornado using EM sources on the ADM grid. Implements the
electromagnetic stress-energy tensor, matter coupling to the ADM evolution, and
the end-to-end simulation runner.

## Dependencies

Depends on:
- Tensor library (research project) — core types, operations, curvature pipeline
- Solver project — ADM types, grid, and time-stepping infrastructure

## Implementation Plan

### Phase 1 — Electromagnetic Source
Faraday tensor and EM stress-energy tensor.

- `faraday(A, point, h)` → Tensor<0,2> — F_{μν} = ∂_μ A_ν - ∂_ν A_μ
- `em_stress_energy(F, g, g_inv, mu_0)` → Tensor<0,2>
- Verify antisymmetry of F, symmetry of T, trace-free T
- **Derivatives:** the Faraday tensor requires ∂_μ A_ν (partial derivatives of the
  4-potential). FD is used: `EmSource` evaluates A_μ analytically at offset points
  and uses central FD (A(x+h)−A(x−h))/2h to get F_{μν}. The EM source function
  is known analytically (Gaussian flux tube), but FD is used because the result
  feeds into grid-stored ADM quantities — the FD error on F_{μν} is dwarfed by
  the FD error already present in the metric derivatives.
  `em_stress_energy` itself is algebraic (products of F, g, g_inv), no derivatives
  taken internally.
- **Tests:** uniform B-field → known T_{μν}, point source E-field, F antisymmetric,
  T^{EM} trace = 0

### Phase 2 — Matter Coupling and Tornado Source
EM matter on the ADM grid. **Grid regime — FD for spatial derivatives.**

- `AdmMatter` — {ρ, j_i, S_{ij}, S} projections of T_{μν}
- `AdmMatter::from_t4d(T, γ_inv)` — geodesic-slicing projection
- `EmSource` — single magnetic Gaussian flux tube
- `TornadoArray` — circular ring of emitters with rotating activation
- `tornado_matter_grid(array, grid, t)` — T_{μν} → AdmMatter at all points
- `adm_step_rk4_with_source(grid, matter, dt)` — matter-coupled evolution
- **Derivatives — FD for spatial, analytic for EM source:**
  - **Spatial derivatives of γ, K:** FD via solver project grid operators, same
    as vacuum ADM
  - **Faraday tensor (∂_μ A_ν):** FD on the analytically known 4-potential
    (phase 1)
  - `em_stress_energy`, `from_t4d`, `matter_dk_correction` are all algebraic
    (no derivatives taken internally)
- **Tests:** single vortex T_{μν} matches analytic, tornado activation rotates,
  matter-coupled flat space evolves (K grows from source), energy density positive

### Phase 3 — Simulation Runner
End-to-end tornado simulation with output. **Grid regime — inherits FD from solver + phase 2.**

- `TornadoConfig` — grid size, time step, duration, source parameters
- `TornadoSnapshot` / `TornadoResult` — diagnostics at each step
- `run_tornado(config)` → TornadoResult (flat IC, RK4 multi-step, snapshots)
- CSV output, print_summary, peak metric deviation
- Binary `tornado` with CLI args
- **Derivatives:** none taken directly. This phase orchestrates the time-stepping
  loop, calling `adm_step_rk4_with_source` (phase 2) which uses FD internally.
  All computation runs inside `aad::no_tape`.
- **Tests:** short run completes without NaN, constraint stays bounded,
  CSV output parseable, snapshot count matches expected

### Dependencies

```
Phase:  1 → 2 → 3
```

All phases are sequential. The EM source (phase 1) feeds into matter coupling
(phase 2), which feeds into the simulation runner (phase 3).

## Decisions

## Open Questions
