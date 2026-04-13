# Graph Report - .  (2026-04-13)

## Corpus Check
- Corpus is ~27,548 words - fits in a single context window. You may not need a graph.

## Summary
- 301 nodes · 635 edges · 28 communities detected
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_ADM Evolution RHS|ADM Evolution RHS]]
- [[_COMMUNITY_ADM Grid Time-Stepping|ADM Grid Time-Stepping]]
- [[_COMMUNITY_Curvature Computation|Curvature Computation]]
- [[_COMMUNITY_Dev Conventions & Constraints|Dev Conventions & Constraints]]
- [[_COMMUNITY_Tensor Core Data Structure|Tensor Core Data Structure]]
- [[_COMMUNITY_Metric Operations|Metric Operations]]
- [[_COMMUNITY_Differential Geometry Docs|Differential Geometry Docs]]
- [[_COMMUNITY_Differentiation & Christoffel|Differentiation & Christoffel]]
- [[_COMMUNITY_Tensor Arithmetic Ops|Tensor Arithmetic Ops]]
- [[_COMMUNITY_Christoffel Symbols|Christoffel Symbols]]
- [[_COMMUNITY_Einstein Residual & Enzyme|Einstein Residual & Enzyme]]
- [[_COMMUNITY_Newton-Raphson Solver|Newton-Raphson Solver]]
- [[_COMMUNITY_Covariant Derivative|Covariant Derivative]]
- [[_COMMUNITY_Architecture & Context Docs|Architecture & Context Docs]]
- [[_COMMUNITY_Generic TensorM,N Ops|Generic Tensor<M,N> Ops]]
- [[_COMMUNITY_Gauge Choice Docs|Gauge Choice Docs]]
- [[_COMMUNITY_Scalar f64 Ops|Scalar f64 Ops]]
- [[_COMMUNITY_tensor-core lib|tensor-core lib]]
- [[_COMMUNITY_solver lib|solver lib]]
- [[_COMMUNITY_Specifications Docs|Specifications Docs]]
- [[_COMMUNITY_L0 Testing Convention|L0 Testing Convention]]
- [[_COMMUNITY_Crate Structure Style|Crate Structure Style]]
- [[_COMMUNITY_Naming Conventions|Naming Conventions]]
- [[_COMMUNITY_CFL Stability|CFL Stability]]
- [[_COMMUNITY_Einstein Summation|Einstein Summation]]
- [[_COMMUNITY_Index RaisingLowering|Index Raising/Lowering]]
- [[_COMMUNITY_Tensor Contraction|Tensor Contraction]]
- [[_COMMUNITY_Tensor Outer Product|Tensor Outer Product]]

## God Nodes (most connected - your core abstractions)
1. `riemann()` - 21 edges
2. `AdmGrid` - 17 edges
3. `adm_rhs_geodesic()` - 17 edges
4. `ricci_tensor()` - 16 edges
5. `invert_metric()` - 14 edges
6. `geodesic_rhs()` - 14 edges
7. `adm_rhs_vacuum()` - 14 edges
8. `Tensor<M,N> (rank-(M,N) tensor)` - 13 edges
9. `ricci_scalar()` - 12 edges
10. `Tensor<M, N>` - 12 edges

## Surprising Connections (you probably didn't know these)
- `invert_metric()` --semantically_similar_to--> `gaussian_elim()`  [INFERRED] [semantically similar]
  tensor-core/src/metric.rs → solver/src/newton.rs
- `adm_rhs_geodesic()` --calls--> `riemann()`  [EXTRACTED]
  solver/src/adm.rs → tensor-core/src/curvature.rs
- `adm_rhs_vacuum()` --calls--> `riemann()`  [EXTRACTED]
  solver/src/adm.rs → tensor-core/src/curvature.rs
- `hamiltonian_constraint()` --calls--> `riemann()`  [EXTRACTED]
  solver/src/adm.rs → tensor-core/src/curvature.rs
- `adm_rhs_geodesic()` --calls--> `ricci_tensor()`  [EXTRACTED]
  solver/src/adm.rs → tensor-core/src/curvature.rs

## Hyperedges (group relationships)
- **4D Curvature Pipeline: Christoffel + Derivative → Riemann → Ricci → Einstein** — christoffel_Christoffel, curvature_ChristoffelDerivative, curvature_riemann, curvature_ricci_tensor, curvature_ricci_scalar, curvature_einstein_tensor [EXTRACTED 1.00]
- **ADM Evolution System: AdmState + Christoffel + RHS + RK4 Grid** — adm_AdmState, adm_ExtrinsicCurvature, adm_adm_rhs_geodesic, grid_AdmGrid, grid_adm_step_rk4 [EXTRACTED 0.95]
- **Enzyme Autodiff Jacobian Pattern: reverse_pass → jacobian_from_reverse → linear system** — concept_enzyme_autodiff, deriv_jacobian_from_reverse, newton_newton_step, residual_christoffel_deriv_from_jacobian [INFERRED 0.88]
- **Curvature Computation Chain: Metric → Christoffel → Riemann → Ricci → Einstein** — domain_tensor_metric, domain_diffgeo_christoffel, domain_diffgeo_riemann, domain_diffgeo_ricci_tensor, domain_diffgeo_ricci_scalar, domain_diffgeo_einstein_tensor, domain_diffgeo_curvature_pipeline [EXTRACTED 1.00]
- **ADM 3+1 Evolution System: Variables + Equations + Constraints + Grid** — domain_adm_variables, domain_adm_evolution_equations, domain_adm_constraints, domain_adm_grid_structure, domain_adm_geodesic_slicing, domain_adm_matter_projections [EXTRACTED 1.00]
- **Derivative Strategy: Function Regime (Enzyme) vs Grid Regime (FD)** — project_tensor_core_function_regime, project_tensor_core_grid_regime, domain_numm_finite_differences, project_tensor_core_enzyme, rationale_enzyme_not_grid [EXTRACTED 0.95]

## Communities

### Community 0 - "ADM Evolution RHS"
Cohesion: 0.16
Nodes (25): AdmState — 3+1 dynamical variables (γ_{ij}, K_{ij}, α, β^i), GaugeDeriv — spatial derivatives of lapse α and shift β^i, adm_rhs_geodesic(), adm_rhs_vacuum(), AdmState, ExtrinsicCurvature, flat_geodesic_rhs_zero(), flat_hamiltonian_constraint_zero() (+17 more)

### Community 1 - "ADM Grid Time-Stepping"
Cohesion: 0.2
Nodes (15): ExtrinsicCurvature — K_{ij} symmetric spatial tensor, Ghost zone / boundary band (2-cell frozen boundary in AdmGrid), AdmGrid — flat 3D spatial grid for ADM evolution (22 f64/point), adm_step_rk4(), AdmGrid, boundary_cells_unchanged(), christoffel_at(), christoffel_deriv_at() (+7 more)

### Community 2 - "Curvature Computation"
Cohesion: 0.25
Nodes (20): bianchi_identity_flrw(), ChristoffelDerivative, einstein_symmetry(), einstein_tensor(), flat_space_all_curvature_zero(), flrw_christoffel_at(), flrw_einstein_analytic(), flrw_metric() (+12 more)

### Community 3 - "Dev Conventions & Constraints"
Cohesion: 0.08
Nodes (27): Hot Path Performance Rules, Memory Layout Convention (flat Vec<f64>, row-major), Serial Test Execution for AAD Tape Safety, Tape Cleanup Pattern (clear_tape before adjoint tests), ADM Constraint Equations (Hamiltonian H, Momentum M_i), ADM Evolution Equations (∂_t γ_{ij}, ∂_t K_{ij}), ADM Grid Structure (22 f64/pt, N×N×N, 2-cell boundary band), ADM Variables (γ_{ij}, K_{ij}, α, β^i) (+19 more)

### Community 4 - "Tensor Core Data Structure"
Cohesion: 0.13
Nodes (12): flat_index_roundtrip(), from_vec_and_access(), from_vec_wrong_size(), index_out_of_range(), minkowski_metric(), rank_queries(), scalar_is_single_element(), set_component() (+4 more)

### Community 5 - "Metric Operations"
Cohesion: 0.22
Nodes (23): Flat Vec<f64> row-major tensor storage, approx_eq(), assert_metric_inverse_identity(), enforce_symmetry(), enforce_symmetry_averages(), enforce_symmetry_noop_on_symmetric(), inverse_symmetry_preserved(), invert_metric() (+15 more)

### Community 6 - "Differential Geometry Docs"
Cohesion: 0.13
Nodes (23): ADM Matter Projections (ρ, j_i, S_{ij}, S), Christoffel Symbols Γ^λ_{μν}, Covariant Derivative ∇_k, Curvature Computation Pipeline (g→∂g→Γ→∂Γ→R→Ric→R_scalar→G), Einstein Tensor G_{μν}, Ricci Scalar R, Ricci Tensor R_{μν}, Riemann Curvature Tensor R^ρ_{σμν} (+15 more)

### Community 7 - "Differentiation & Christoffel"
Cohesion: 0.19
Nodes (16): Christoffel::from_metric() — compute Γ from g_inv and ∂g, deep_chain(), enzyme_deep_chain_vs_fd(), enzyme_jacobian_analytic(), enzyme_jacobian_vs_fd(), enzyme_schwarzschild_partials(), enzyme_vs_fd_schwarzschild(), fd_jacobian_standalone() (+8 more)

### Community 8 - "Tensor Arithmetic Ops"
Cohesion: 0.17
Nodes (10): contract(), contract_identity_is_dim(), contract_rank_21_to_10(), contract_trace_manual(), outer(), outer_dim_mismatch(), outer_then_contract_is_dot(), outer_two_covectors() (+2 more)

### Community 9 - "Christoffel Symbols"
Cohesion: 0.26
Nodes (9): Christoffel, flat_metric_christoffel_zero(), from_flat_and_access(), schwarzschild_christoffel_analytic(), schwarzschild_christoffel_from_analytic_partials(), schwarzschild_christoffel_via_enzyme(), schwarzschild_metric(), set_component_symmetric() (+1 more)

### Community 10 - "Einstein Residual & Enzyme"
Cohesion: 0.31
Nodes (8): Enzyme Autodiff (#[autodiff_reverse] / forward-mode), ChristoffelDerivative — ∂_l Γ^i_{jk}, christoffel_deriv_from_jacobian(), enzyme_christoffel_deriv_vs_fd(), schwarzschild_christoffel(), schwarzschild_christoffel_matches_reference(), schwarzschild_metric_and_inverse(), schwarzschild_vacuum_residual_zero()

### Community 11 - "Newton-Raphson Solver"
Cohesion: 0.42
Nodes (8): circle_line(), gaussian_elim(), gaussian_elim_3x3(), newton_2d_converges(), newton_quadratic_converges(), newton_step(), quadratic(), solve()

### Community 12 - "Covariant Derivative"
Cohesion: 0.52
Nodes (6): covariant_derivative(), metric_compatibility_schwarzschild(), metric_compatibility_via_enzyme(), scalar_gradient_equals_partial(), schwarzschild_metric(), vector_flat_space_equals_partial()

### Community 13 - "Architecture & Context Docs"
Cohesion: 0.33
Nodes (6): Curvature Pipeline: metric → Christoffel → Riemann → Ricci → Einstein, Context Index (top-level progressive disclosure index), Context Engineering README (folder structure + usage pattern), Architecture README (layers, module map, data flow, ADRs), Progress README (status, roadmap, changelog guidance), Prompts README (task-specific prompt templates)

### Community 14 - "Generic Tensor<M,N> Ops"
Cohesion: 0.4
Nodes (1): &Tensor<M, N>

### Community 15 - "Gauge Choice Docs"
Cohesion: 0.67
Nodes (3): Geodesic Slicing Gauge (α=1, β^i=0), Gauge Freedom and Coordinate Choices (geodesic, harmonic, maximal slicing), Rationale: Geodesic Slicing as Starting Point (simplest gauge for implementation)

### Community 16 - "Scalar f64 Ops"
Cohesion: 1.0
Nodes (1): f64

### Community 17 - "tensor-core lib"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "solver lib"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Specifications Docs"
Cohesion: 1.0
Nodes (1): Specifications Directory

### Community 20 - "L0 Testing Convention"
Cohesion: 1.0
Nodes (1): L0 Requirements-Level Tests (Arrange-Act-Assert)

### Community 21 - "Crate Structure Style"
Cohesion: 1.0
Nodes (1): Rust Crate Structure Convention

### Community 22 - "Naming Conventions"
Cohesion: 1.0
Nodes (1): Rust Naming Conventions (PascalCase, snake_case)

### Community 23 - "CFL Stability"
Cohesion: 1.0
Nodes (1): CFL Stability Condition

### Community 24 - "Einstein Summation"
Cohesion: 1.0
Nodes (1): Einstein Summation Convention

### Community 25 - "Index Raising/Lowering"
Cohesion: 1.0
Nodes (1): Raising and Lowering Indices

### Community 26 - "Tensor Contraction"
Cohesion: 1.0
Nodes (1): Tensor Contraction

### Community 27 - "Tensor Outer Product"
Cohesion: 1.0
Nodes (1): Outer Product of Tensors

## Knowledge Gaps
- **34 isolated node(s):** `f64`, `Tensor`, `GaugeDeriv — spatial derivatives of lapse α and shift β^i`, `Context Engineering README (folder structure + usage pattern)`, `Prompts README (task-specific prompt templates)` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Scalar f64 Ops`** (2 nodes): `f64`, `.mul()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `tensor-core lib`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `solver lib`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Specifications Docs`** (1 nodes): `Specifications Directory`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `L0 Testing Convention`** (1 nodes): `L0 Requirements-Level Tests (Arrange-Act-Assert)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Crate Structure Style`** (1 nodes): `Rust Crate Structure Convention`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Naming Conventions`** (1 nodes): `Rust Naming Conventions (PascalCase, snake_case)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CFL Stability`** (1 nodes): `CFL Stability Condition`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Einstein Summation`** (1 nodes): `Einstein Summation Convention`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Index Raising/Lowering`** (1 nodes): `Raising and Lowering Indices`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensor Contraction`** (1 nodes): `Tensor Contraction`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensor Outer Product`** (1 nodes): `Outer Product of Tensors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tensor<M,N> (rank-(M,N) tensor)` connect `Metric Operations` to `ADM Evolution RHS`, `ADM Grid Time-Stepping`, `Curvature Computation`, `Differentiation & Christoffel`, `Tensor Arithmetic Ops`, `Covariant Derivative`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `riemann()` connect `Curvature Computation` to `ADM Evolution RHS`, `Einstein Residual & Enzyme`, `Metric Operations`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `geodesic_rhs()` connect `ADM Grid Time-Stepping` to `ADM Evolution RHS`, `Differentiation & Christoffel`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **What connects `f64`, `Tensor`, `GaugeDeriv — spatial derivatives of lapse α and shift β^i` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dev Conventions & Constraints` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Tensor Core Data Structure` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._
- **Should `Differential Geometry Docs` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._