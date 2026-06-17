# Evolution Formulations — ADM vs Alternatives

How to evolve the Einstein equations in time, and why the current ADM +
geodesic slicing setup will not survive a sustained tornado run. Two axes get
conflated as "stability" — they are independent and **both** affect this
project.

See also: [adm_decomposition.md](adm_decomposition.md) (the 3+1 split itself),
[einstein_equations.md](einstein_equations.md) (gauge freedom).

## Axis 1 — Formulation (hyperbolicity)

Is the evolution PDE system well-posed? Do constraint violations grow?

| Formulation | Hyperbolicity | Constraint handling | Extra evolved vars vs ADM | Used by |
|-------------|---------------|---------------------|---------------------------|---------|
| **ADM** (current) | **weakly** → ill-posed IBVP | none — violations grow ~e^{λt} | — | nobody, long-term |
| **BSSN** | strongly | conformal connection Γ̃^i fixes the bad modes; no active damping | φ, K, Γ̃^i (conformal-traceless split) | moving-puncture BH/NS codes |
| **Z4c** | strongly | **active damping** (κ₁,κ₂ drive violations → 0) | + Θ (energy-constraint field) | modern NR (BAM, GR-Athena) |
| **CCZ4** | strongly | active damping, fully covariant | + Z^i, Θ | Einstein Toolkit |
| **GHG** (harmonic) | symmetric hyperbolic (clean wave eqs) | Gundlach damping terms | gauge source functions H^μ | SpEC (binary BH) |

- **ADM is weakly hyperbolic:** some characteristic modes do not propagate, the
  principal-symbol Jacobian is not diagonalizable. A tiny numerical constraint
  violation grows exponentially. NaN is a matter of *when*, not *if*. Fine for
  short test runs (why the 82 tests pass); fatal for sustained evolution.
- **BSSN trick:** promote the contracted conformal Christoffel Γ̃^i to an
  evolved field → converts the dangerous 2nd derivatives in the Ricci tensor
  into 1st-order propagated quantities → strongly hyperbolic. Violations only
  *propagate away*, not actively killed.
- **Z4/CCZ4:** add auxiliary constraint fields + damping parameters →
  violations **decay**. Strictly better than BSSN when constraints get
  continuously kicked — exactly the matter-sourced (EM) tornado case, where the
  source injects constraint-violating perturbations every step.

## Axis 2 — Gauge (the trap we are already in)

Independent of formulation. `adm_rhs_geodesic` uses **geodesic slicing**
(α = 1, β^i = 0). This is the worst gauge:

- Normal observers are geodesics. In any region of positive curvature they
  **focus → coordinate caustic → γ_{ij} degenerates → det γ → 0 →
  `invert_metric` panics.** Happens even in *weak* fields given time. Classic,
  guaranteed failure mode — not a bug, geodesic slicing literally forms
  coordinate singularities.

Real codes never use it for evolution. They use:
- **1+log slicing:** ∂_t α = −2αK — singularity-avoiding ("collapse of the
  lapse" freezes evolution near a forming singularity).
- **Gamma-driver shift:** ∂_t β^i ∝ Γ̃^i — keeps coordinates from
  drifting/stretching.

`adm_rhs_vacuum` already accepts general α, β^i + gauge derivatives, so the
*plumbing* for a live gauge exists. The missing piece is a gauge **evolution
equation** (currently α, β are inputs, not evolved).

## What this means for the tornado

**The deciding unknown is field strength** (target peak metric deviation |h|):

- **Weak field** (likely — EM flux tubes, frame-dragging / gravito-magnetic
  effect): curvature small, caustics + constraint growth slow. ADM + geodesic
  *might* survive a short run (cheap experiment worth trying). A **linearized /
  post-Minkowski** approach (g = η + h, evolve h) would be far simpler, stable,
  and probably capture the physics — a strong candidate not yet considered.
- **Moderate/strong field** (sustained, long runs): ADM dies, geodesic slicing
  dies faster. Minimum viable = **BSSN + 1+log + Gamma-driver**. Better for
  matter-sourced = **Z4c** (damping counteracts the EM source's per-step
  constraint kicks; small increment over BSSN).

## Cost — what actually changes

**Key architectural fact: tensor-core is formulation-agnostic.** The
Riemann/Ricci/Einstein pipeline is reused unchanged by all formulations. The
formulation only rewrites `solver/src/adm.rs` (evolution RHS) and
`solver/src/grid.rs` (what is stored/stepped).

| Switch | Scope | Effort |
|--------|-------|--------|
| add 1+log + Gamma-driver gauge | adm.rs evolution eqs | small — plumbing exists |
| ADM → BSSN | adm.rs + grid.rs (5 conformal vars, new RHS) | medium |
| BSSN → Z4c | + 1 field, damping terms | small increment |
| ADM → GHG | larger rethink (harmonic gauge, 4D) | large |
| linearized | new simpler solver, mostly bypasses adm.rs | small–medium, different |

## Recommendation (decision pending)

1. **Settle the regime first.** Target peak |h| picks the path.
2. **If weak:** prototype linearized evolution — cheapest, stable, likely
   sufficient. Don't pay for BSSN you don't need.
3. **If strong / want robustness:** go **Z4c + 1+log + Gamma-driver**, skip
   plain BSSN (Z4c is a small increment and handles matter-sourced constraint
   violation better). ADM evolution stays as a validated *reference*, not the
   production path.
4. **Regardless:** drop geodesic slicing for anything beyond unit tests — it is
   a guaranteed caustic.
