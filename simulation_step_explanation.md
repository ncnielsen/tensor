# One Simulation Step

## 0. State at the start of a step

The grid holds, at every point, the ADM 3+1 variables:

- **γ_{ij}** — the spatial metric (how distances are measured on the 3D slice)
- **K_{ij}** — the extrinsic curvature (how the slice is embedded in 4D spacetime; encodes the "velocity" of the metric)
- **α = 1, β^i = 0** — lapse and shift (geodesic slicing: observers free-fall with no spatial drift)

---

## 1. Evaluate the EM source (`tornado_matter_grid`)

At every grid point, the rotating spotlight selects the currently active vortex source. For that source:

1. **4-potential A_μ(x)** — a static magnetic flux tube centred on the active source location, evaluated via finite differences in all 4 spacetime directions.
2. **Faraday tensor** F_{μν} = ∂_μ A_ν − ∂_ν A_μ — encodes the electromagnetic field strengths.
3. **EM stress-energy tensor** T_{μν} — the quadratic EM field energy/momentum/stress, computed from F_{μν} and the current 4-metric (assembled from γ_{ij} plus geodesic-slicing g_{00} = −1).
4. **ADM projection** → `AdmMatter {ρ, J^i, S_{ij}}`:
   - ρ = T_{00}: EM energy density (seen by the normal observer)
   - S_{ij} = T_{ij}: spatial stress (radiation pressure)

This produces one `AdmMatter` per grid point. The source is **held fixed** across all four RK4 stages — it is an externally prescribed driver.

---

## 2. Compute the right-hand side (`geodesic_rhs_with_matter`)

For each interior point (the boundary 2-cell band is frozen), compute ∂_t γ_{ij} and ∂_t K_{ij}:

### 2a. Spatial geometry at the point

- Read γ_{ij} and K_{ij} from the grid.
- Invert γ_{ij} → γ^{ij} (3×3 Gauss-Jordan).
- Centred finite differences on the neighbouring grid values → ∂_k γ_{ij}.
- **Christoffel symbols** Γ^k_{ij} = ½ γ^{kl}(∂_i γ_{jl} + ∂_j γ_{il} − ∂_l γ_{ij}).

### 2b. Curvature

- Centred FD of Christoffels at the ±1 neighbours → ∂_ν Γ^ρ_{κμ}.
- **Riemann tensor** R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ}.
- **Ricci tensor** ³R_{ij} = R^k_{ikj} (contraction).

### 2c. Vacuum ADM RHS (geodesic slicing)

$$\partial_t \gamma_{ij} = -2 K_{ij}$$

$$\partial_t K_{ij}^{\text{vac}} = {}^{3\!}R_{ij} + K\,K_{ij} - 2 K_{ik}K^k{}_j$$

where K = γ^{ij} K_{ij} is the trace.

### 2d. Matter correction

$$\partial_t K_{ij}^{\text{matter}} = -8\pi S_{ij} + 4\pi\gamma_{ij}(S - \rho)$$

This is the term that couples the EM source to the geometry — it acts as a forcing term that pushes K_{ij} away from its vacuum trajectory.

**Total:** ∂_t K_{ij} = vacuum part + matter correction.

---

## 3. Advance in time (RK4)

Standard 4th-order Runge-Kutta on the field vector (γ_{ij}, K_{ij}) at all interior points simultaneously:

```
k1 = RHS(y_n)
k2 = RHS(y_n + dt/2 · k1)
k3 = RHS(y_n + dt/2 · k2)
k4 = RHS(y_n + dt  · k3)

y_{n+1} = y_n + dt/6 · (k1 + 2k2 + 2k3 + k4)
```

Steps k2–k4 recompute the full geometric RHS (Christoffel → Riemann → Ricci → ADM equations) at the intermediate field values, but reuse the same matter source from step 1. Boundary values are frozen throughout.

---

## 4. Record diagnostics (every `output_every` steps)

| Diagnostic | Formula | Meaning |
|---|---|---|
| **H_L2** | ‖ ³R + K² − K_{ij}K^{ij} ‖₂ | Hamiltonian constraint violation; zero for exact solutions — measures numerical accuracy |
| **K_offdiag_rms** | RMS of {K_{01}, K_{02}, K_{12}} | Zero in flat space; grows as the source seeds rotational extrinsic curvature |
| **γ_perturb_rms** | RMS of γ_{ij} − δ_{ij} | Measures metric deformation from flat |
| **max_ρ_EM** | max |ρ| over grid | Peak EM energy density |

---

## The physical picture

Each step, the active vortex source pushes K_{ij} via the matter correction. Over many steps, as the spotlight rotates around the ring, it deposits extrinsic curvature at successive azimuthal angles — the signature of angular momentum being injected into the geometry. The spatial metric γ_{ij} then responds to the accumulated K through the ∂_t γ = −2K equation, developing off-diagonal structure that reflects the growing spacetime rotation.
