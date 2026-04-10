# ADM 3+1 Decomposition

The Arnowitt-Deser-Misner formalism splits 4D spacetime into a foliation of 3D
spatial slices evolving in time. This converts the Einstein equations from 10
coupled PDEs into an initial-value problem with evolution equations and constraints.

## The 3+1 Split

Spacetime is foliated into spatial hypersurfaces Σ_t labeled by coordinate time t.
The 4D metric decomposes as:

```
ds² = -α² dt² + γ_{ij} (dx^i + β^i dt)(dx^j + β^j dt)
```

### ADM Variables (all 3-dimensional, indices i,j = 0,1,2 for x,y,z):

| Variable | Type | Meaning |
|----------|------|---------|
| γ_{ij} | Symmetric (0,2) tensor | Spatial metric (6 components) |
| K_{ij} | Symmetric (0,2) tensor | Extrinsic curvature (6 components) |
| α | Scalar | Lapse function (how fast time flows) |
| β^i | Vector | Shift vector (how coordinates move between slices) |

**Total: 6 + 6 + 1 + 3 = 16 quantities per spatial point.**

### Relationship to 4D Metric

```
g_{00} = -α² + β_k β^k     (β_k = γ_{ki} β^i)
g_{0i} = β_i
g_{ij} = γ_{ij}
```

Inverse:
```
g^{00} = -1/α²
g^{0i} = β^i/α²
g^{ij} = γ^{ij} - β^i β^j/α²
```

## Extrinsic Curvature K_{ij}

Measures how each spatial slice is embedded in 4D spacetime — the "bending" of
the slice:

```
K_{ij} = -½α⁻¹ (∂_t γ_{ij} - D_i β_j - D_j β_i)
```

where D_i is the 3D covariant derivative compatible with γ_{ij}.

Equivalently, K_{ij} is the negative of the Lie derivative of γ along the
unit normal to the slice, divided by 2.

**Symmetric:** K_{ij} = K_{ji} (6 independent components in 3D).

## Evolution Equations

### Metric evolution:
```
∂_t γ_{ij} = -2α K_{ij} + D_i β_j + D_j β_i
```

### Extrinsic curvature evolution:
```
∂_t K_{ij} = -D_i D_j α + α (R^(3)_{ij} + K K_{ij} - 2 K_{ik} K^k_j)
             + β^k D_k K_{ij} + K_{ik} D_j β^k + K_{kj} D_i β^k
             - 8π α (S_{ij} - ½ γ_{ij} (S - ρ))
```

where:
- R^(3)_{ij} — 3D Ricci tensor of γ_{ij}
- K = γ^{ij} K_{ij} — trace of extrinsic curvature
- D_i — 3D covariant derivative
- ρ, S_{ij}, S — matter projections (see matter section below)

### Geodesic Slicing (simplest gauge)

Set α = 1, β^i = 0. Evolution simplifies to:

```
∂_t γ_{ij} = -2 K_{ij}

∂_t K_{ij} = R^(3)_{ij} + K K_{ij} - 2 K_{ik} K^k_j
             - 8π (S_{ij} - ½ γ_{ij} (S - ρ))
```

This is the starting point for implementation. The shift and lapse terms can be
added later for more sophisticated gauge choices.

## Constraint Equations

Not evolved — they must be satisfied on each time slice. Used as diagnostics to
monitor numerical accuracy.

### Hamiltonian Constraint:
```
H = R^(3) + K² - K_{ij} K^{ij} - 16π ρ = 0
```

where R^(3) = γ^{ij} R^(3)_{ij} is the 3D Ricci scalar and
K² = (γ^{ij} K_{ij})².

### Momentum Constraint:
```
M_i = D_j (K^j_i - δ^j_i K) - 8π j_i = 0
```

In vacuum (ρ = j_i = 0), these simplify to:

```
H = R^(3) + K² - K_{ij} K^{ij} = 0
M_i = D_j K^j_i - D_i K = 0
```

## 3D Spatial Geometry

The 3D Christoffel symbols, Riemann tensor, Ricci tensor, and Ricci scalar are
computed exactly like the 4D versions, but using γ_{ij} instead of g_{μν} and
with all indices running 0..2 instead of 0..3:

```
Γ^(3)^k_{ij} = ½ γ^{kl} (∂_i γ_{jl} + ∂_j γ_{il} - ∂_l γ_{ij})

R^(3)^l_{ijk} = ∂_j Γ^l_{ki} - ∂_k Γ^l_{ji} + Γ^l_{jm} Γ^m_{ki} - Γ^l_{km} Γ^m_{ji}

R^(3)_{ij} = R^(3)^k_{ikj}

R^(3) = γ^{ij} R^(3)_{ij}
```

## Matter Projections (3+1 decomposition of T_{μν})

Given the 4D stress-energy tensor T_{μν}, project onto the spatial slices:

```
ρ = n^μ n^ν T_{μν}              — energy density (scalar)
j_i = -γ^μ_i n^ν T_{μν}         — momentum density (3-vector)
S_{ij} = γ^μ_i γ^ν_j T_{μν}    — spatial stress (symmetric 3-tensor)
S = γ^{ij} S_{ij}               — stress trace (scalar)
```

where n^μ is the unit normal to the spatial slice:

```
n^μ = (1/α, -β^i/α)    (future-pointing unit normal)
n_μ = (-α, 0, 0, 0)
```

In **geodesic slicing** (α=1, β=0):

```
ρ = T_{00}
j_i = -T_{0i}
S_{ij} = T_{ij}
S = γ^{ij} T_{ij}
```

## K_{ij} Squared and Trace

For the evolution and constraint equations:

```
K = γ^{ij} K_{ij}                     — trace
K_{ij} K^{ij} = γ^{ia} γ^{jb} K_{ij} K_{ab}  — squared norm
K² = K · K = (γ^{ij} K_{ij})²         — trace squared
```

## Initial Data

Must satisfy the constraint equations at t = 0.

### Flat space (trivial):
```
γ_{ij} = δ_{ij}    (identity matrix)
K_{ij} = 0
α = 1, β^i = 0
```

Satisfies H = 0, M_i = 0 trivially.

### Perturbations:
Start from flat space and add small perturbations to γ_{ij} or K_{ij}, then
solve the constraints to find consistent initial data. For small perturbations,
the constraint violation is second-order and may be acceptable.

## Grid Structure

For numerical evolution on an N×N×N grid:

- Interior points: indices [2, N-3] (0-based) — where evolution equations apply
- Boundary band: 2 cells deep on each side — held fixed (Dirichlet BC)
- Minimum grid: 5×5×5 (gives exactly 1 interior point at (2,2,2))

Each grid point stores 22 f64 values:
- γ_{ij}: 6 (symmetric)
- K_{ij}: 6 (symmetric)
- α: 1
- β^i: 3
- Reserved/padding: 6 (for future use)

Flat storage: `data[((iz * ny + iy) * nx + ix) * 22 + field_offset]`
