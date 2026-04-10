# Einstein Field Equations

The governing equations of general relativity, linking spacetime geometry to
matter-energy content.

## The Field Equations

```
G_{μν} = κ T_{μν}

where κ = 8πG/c⁴
```

- G_{μν} — Einstein tensor (geometry, from the metric)
- T_{μν} — stress-energy tensor (matter/energy content)
- G — Newton's gravitational constant
- c — speed of light

In **geometrized units** (G = c = 1): κ = 8π.

These are 10 independent second-order nonlinear PDEs for the 10 independent
components of g_{μν}. However, 4 are constraint equations (via ∇^μ G_{μν} = 0),
leaving 6 true evolution equations — matching the 4 coordinate (gauge) freedoms
that reduce 10 metric components to 6 physical degrees of freedom.

## Stress-Energy Tensor T_{μν}

Symmetric rank (0, 2) tensor describing the density and flux of energy-momentum.

### Components (in a local frame):
```
T_{00} = energy density (ρ)
T_{0i} = energy flux / momentum density
T_{ij} = stress (pressure on diagonal, shear off-diagonal)
```

### Conservation:
```
∇^μ T_{μν} = 0     (follows from ∇^μ G_{μν} = 0)
```

### Sources relevant to this project:
- **Electromagnetic field** — derived from Faraday tensor (see electromagnetism.md)
- **Perfect fluid** — T_{μν} = (ρ + p) u_μ u_ν + p g_{μν}
- **Vacuum** — T_{μν} = 0, so G_{μν} = 0 (vacuum Einstein equations)

## Solving Strategy: Residual Formulation

Rather than solving G_{μν} = κ T_{μν} directly, formulate the **residual**:

```
F_{μν}(g) = G_{μν}(g) - κ T_{μν}
```

The solution satisfies F_{μν} = 0. This converts the PDE into a root-finding
problem at each grid point, amenable to Newton-Raphson iteration.

### Residual Pipeline

At each spatial point, given a metric function g_{μν}(x):

1. Evaluate g_{μν} and g^{μν} at this point
2. Compute ∂_k g_{μν} via finite differences on neighboring points
3. Compute Γ^λ_{μν} from metric + derivatives
4. Compute ∂_k Γ^λ_{μν} via finite differences
5. Compute R^ρ_{σμν} from Γ + ∂Γ
6. Contract to R_{μν}, then R, then G_{μν}
7. Compute T_{μν} from the matter source
8. Return G_{μν} - κ T_{μν} as the residual

### Newton-Raphson for Metric

Treat the independent metric components at each grid point as unknowns.
The metric is symmetric, so in 4D there are 10 unknowns per point.

```
x_{n+1} = x_n - J^{-1} F(x_n)
```

where J is the Jacobian ∂F_i/∂x_j, computed numerically:

```
J_{ij} ≈ ( F_i(x + h·e_j) - F_i(x - h·e_j) ) / (2h)
```

Solve J · δx = -F(x) via Gaussian elimination, then update x += δx.

## Geometric Units and Constants

For numerical work, use **geometrized units** where G = c = 1:

```
κ = 8π
```

Physical quantities have dimensions of length:
- Mass: M → GM/c²
- Time: t → ct
- Energy density: ρ → (G/c⁴)ρ

Conversion back to SI requires reinserting G and c factors.

## Gauge Freedom

The Einstein equations are **diffeomorphism invariant** — any coordinate
transformation of a solution is also a solution. This means 4 coordinates can be
freely chosen (gauge fixing):

- **Harmonic gauge:** □x^μ = 0
- **Geodesic slicing:** lapse α = 1, shift β^i = 0 (simplest for ADM evolution)
- **Maximal slicing:** tr(K) = 0

Gauge choice affects stability and computational cost of the evolution.
