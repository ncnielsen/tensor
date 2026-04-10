# Differential Geometry

Connection, curvature, and covariant differentiation on curved manifolds.

## Christoffel Symbols Γ^λ_{μν}

The Levi-Civita connection — **not a tensor** (transforms inhomogeneously under
coordinate changes). Computed entirely from the metric:

```
Γ^λ_{μν} = ½ g^{λσ} ( ∂_μ g_{νσ} + ∂_ν g_{μσ} - ∂_σ g_{μν} )
```

**Symmetry in lower indices:**

```
Γ^λ_{μν} = Γ^λ_{νμ}
```

This symmetry must be enforced at construction — it reduces storage from d³ to
d · d(d+1)/2 independent components. In 4D: 40 independent components (not 64).

### Partial Derivatives of the Metric

The formula requires ∂_μ g_{νσ} — the partial derivative of each metric component
with respect to each coordinate. For a d-dimensional metric, this is d × d(d+1)/2
values (exploiting metric symmetry), or d³ values without exploiting it.

In code: either pass an analytic derivative function, or approximate with finite
differences:

```
∂_k g_{ij} ≈ ( g_{ij}(x + h·e_k) - g_{ij}(x - h·e_k) ) / (2h)
```

## Covariant Derivative ∇_k

The covariant derivative generalizes partial differentiation to curved spaces.
Unlike partial derivatives, it produces tensors from tensors.

### On a scalar:
```
∇_k φ = ∂_k φ
```

### On a contravariant vector:
```
∇_k V^i = ∂_k V^i + Γ^i_{kl} V^l
```

### On a covariant vector:
```
∇_k W_j = ∂_k W_j - Γ^l_{kj} W_l
```

### General rank (M, N) tensor:
```
∇_k T^{i1...iM}_{j1...jN} = ∂_k T^{i1...iM}_{j1...jN}
    + Σ_{p=1}^{M}  Γ^{ip}_{k l}  T^{i1...l...iM}_{j1...jN}    (one +Γ per upper index)
    - Σ_{q=1}^{N}  Γ^{l}_{k jq}  T^{i1...iM}_{j1...l...jN}    (one -Γ per lower index)
```

The result has rank (M, N+1) — k becomes a new lower index (appended last).

### Partial Derivative Layout

For implementation, the partial derivative ∂_k T is laid out as a flat array with
index order [i1, ..., iM, j1, ..., jN, k], where k (the derivative direction) is
appended as the last index.

## Riemann Curvature Tensor R^ρ_{σμν}

Measures the curvature of spacetime. Rank (1, 3):

```
R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}
```

This requires both Christoffel symbols and their partial derivatives.

### Symmetries of Riemann (with all lower indices R_{ρσμν} = g_{ρα} R^α_{σμν}):
```
R_{ρσμν} = -R_{σρμν} = -R_{ρσνμ} = R_{μνρσ}
R_{ρσμν} + R_{ρμνσ} + R_{ρνσμ} = 0    (first Bianchi identity)
```

In 4D: 20 independent components (not 256).

### Christoffel Derivative Layout

∂_k Γ^i_{jl} has index order [i, j, l, k] — the derivative direction k appended last.

The derivative can be computed analytically or via finite differences:

```
∂_k Γ^i_{jl} ≈ ( Γ^i_{jl}(x + h·e_k) - Γ^i_{jl}(x - h·e_k) ) / (2h)
```

## Ricci Tensor R_{μν}

Contraction of the Riemann tensor on its first and third indices:

```
R_{μν} = R^λ_{μλν} = Σ_λ R^λ_{μλν}
```

Symmetric: R_{μν} = R_{νμ}. In 4D: 10 independent components.

## Ricci Scalar R

Trace of the Ricci tensor:

```
R = g^{μν} R_{μν} = Σ_{μν} g^{μν} R_{μν}
```

A single scalar measuring the overall curvature at a point.

## Einstein Tensor G_{μν}

```
G_{μν} = R_{μν} - ½ g_{μν} R
```

Symmetric, divergence-free (∇^μ G_{μν} = 0 by the contracted Bianchi identity).
In 4D: 10 independent components.

## The Pipeline

The complete computation chain from metric to Einstein tensor:

```
g_{μν}  →  ∂g  →  Γ^λ_{μν}  →  ∂Γ  →  R^ρ_{σμν}  →  R_{μν}  →  R  →  G_{μν}
```

Each step depends only on the previous step's output (plus the metric/inverse metric
for index raising and the trace).
