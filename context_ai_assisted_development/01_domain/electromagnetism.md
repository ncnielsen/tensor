# Electromagnetism in Curved Spacetime

Maxwell's equations and the electromagnetic stress-energy tensor, formulated
for use as a matter source in the Einstein field equations.

## Electromagnetic 4-Potential A_μ

The fundamental EM field is described by a 4-potential:

```
A_μ = (φ/c, A_x, A_y, A_z)
```

where φ is the electric scalar potential and A_i is the magnetic vector potential.

In geometrized units (c = 1): A_μ = (φ, A_x, A_y, A_z).

## Faraday Tensor F_{μν}

The electromagnetic field strength tensor — antisymmetric rank (0, 2):

```
F_{μν} = ∂_μ A_ν - ∂_ν A_μ
F_{μν} = -F_{νμ}
```

In components (flat space, Cartesian):

```
F_{μν} = | 0    -E_x  -E_y  -E_z |
          | E_x   0     B_z  -B_y |
          | E_y  -B_z   0     B_x |
          | E_z   B_y  -B_x   0   |
```

where E_i and B_i are the electric and magnetic field components.

### Raised Index Version

```
F^{μν} = g^{μα} g^{νβ} F_{αβ}
```

In flat Minkowski space (η = diag(-1,1,1,1)):

```
F^{0i} = -F_{0i} = E_i
F^{ij} = F_{ij}
```

### Computing from 4-Potential on a Grid

Using central finite differences:

```
F_{μν}(x) = ( A_ν(x + h·e_μ) - A_ν(x - h·e_μ) ) / (2h)
           - ( A_μ(x + h·e_ν) - A_μ(x - h·e_ν) ) / (2h)
```

The antisymmetry F_{μν} = -F_{νμ} is automatic from the definition.

## Electromagnetic Stress-Energy Tensor

```
T^{EM}_{μν} = (1/μ₀) ( F_{μα} F_ν^α - ¼ g_{μν} F_{αβ} F^{αβ} )
```

Step by step:

1. Raise one index: F_ν^α = g^{αβ} F_{νβ}
2. Contract: F_{μα} F_ν^α = Σ_α F_{μα} · F_ν^α
3. Scalar invariant: F_{αβ} F^{αβ} = Σ_{α,β} F_{αβ} · F^{αβ}
4. Assemble: T_{μν} = (1/μ₀)(step2 - ¼ g_{μν} · step3)

In geometrized Gaussian units: the 1/μ₀ prefactor becomes 1/(4π).

**Properties:**
- Symmetric: T_{μν} = T_{νμ}
- Trace-free: g^{μν} T_{μν} = 0 (EM radiation has zero trace)
- Satisfies ∇^μ T_{μν} = 0 when Maxwell's equations hold

## Maxwell's Equations in Curved Spacetime

### Homogeneous (automatic from F = dA):
```
∇_{[λ} F_{μν]} = 0    or equivalently    ∂_λ F_{μν} + ∂_μ F_{νλ} + ∂_ν F_{λμ} = 0
```

### Inhomogeneous (with source current J^μ):
```
∇_μ F^{μν} = μ₀ J^ν
```

In curved spacetime, the covariant divergence involves Christoffel symbols:
```
∇_μ F^{μν} = ∂_μ F^{μν} + Γ^μ_{μλ} F^{λν} + Γ^ν_{μλ} F^{μλ}
```

For source-free regions: J^μ = 0, so ∇_μ F^{μν} = 0.

## Spacetime Tornado: EM Source Configuration

The "spacetime tornado" uses a circular array of magnetic flux tubes as the
EM source, creating a rotating pattern of stress-energy.

### Single Magnetic Vortex

A Gaussian flux tube along the z-axis centered at (x₀, y₀):

```
A_z(x, y) = amplitude · exp( -((x-x₀)² + (y-y₀)²) / (2σ²) )
A_t = A_x = A_y = 0
```

This produces a magnetic field circulating around (x₀, y₀) in the xy-plane.

### Tornado Array

N emitters arranged in a circle of given radius, activated in sequence:

```
emitter_k position: (R cos(2πk/N), R sin(2πk/N), 0)
active_index(t) = floor(t / period) mod N
```

Only one emitter is active at each time step, creating a "rotating spotlight"
pattern. The rotation frequency is 1/period.

### 3+1 Projection

The 4D T^{EM}_{μν} is projected onto the spatial slice using the matter
projection formulas from adm_decomposition.md to obtain ρ, j_i, S_{ij} for
the ADM evolution equations.
