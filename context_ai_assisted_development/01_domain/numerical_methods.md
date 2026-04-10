# Numerical Methods

Discretization, time integration, and root-finding methods used in the tensor
library and GR solver.

## Finite Differences

### Central Differences (2nd-order accurate)

First derivative:
```
∂f/∂x_k ≈ ( f(x + h·e_k) - f(x - h·e_k) ) / (2h)
```

Second derivative:
```
∂²f/∂x_k² ≈ ( f(x + h·e_k) - 2f(x) + f(x - h·e_k) ) / h²
```

Mixed second derivative:
```
∂²f/∂x_j∂x_k ≈ ( f(x+h·e_j+h·e_k) - f(x+h·e_j-h·e_k)
                 - f(x-h·e_j+h·e_k) + f(x-h·e_j-h·e_k) ) / (4h²)
```

### Application to Tensors

For a tensor field T(x) on a grid, finite differences are applied component-wise.
The result ∂_k T has the derivative direction k appended as a new trailing index.

For Christoffel symbols (not tensors), finite differences are still valid because
we work in a single coordinate system throughout.

### Grid Spacing

On a uniform N×N×N grid over domain [-L, L]³:
```
h = 2L / (N - 1)
```

Step size h affects:
- **Accuracy** of finite differences: error ~ O(h²) for central differences
- **Stability** of time integration: Δt must satisfy CFL condition (Δt ≤ C·h)
- **Resolution** of features: need h << feature_size (e.g., Gaussian σ)

## Runge-Kutta 4th Order (RK4)

Standard 4th-order explicit time integration for ∂_t y = f(t, y):

```
k1 = f(t_n, y_n)
k2 = f(t_n + Δt/2, y_n + Δt/2 · k1)
k3 = f(t_n + Δt/2, y_n + Δt/2 · k2)
k4 = f(t_n + Δt, y_n + Δt · k3)

y_{n+1} = y_n + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

**Error:** O(Δt⁴) per step, O(Δt⁴) globally (4th-order convergence).

### For ADM Evolution

The state vector y contains all grid point values (γ_{ij}, K_{ij}, α, β^i).
The RHS function f computes the time derivatives from the ADM evolution equations.

Each RK4 step requires **4 RHS evaluations** (expensive — each involves computing
Christoffel symbols, Ricci tensor, and matter sources at every interior grid point).

### Boundary Treatment

During each RK4 substep:
1. Compute RHS at all interior points
2. Set RHS = 0 at boundary points (freeze boundary values)
3. Update: boundary values remain at their initial values throughout

## Newton-Raphson Root Finding

For solving F(x) = 0 where F: R^n → R^n:

```
x_{n+1} = x_n - J^{-1} F(x_n)
```

### Numerical Jacobian

```
J_{ij} = ∂F_i/∂x_j ≈ ( F_i(x + ε·e_j) - F_i(x - ε·e_j) ) / (2ε)
```

Cost: 2n function evaluations for an n×n Jacobian. For 10 unknowns (symmetric 4D
metric): 20 residual evaluations per Newton step.

### Linear Solve via Gaussian Elimination

Solve J · δx = -F(x) using Gaussian elimination with partial pivoting:

1. Forward elimination with row swaps (pivot on largest absolute value)
2. Back-substitution

No need for iterative solvers at the per-point level — the system is small (n ≤ 10).

### Convergence

Newton-Raphson converges quadratically near the solution:
```
|x_{n+1} - x*| ≈ C · |x_n - x*|²
```

Typical iteration count: 5-15 for well-conditioned problems. Convergence is
monitored by |F(x)|₂ (L2 norm of the residual).

## Boundary Conditions

### Dirichlet (Fixed Value)

Boundary points hold constant values throughout the simulation. Simplest approach,
suitable when the boundary is far from the region of interest.

For flat-space initial conditions: boundary holds γ_{ij} = δ_{ij}, K_{ij} = 0.

### Boundary Band

Use a 2-cell-deep boundary band on each side of the grid:
- Allows central finite differences at the first interior point
- Interior points: indices [2, N-3] (0-based)
- Boundary points: indices [0,1] and [N-2, N-1] on each axis

### Minimum Grid Size

5×5×5 gives 1 interior point at (2,2,2). Useful for testing but not for
real simulations. Practical minimum: ~11×11×11 (7³ = 343 interior points).

## Stability Considerations

### CFL Condition

For wave-like equations (which GR evolution is), the time step must satisfy:

```
Δt ≤ C_CFL · h / v_max
```

where v_max is the maximum signal speed (= c = 1 in geometrized units) and
C_CFL depends on the integration scheme (for RK4, C_CFL ≈ 1.0).

Conservative choice: Δt = 0.25 · h.

### Constraint Damping

The Hamiltonian and momentum constraints should remain near zero during evolution.
Monitor the L2 norm of the Hamiltonian constraint:

```
||H||₂ = sqrt( Σ_{interior} H(x)² / N_interior )
```

If ||H||₂ grows exponentially, the simulation is unstable. Common causes:
- Δt too large (CFL violation)
- Grid too coarse for the features present
- Boundary conditions reflecting energy back into the domain

## Performance Notes

### Hot Path

The RHS evaluation (computing ∂_t γ and ∂_t K at every grid point) is the
computational bottleneck. For each interior point, it requires:
- 6 metric derivative evaluations (finite differences, 2 neighbors each direction)
- Christoffel symbol computation (40 components in 4D, or ~18 in 3D exploiting symmetry)
- Christoffel derivative computation (more finite differences)
- Riemann → Ricci → Ricci scalar contraction
- K_{ij} algebra (raise index, contract, square)
- Matter source evaluation (if present)

### Autodiff Tape

When using an autodiff library (AAD) for automatic differentiation, the global
tape records every arithmetic operation. In the solver hot path, this creates
massive overhead from mutex acquisitions.

**Solution:** wrap the RHS evaluation in a `no_tape` context — the RHS only needs
primal values, not derivatives. Reserve tape recording for when you actually need
gradients (e.g., sensitivity analysis).
