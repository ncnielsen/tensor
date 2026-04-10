# Domain Knowledge

Mathematical and physics foundations the AI needs to write correct code.

## What goes here

- **Tensor algebra** — index notation, Einstein summation, raising/lowering indices,
  symmetries, contractions, outer products
- **Differential geometry** — Christoffel symbols (not tensors!), covariant derivatives,
  Riemann/Ricci/Einstein tensors, metric properties
- **General relativity** — Einstein field equations, ADM 3+1 decomposition, extrinsic
  curvature, lapse/shift, constraint equations
- **Electromagnetism** — Faraday tensor, stress-energy tensor, Maxwell's equations in
  curved spacetime
- **Numerical methods** — finite differences, Runge-Kutta integration, Newton-Raphson,
  boundary conditions

## File naming

Use descriptive names with the topic:

```
tensor_algebra.md
differential_geometry.md
einstein_equations.md
adm_decomposition.md
numerical_methods.md
```

## Guidelines

- Write for an AI that knows general math but not GR-specific conventions
- Include the exact index formulas the code must implement (e.g., Christoffel from metric)
- Note where conventions differ between textbooks (sign conventions, index ordering)
- Keep each file focused on one topic — the AI only needs the relevant one per task
