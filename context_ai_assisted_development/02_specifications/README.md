# Specifications

What to build — requirements, type signatures, API contracts, and acceptance criteria.

## What goes here

- **Type specs** — data structures with their invariants (e.g., Christoffel lower-index
  symmetry, metric symmetry, extrinsic curvature symmetry)
- **Operation specs** — input/output types, index formulas, edge cases
- **Module specs** — what a module exposes, its dependencies, its responsibilities
- **Acceptance criteria** — concrete test cases or properties the implementation must satisfy

## File naming

One file per module or major feature:

```
tensor_type.md
christoffel.md
operations.md
covariant_derivative.md
adm_types.md
adm_evolution.md
solver.md
```

## Template

```markdown
# Feature: <name>

## Purpose
<One paragraph: what it does and why>

## Types
<Struct/enum definitions with field explanations>

## Operations
<Function signatures with index formulas>

## Invariants
<Properties that must always hold>

## Acceptance Tests
<Concrete input/output pairs or properties to verify>

## Dependencies
<What this module needs from other modules>
```
