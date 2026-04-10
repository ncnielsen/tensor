# Specifications

What to build — Rust types, trait contracts, function signatures, and acceptance criteria.

## What goes here

- **Type specs** — structs with their invariants (e.g., Christoffel lower-index
  symmetry, metric symmetry, extrinsic curvature symmetry)
- **Operation specs** — function signatures, index formulas, edge cases
- **Trait specs** — trait definitions and what implementors must guarantee
- **Module specs** — what a module exposes (`pub`), its dependencies, its responsibilities
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
<Rust struct/enum definitions with field explanations>

## Traits
<Trait bounds, what implementations must guarantee>

## Functions
<pub fn signatures with index formulas>

## Invariants
<Properties that must always hold — enforced at construction or by type system>

## Acceptance Tests
<Concrete input/output pairs or properties to verify>

## Dependencies
<Crate dependencies and internal module imports>
```
