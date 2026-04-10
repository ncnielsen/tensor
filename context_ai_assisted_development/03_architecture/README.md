# Architecture

How the Rust crate is structured — layers, modules, data flow, and design decisions.

## What goes here

- **Layer diagram** — the dependency stack (autodiff crate -> tensor crate -> solver/bin)
- **Module map** — which `.rs` files contain what, and their `use` dependencies
- **Data flow** — how data moves through the pipeline
  (metric -> Christoffel -> Riemann -> Ricci -> Einstein -> residual)
- **Design decisions** — choices made and their rationale (especially rejected alternatives)
- **Storage layout** — how tensors are stored in memory (flat Vec<f64>, row-major, index ordering)
- **Trait architecture** — which traits exist, what they abstract, why

## Key documents

```
layers.md           — the crate dependency stack
module_map.md       — file-by-file responsibilities and `pub` surface
data_flow.md        — pipeline from metric to field equations
storage_layout.md   — flat array layout, index encoding/decoding
design_decisions.md — ADRs: what was chosen, what was rejected, why
```

## Design decision format

```markdown
# Decision: <title>

## Context
<What problem needed solving>

## Decision
<What was chosen>

## Rejected alternatives
<What was considered and why it was rejected>

## Consequences
<What this decision enables or constrains>
```
