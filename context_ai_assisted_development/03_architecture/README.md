# Architecture

How the library is structured — layers, modules, data flow, and design decisions.

## What goes here

- **Layer diagram** — the dependency stack (autodiff -> tensor -> solvers)
- **Module map** — which files contain what, and how they depend on each other
- **Data flow** — how data moves through the pipeline
  (metric -> Christoffel -> Riemann -> Ricci -> Einstein -> residual)
- **Design decisions** — choices made and their rationale (especially rejected alternatives)
- **Storage layout** — how tensors are stored in memory (row-major, index ordering)

## Key documents

```
layers.md          — the three-layer architecture (AAD, tensor, solvers)
module_map.md      — file-by-file responsibilities and dependencies
data_flow.md       — pipeline from metric to field equations
storage_layout.md  — flat array layout, index encoding/decoding
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
