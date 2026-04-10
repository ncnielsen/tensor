---
status: active
created: 2026-04-10
---

# Project: Research

## Goal

Research and evaluate approaches to implementing a tensor library for general
relativity simulations. Determine the right design choices before writing code.

## Plan

(To be iterated)

## Decisions

- **Language:** Rust

## Open Questions

- Generic over scalar type, or concrete f64?
- Compile-time rank checking vs runtime?
- Storage format — dense flat array, sparse, or hybrid?
- How to handle Christoffel (not a tensor) cleanly?
- Symmetry exploitation — store only independent components, or full + enforce?
- Autodiff integration — baked in, trait-based, or separate layer?
- Scope — just tensor algebra, or include solvers/grid/ADM from the start?
