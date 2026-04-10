# Rust Coding Style

## Naming

- Types, traits: `PascalCase` — `Tensor`, `Christoffel`, `ExtrinsicCurvature`
- Functions, methods, variables: `snake_case` — `from_metric`, `covariant_derivative`
- Constants: `UPPER_SNAKE_CASE` — `SPEED_OF_LIGHT`
- Type parameters: single uppercase letter or short descriptive — `T`, `S`, `M`, `N`
- Modules: `snake_case`, one per file — `tensor.rs`, `christoffel.rs`, `adm_rhs.rs`

## Crate Structure

```
src/
├── lib.rs          — pub exports, crate-level docs
├── tensor.rs       — core Tensor type
├── christoffel.rs  — Christoffel symbol type
├── ops/
│   └── mod.rs      — tensor operations (add, outer, contract, etc.)
├── adm.rs          — ADM types
├── adm_rhs.rs      — ADM evolution equations
├── solver.rs       — Newton-Raphson, linear algebra
└── bin/
    └── tornado.rs  — binary entry points
```

## General Rules

- Prefer `impl` blocks over free functions when there's a clear receiver
- Use `pub` sparingly — only expose what's part of the API
- Avoid `unwrap()` in library code; use `Result` or document panics
- No `unsafe` unless absolutely necessary and well-documented
- Prefer owned types over lifetimes at API boundaries for simplicity
- Use `#[inline]` only after profiling shows it matters
