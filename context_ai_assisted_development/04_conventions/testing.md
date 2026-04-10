# Testing Conventions

## Test Location

- Unit tests: `#[cfg(test)] mod tests` at bottom of each source file
- Integration tests: `tests/` directory at crate root
- Binary smoke tests: `#[ignore]` attribute, run explicitly with `--ignored`

## Serial Execution

If using a global autodiff tape (AAD library), all tests touching tape state
must use `#[serial]` from the `serial_test` crate to prevent data races:

```rust
use serial_test::serial;

#[test]
#[serial]
fn test_with_tape() { ... }
```

Tests that only use `f64` (no tape) do not need `#[serial]`.

## Tape Cleanup

Call `clear_tape()` before any test that records adjoint operations:

```rust
#[test]
#[serial]
fn test_adjoint() {
    clear_tape();
    // ... test body
}
```

## Construction Patterns

- **No tape needed:** `Tensor::from_f64(...)`, `Christoffel::from_f64(...)`
- **Tape needed (adjoint):** `Tensor::new(...)` inside `ad.derivatives()` closure

## Numeric Tolerances

- Exact results (flat space, identity operations): assert equality to f64 epsilon
- Finite-difference results: tolerance ~1e-6 to 1e-8 depending on step size
- Iterative solver results: tolerance ~1e-10 or match the solver's own threshold

## Test Naming

`test_<what>_<condition>` — e.g., `test_contraction_rank2`, `test_ricci_flat_space`
