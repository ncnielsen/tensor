# Testing Conventions

## Every Phase Gets Tests

Every phase of a project must have its own test suite before the phase is
considered complete. Tests are the proof that a phase works. No phase is done
until its tests pass.

## L0 Tests — Requirements Level

All tests are written at the **requirements level** (L0): they verify observable
behavior as stated in the specification, not internal implementation details.

Use the **Arrange-Act-Assert** pattern:

```rust
#[test]
fn test_contraction_removes_one_upper_and_one_lower_index() {
    // Arrange — set up inputs matching the spec
    let t = Tensor::<2, 1>::from_f64(dim, &components);

    // Act — perform the operation under test
    let result = contract(&t, 0, 0);

    // Assert — verify the requirement is met
    assert_eq!(result.rank(), (1, 0));
    assert_abs_diff_eq!(result.component(&[1]), expected, epsilon = 1e-12);
}
```

**Rules:**
- Test the *what* (spec), not the *how* (implementation)
- Each test targets one requirement or one edge case
- Test names describe the requirement: `test_<what>_<expected_behavior>`
- If a test would break from a valid refactor, it's testing implementation — rewrite it

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
