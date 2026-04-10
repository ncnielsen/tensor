# AAD — Automatic Adjoint Differentiation Library

Local crate at `/home/ncn/git/automatic_adjoint_differentiation`.
Crate name: `aad`, edition 2024.

## Purpose

Reverse-mode automatic differentiation. Records arithmetic operations on a global
tape, then computes adjoint (reverse) derivatives via backpropagation. The tensor
library uses `Number` as its scalar type to get exact gradients of any computation.

## Core Types

### `Number` (`number.rs`)
```rust
pub struct Number {
    pub result: f64,   // primal value
    pub id: i64,       // tape node ID (0 when tape is off)
    leaf: bool,        // true if this is an input variable
}
```

**Construction:**
- `Number::new(val)` — creates a leaf node, records on tape (unless `no_tape`)
- Arithmetic ops (`+`, `-`, `*`, `/`) return new `Number` with recorded dependency

**Math operations** (all return `Number`, all tape-aware):
`ln`, `sin`, `cos`, `exp`, `pow(n: f64)`, `sqrt`, `log(b: f64)`, `cdf`

**Mixed-type arithmetic:** `Number op f64` and `f64 op Number` are implemented
for `Add`, `Sub`, `Mul`, `Div`.

### `AutomaticDifferentiator` (`automatic_differentiator.rs`)
```rust
pub struct AutomaticDifferentiator { ... }

impl AutomaticDifferentiator {
    pub fn new() -> Self;
    pub fn derivatives<F>(&mut self, func: F, arguments: &[Number]) -> Evaluation
    where F: FnOnce(&[Number]) -> Number;
}
```

- `derivatives(func, args)` — evaluates `func`, then backpropagates to get
  ∂result/∂arg for each input. Returns `Evaluation { result, derivatives }`.

### `Evaluation` / `Derivative`
```rust
pub struct Evaluation {
    pub result: Number,
    pub derivatives: Vec<Derivative>,
}
pub struct Derivative {
    pub id: i64,
    pub derivative: f64,
}
```

## no_tape Mode

```rust
pub fn no_tape<F, T>(f: F) -> T
```

Disables tape recording for the current thread inside the closure. All `Number`
arithmetic computes only primal values — no mutex, no tape nodes. Restores
recording on return (even on panic, via drop guard).

**When to use:** any hot path that needs `Number` values but not gradients —
solver RHS evaluations, grid sweeps, residual computation.

**Critical:** all operator variants (Number+Number, Number+f64, f64+Number) are
guarded. Earlier bug where mixed-type ops bypassed the guard caused 47x slowdown.

## Dependencies

```toml
[dependencies]
once_cell = "1.21.3"
ordered_hash_map = "0.4.0"
global_counter = "0.2.2"
sorted-vec = "0.8.6"
statrs = "0.18.0"
```

## Usage from Tensor Crate

```toml
# in tensor's Cargo.toml
[dependencies]
aad = { path = "../automatic_adjoint_differentiation" }
```

```rust
use aad::number::Number;
use aad::no_tape;
use aad::automatic_differentiator::AutomaticDifferentiator;

// Hot path — no tape
let residual = no_tape(|| compute_rhs(&grid));

// Gradient computation — tape on
let mut ad = AutomaticDifferentiator::new();
let eval = ad.derivatives(|args| objective(args), &inputs);
let grad = eval.derivatives; // ∂objective/∂input_i
```

## Thread Safety Caveat

The global tape uses `Mutex`-protected data structures. In tests, use `#[serial]`
to avoid concurrent tape corruption. The `no_tape` flag is thread-local and safe.
