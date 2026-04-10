# Performance Conventions

## Hot Path Rules

The RHS evaluation (computing time derivatives at every grid point) is the
bottleneck. Rules for code in this path:

- Wrap in `aad::no_tape(|| { ... })` — no tape recording in the solver loop
- Avoid heap allocation — preallocate buffers, reuse vectors
- Prefer `f64` arithmetic over `Number` when gradients aren't needed
- Avoid trait objects / dynamic dispatch in inner loops

## Profiling First

Do not optimize without measurement. Use:

```bash
cargo build --release
cargo bench                    # if benchmarks exist
perf record --call-graph dwarf ./target/release/binary
perf report
```

## Release vs Debug

- Debug builds are ~50-100x slower due to bounds checks and no inlining
- Always benchmark and run real simulations in `--release` mode
- Use `#[cfg(debug_assertions)]` for expensive sanity checks in debug only

## Memory Layout

- Tensors use flat `Vec<f64>` in row-major order — cache-friendly for sequential access
- Grid data uses flat contiguous array with stride-based indexing
- Keep grid point data (22 f64) contiguous per point for locality
