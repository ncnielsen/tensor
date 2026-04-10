# Prompts

Reusable, task-specific prompts for common Rust development activities.

## What goes here

Prompt templates for recurring tasks. Each prompt should specify:
- What context documents to include alongside it
- The task objective
- Constraints and quality checks
- Expected output (Rust code, tests, or analysis)

## File naming

Name by task type:

```
implement_type.md        — adding a new Rust struct/enum with invariants
implement_operation.md   — adding a new tensor operation (fn signature + formula)
add_tests.md             — writing #[test] functions for existing code
debug_numerics.md        — diagnosing numerical issues (NaN, divergence, instability)
refactor_module.md       — restructuring without changing behavior
review_formulas.md       — verifying math against reference formulas
optimize_hotpath.md      — profiling and optimizing with --release benchmarks
```

## Template

```markdown
# Prompt: <task type>

## Include context
<List of files from other folders to include>

## Task
<Clear objective statement>

## Constraints
- Must compile with `cargo build`
- Must pass `cargo test` (with #[serial] where needed)
- <task-specific constraints>

## Verification
<How to confirm the result is correct — cargo test, specific assertions, etc.>
```
