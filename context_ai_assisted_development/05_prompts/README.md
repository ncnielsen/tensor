# Prompts

Reusable, task-specific prompts for common development activities.

## What goes here

Prompt templates for recurring tasks. Each prompt should specify:
- What context documents to include alongside it
- The task objective
- Constraints and quality checks
- Expected output format

## File naming

Name by task type:

```
implement_operation.md   — adding a new tensor operation
implement_type.md        — adding a new mathematical type
add_tests.md             — writing tests for existing code
debug_numerics.md        — diagnosing numerical issues
refactor_module.md       — restructuring without changing behavior
review_formulas.md       — verifying math against reference
performance_optimize.md  — profiling and optimizing hot paths
```

## Template

```markdown
# Prompt: <task type>

## Include context
<List of files from other folders to include>

## Task
<Clear objective statement>

## Constraints
<What the AI must/must not do>

## Verification
<How to confirm the result is correct>

## Example usage
<A concrete example of invoking this prompt>
```
