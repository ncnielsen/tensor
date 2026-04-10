# Conventions

Coding style, testing patterns, naming rules, and project-wide practices.
These documents should be **small and always included** in every AI session.

## What goes here

- **Coding style** — naming conventions, module organization, error handling
- **Testing patterns** — test structure, when to use `#[serial]`, tape management
- **Performance rules** — when to use `no_tape`, what to avoid in hot paths
- **Index conventions** — upper-before-lower, row-major, dimension ordering

## Key documents

```
style.md        — naming, formatting, module conventions
testing.md      — test patterns, serial requirements, tape cleanup
performance.md  — hot path rules, no_tape usage, allocation avoidance
index_rules.md  — tensor index ordering conventions used throughout
```

## Guidelines

- Keep each file under 100 lines — these are always-include documents
- Be prescriptive, not descriptive: "do X" not "we sometimes do X"
- Include brief rationale so the AI understands *why*, not just *what*
