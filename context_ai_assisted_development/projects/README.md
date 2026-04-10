# Projects

Temporary working folders for active development efforts. Each subfolder holds
iterative plans, decisions, and context for a single project.

## Lifecycle

1. **Created** — user declares a new project; create a subfolder with a `plan.md`
2. **Iterated** — plan is refined across sessions until the work is complete
3. **Completed** — user declares the project done; mark `plan.md` as completed

## Folder structure per project

```
projects/
└── <project_name>/
    ├── plan.md        — current plan (overwritten as it evolves)
    ├── decisions.md   — key decisions made during iteration (append-only)
    └── notes.md       — scratch space, open questions, blockers (optional)
```

## Rules

- When a project is completed, add a `status: completed` line to `plan.md` frontmatter
- Completed projects remain as a historical record
- One project per subfolder; no nesting
