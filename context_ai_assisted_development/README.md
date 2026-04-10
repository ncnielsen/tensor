# Context Engineering — Tensor Library

This directory contains structured context documents for AI-assisted development
of a tensor library for general relativity simulations.

The goal is to provide an AI assistant with precisely the right context at each
stage of development — not too much (dilutes focus), not too little (causes
errors and hallucination).

## Folder Structure

| Folder | Purpose | When to feed to AI |
|--------|---------|-------------------|
| `01_domain/` | Mathematical and physics foundations | At project start, or when working on new physics |
| `02_specifications/` | What to build: requirements, API contracts, type signatures | When starting a new module or feature |
| `03_architecture/` | How it's structured: layers, modules, data flow | When making design decisions or adding modules |
| `04_conventions/` | Coding style, testing patterns, naming, error handling | Every session (keep small, always include) |
| `05_prompts/` | Reusable task-specific prompts | Pick the one matching the current task |
| `06_references/` | External resources: papers, prior art, related libraries | When the AI needs deeper background |
| `07_progress/` | Roadmap, completed milestones, current status | Every session (keep current) |

## Usage Pattern

For a typical AI-assisted development session, compose context from:

1. **Always include:** `04_conventions/` + `07_progress/status.md`
2. **Task-dependent:** relevant spec from `02_specifications/` + matching prompt from `05_prompts/`
3. **As needed:** domain docs from `01_domain/` or architecture from `03_architecture/`

## Principles

- **Specificity over volume** — a focused 500-line context outperforms a 5000-line dump
- **Layered disclosure** — start minimal, add context when the AI gets stuck
- **Living documents** — update after each session; stale context is worse than none
- **Separation of concerns** — domain math, code specs, and style rules are different documents
- **Verifiable** — specs should include test criteria the AI can check against
