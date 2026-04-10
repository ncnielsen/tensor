# References — Index

# References — Index

## AAD Library (`/home/ncn/git/automatic_adjoint_differentiation`)

Reverse-mode autodiff crate. Tensor library uses its `Number` as scalar type.

- [aad_library.md § Number](aad_library.md) — `Number` struct: fields (result, id, leaf), construction, math ops (ln, sin, cos, exp, pow, sqrt, log, cdf), mixed-type arithmetic (Number+f64, f64+Number)
- [aad_library.md § AutomaticDifferentiator](aad_library.md) — `derivatives(func, args) -> Evaluation`, backpropagation API, `Evaluation`/`Derivative` types
- [aad_library.md § no_tape](aad_library.md) — Disables tape in hot paths, thread-local flag, drop guard, critical mixed-type bug history (47x slowdown fix)
- [aad_library.md § Usage](aad_library.md) — Cargo dependency setup (`path = ".."`), hot-path vs gradient patterns, `#[serial]` thread safety caveat

## Spacetime Manipulation Article (`/home/ncn/git/tornado/template_Article.tex`)

Motivating paper for the project. Proposes artificial spacetime curvature via EM-source pumping.

- [spacetime_article.md](spacetime_article.md) — Title, summary (laser pump analogy, circular EM array, gravitational wave interaction), relation to tensor crate
