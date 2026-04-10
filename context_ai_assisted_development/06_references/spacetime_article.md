# Spacetime Manipulation Article

LaTeX source at `/home/ncn/git/tornado/template_Article.tex`.
Author: Niels Christian Nielsen.

## Title

"Artificial spacetime manipulation - or how to create a spacetime tornado"

## Summary

The article proposes a method for artificially manipulating spacetime curvature
using principles analogous to laser pumping. Key ideas:

1. **Gravity as geometry** — Einstein's GR: spacetime curves in response to
   energy; gravity is not a force but geometric curvature.

2. **The scale problem** — Newton's constant G is tiny (~6.674e-11), so no
   instantaneous energy source on Earth produces measurable curvature. Need a
   storage + pumping strategy instead.

3. **Laser pump analogy** — a laser stores photon energy between mirrors and
   pumps over time. A spacetime pump would store rotational gravitational
   energy and pump it via timed EM sources.

4. **Gravitational wave interaction** — spacetime behaves like a fluid;
   gravitational waves can constructively interfere. A circular array of
   electrically charged sources, software-timed, produces constructive wave
   patterns that build rotational spacetime curvature (a "spacetime tornado").

5. **Simulation approach** — simulate the EM sources + control software using
   Einstein's field equations and Maxwell's equations. If the pump works in
   simulation, move to a physical prototype.

## Relation to Tensor Crate

This article is the motivating document for the entire tensor/simulation
project. The simulation pipeline (metric -> Christoffel -> Riemann -> Ricci ->
Einstein tensor, coupled with Maxwell/EM stress-energy) directly implements the
physics described in the article. The `TornadoArray` and `EmSource` types in
the crate model the circular EM source array proposed here.
