use crate::christoffel::Christoffel;
use crate::tensor::Tensor;

/// Partial derivatives of Christoffel symbols: ∂_l Γ^i_{jk}.
///
/// Not a tensor — transforms inhomogeneously. Stores `dim^4` components.
/// Layout: `data[i * dim³ + j * dim² + k * dim + l]`.
///
/// `component(i, j, k, l)` returns ∂_l Γ^i_{jk}.
#[derive(Debug, Clone, PartialEq)]
pub struct ChristoffelDerivative {
    dim: usize,
    data: Vec<f64>,
}

impl ChristoffelDerivative {
    /// Zero derivatives.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; dim.pow(4)],
        }
    }

    /// Create from a flat data vector (`dim^4` components).
    pub fn from_flat(dim: usize, data: Vec<f64>) -> Self {
        let expected = dim.pow(4);
        assert_eq!(
            data.len(),
            expected,
            "expected {} components for dim={}, got {}",
            expected,
            dim,
            data.len()
        );
        Self { dim, data }
    }

    /// Compute via central finite differences.
    ///
    /// `christoffel_at(coords)` returns the Christoffel symbols at the given
    /// coordinate point. Step size `h` controls FD accuracy (O(h²)).
    pub fn from_fd(
        christoffel_at: impl Fn(&[f64]) -> Christoffel,
        coords: &[f64],
        h: f64,
    ) -> Self {
        let dim = coords.len();
        let d3 = dim * dim * dim;
        let d2 = dim * dim;
        let mut data = vec![0.0; dim.pow(4)];

        let mut xp = coords.to_vec();
        let mut xm = coords.to_vec();

        for l in 0..dim {
            xp[l] = coords[l] + h;
            xm[l] = coords[l] - h;

            let gamma_p = christoffel_at(&xp);
            let gamma_m = christoffel_at(&xm);

            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        data[i * d3 + j * d2 + k * dim + l] =
                            (gamma_p.component(i, j, k) - gamma_m.component(i, j, k))
                                / (2.0 * h);
                    }
                }
            }

            xp[l] = coords[l];
            xm[l] = coords[l];
        }

        Self { dim, data }
    }

    /// Compute analytically from the metric and its first/second partials.
    ///
    /// Uses the chain rule on Γ^i_{jk} = ½ g^{il}(∂_j g_{lk} + ∂_k g_{lj} − ∂_l g_{jk}):
    ///
    /// ```text
    /// ∂_m Γ^i_{jk} = ½ (∂_m g^{il})(∂_j g_{lk} + ∂_k g_{lj} − ∂_l g_{jk})
    ///              + ½ g^{il} (∂_m∂_j g_{lk} + ∂_m∂_k g_{lj} − ∂_m∂_l g_{jk})
    /// ```
    ///
    /// where ∂_m g^{il} = −g^{ia} g^{lb} ∂_m g_{ab} (inverse-metric derivative
    /// identity, from differentiating g^{il} g_{lm} = δ^i_m).
    ///
    /// This replaces the FD-of-FD approach (`from_fd`) with a single-layer
    /// derivative: the only discretisation error is in `partial_g` and
    /// `partial2_g` themselves. With 4th-order stencils for both, ∂Γ is
    /// 4th-order accurate (vs 2nd-order from `from_fd` with 2nd-order inputs).
    ///
    /// - `g_inv`: g^{ij} at the point
    /// - `partial_g`: `partial_g[m]` = ∂_m g_{ab} (length `dim`)
    /// - `partial2_g`: `partial2_g[m * dim + n]` = ∂_m ∂_n g_{ab} (length `dim²`,
    ///   symmetric in m,n but stored full)
    pub fn from_metric_analytic(
        g_inv: &Tensor<2, 0>,
        partial_g: &[Tensor<0, 2>],
        partial2_g: &[Tensor<0, 2>],
    ) -> Self {
        let dim = g_inv.dim();
        assert_eq!(partial_g.len(), dim, "partial_g must have dim entries");
        assert_eq!(partial2_g.len(), dim * dim, "partial2_g must have dim² entries");

        let d3 = dim * dim * dim;
        let d2 = dim * dim;
        let mut data = vec![0.0; dim.pow(4)];

        let p2 = |m: usize, n: usize, a: usize, b: usize| -> f64 {
            partial2_g[m * dim + n].component(&[a, b])
        };

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for m in 0..dim {
                        let mut val = 0.0;
                        for l in 0..dim {
                            // ∂_m g^{il} = −g^{ia} g^{lb} ∂_m g_{ab}
                            let mut d_ginv_il = 0.0;
                            for a in 0..dim {
                                for b in 0..dim {
                                    d_ginv_il -= g_inv.component(&[i, a])
                                        * g_inv.component(&[l, b])
                                        * partial_g[m].component(&[a, b]);
                                }
                            }

                            let bracket1 = partial_g[j].component(&[l, k])
                                + partial_g[k].component(&[l, j])
                                - partial_g[l].component(&[j, k]);
                            let term1 = 0.5 * d_ginv_il * bracket1;

                            let bracket2 = p2(m, j, l, k)
                                + p2(m, k, l, j)
                                - p2(m, l, j, k);
                            let term2 = 0.5 * g_inv.component(&[i, l]) * bracket2;

                            val += term1 + term2;
                        }
                        data[i * d3 + j * d2 + k * dim + m] = val;
                    }
                }
            }
        }

        Self { dim, data }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get ∂_l Γ^i_{jk}.
    pub fn component(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let d = self.dim;
        self.data[i * d * d * d + j * d * d + k * d + l]
    }

    /// Set ∂_l Γ^i_{jk}.
    pub fn set_component(&mut self, i: usize, j: usize, k: usize, l: usize, value: f64) {
        let d = self.dim;
        self.data[i * d * d * d + j * d * d + k * d + l] = value;
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Riemann curvature tensor R^ρ_{σμν} as a `Tensor<1,3>`.
///
/// R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ}
///           + Σ_λ (Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ})
pub fn riemann(gamma: &Christoffel, dgamma: &ChristoffelDerivative) -> Tensor<1, 3> {
    let dim = gamma.dim();
    assert_eq!(dim, dgamma.dim(), "dimension mismatch");
    let mut r = Tensor::<1, 3>::new(dim);

    for rho in 0..dim {
        for sigma in 0..dim {
            for mu in 0..dim {
                for nu in 0..dim {
                    let mut val = dgamma.component(rho, nu, sigma, mu)
                        - dgamma.component(rho, mu, sigma, nu);

                    for lam in 0..dim {
                        val += gamma.component(rho, mu, lam) * gamma.component(lam, nu, sigma)
                            - gamma.component(rho, nu, lam) * gamma.component(lam, mu, sigma);
                    }

                    r.set_component(&[rho, sigma, mu, nu], val);
                }
            }
        }
    }

    r
}

/// Ricci tensor R_{μν} = R^λ_{μλν} (contraction of Riemann).
pub fn ricci_tensor(riem: &Tensor<1, 3>) -> Tensor<0, 2> {
    let dim = riem.dim();
    let mut ric = Tensor::<0, 2>::new(dim);

    for mu in 0..dim {
        for nu in 0..dim {
            let mut val = 0.0;
            for lam in 0..dim {
                val += riem.component(&[lam, mu, lam, nu]);
            }
            ric.set_component(&[mu, nu], val);
        }
    }

    ric
}

/// Ricci scalar R = g^{μν} R_{μν}.
pub fn ricci_scalar(ricci: &Tensor<0, 2>, g_inv: &Tensor<2, 0>) -> f64 {
    let dim = ricci.dim();
    assert_eq!(dim, g_inv.dim(), "dimension mismatch");

    let mut scalar = 0.0;
    for mu in 0..dim {
        for nu in 0..dim {
            scalar += g_inv.component(&[mu, nu]) * ricci.component(&[mu, nu]);
        }
    }
    scalar
}

/// Einstein tensor G_{μν} = R_{μν} − (1/2) R g_{μν}.
pub fn einstein_tensor(ricci: &Tensor<0, 2>, scalar: f64, g: &Tensor<0, 2>) -> Tensor<0, 2> {
    assert_eq!(ricci.dim(), g.dim(), "dimension mismatch");
    let half_rg = g * (0.5 * scalar);
    ricci - &half_rg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deriv::{jacobian_from_reverse, metric_partials_from_jacobian};
    use crate::metric::invert_metric;
    use core::autodiff::autodiff_reverse;

    const TOL: f64 = 1e-10;
    const FD_TOL: f64 = 1e-6;

    // -----------------------------------------------------------------------
    // Enzyme-annotated metric functions
    // -----------------------------------------------------------------------

    #[autodiff_reverse(d_schwarzschild, Duplicated, Duplicated)]
    fn schwarzschild_metric(coords: &[f64; 4], out: &mut [f64; 16]) {
        let r = coords[1];
        let theta = coords[2];
        let f = 1.0 - 2.0 / r;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        out[0] = -f;
        out[5] = 1.0 / f;
        out[10] = r * r;
        out[15] = r * r * theta.sin() * theta.sin();
    }

    #[autodiff_reverse(d_flrw, Duplicated, Duplicated)]
    fn flrw_metric(coords: &[f64; 4], out: &mut [f64; 16]) {
        let t = coords[0];
        let big_h = 0.1_f64;
        let a = (big_h * t).exp();
        let a2 = a * a;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        out[0] = -1.0;
        out[5] = a2;
        out[10] = a2;
        out[15] = a2;
    }

    // -----------------------------------------------------------------------
    // Helpers: metric → (g, g_inv, Christoffel) via Enzyme
    // -----------------------------------------------------------------------

    fn schwarzschild_pipeline(coords: &[f64; 4]) -> (Tensor<0, 2>, Tensor<2, 0>, Christoffel) {
        let mut g_flat = [0.0f64; 16];
        schwarzschild_metric(coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(4, g_flat.to_vec());
        let g_inv = invert_metric(&g);

        let jac = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 16];
            let mut dout = [0.0f64; 16];
            dout[i] = 1.0;
            d_schwarzschild(coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let partials = metric_partials_from_jacobian(&jac, 4);
        let gamma = Christoffel::from_metric(&g_inv, &partials);
        (g, g_inv, gamma)
    }

    fn flrw_pipeline(coords: &[f64; 4]) -> (Tensor<0, 2>, Tensor<2, 0>, Christoffel) {
        let mut g_flat = [0.0f64; 16];
        flrw_metric(coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(4, g_flat.to_vec());
        let g_inv = invert_metric(&g);

        let jac = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 16];
            let mut dout = [0.0f64; 16];
            dout[i] = 1.0;
            d_flrw(coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let partials = metric_partials_from_jacobian(&jac, 4);
        let gamma = Christoffel::from_metric(&g_inv, &partials);
        (g, g_inv, gamma)
    }

    fn schwarzschild_christoffel_at(coords: &[f64]) -> Christoffel {
        let c = [coords[0], coords[1], coords[2], coords[3]];
        schwarzschild_pipeline(&c).2
    }

    fn flrw_christoffel_at(coords: &[f64]) -> Christoffel {
        let c = [coords[0], coords[1], coords[2], coords[3]];
        flrw_pipeline(&c).2
    }

    // -----------------------------------------------------------------------
    // Flat space → all curvature = 0
    // -----------------------------------------------------------------------

    #[test]
    fn flat_space_all_curvature_zero() {
        let dim = 4;
        let gamma = Christoffel::new(dim);
        let dgamma = ChristoffelDerivative::new(dim);

        let riem = riemann(&gamma, &dgamma);
        for &v in riem.as_slice() {
            assert!(v.abs() < TOL, "Riemann component {} ≠ 0", v);
        }

        let ric = ricci_tensor(&riem);
        for &v in ric.as_slice() {
            assert!(v.abs() < TOL, "Ricci component {} ≠ 0", v);
        }

        let mut g_inv = Tensor::<2, 0>::new(dim);
        g_inv.set_component(&[0, 0], -1.0);
        for i in 1..dim {
            g_inv.set_component(&[i, i], 1.0);
        }
        let scalar = ricci_scalar(&ric, &g_inv);
        assert!(scalar.abs() < TOL, "R = {} ≠ 0", scalar);

        let mut g = Tensor::<0, 2>::new(dim);
        g.set_component(&[0, 0], -1.0);
        for i in 1..dim {
            g.set_component(&[i, i], 1.0);
        }
        let ein = einstein_tensor(&ric, scalar, &g);
        for &v in ein.as_slice() {
            assert!(v.abs() < TOL, "Einstein component {} ≠ 0", v);
        }
    }

    // -----------------------------------------------------------------------
    // Schwarzschild: vacuum (Ricci = 0, Einstein = 0)
    // -----------------------------------------------------------------------

    #[test]
    fn schwarzschild_ricci_zero() {
        let c = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];
        let (_, g_inv, gamma) = schwarzschild_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &c, 1e-5);

        let riem = riemann(&gamma, &dgamma);
        let ric = ricci_tensor(&riem);

        for mu in 0..4 {
            for nu in 0..4 {
                assert!(
                    ric.component(&[mu, nu]).abs() < FD_TOL,
                    "R_{}{} = {}",
                    mu,
                    nu,
                    ric.component(&[mu, nu])
                );
            }
        }

        let scalar = ricci_scalar(&ric, &g_inv);
        assert!(scalar.abs() < FD_TOL, "R = {}", scalar);
    }

    // -----------------------------------------------------------------------
    // Schwarzschild: Riemann ≠ 0 (spacetime IS curved)
    // -----------------------------------------------------------------------

    #[test]
    fn schwarzschild_riemann_nonzero() {
        let c = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];
        let (_, _, gamma) = schwarzschild_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &c, 1e-5);
        let riem = riemann(&gamma, &dgamma);

        let max_abs = riem.as_slice().iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_abs > 1e-6,
            "Riemann appears zero — Schwarzschild should be curved"
        );
    }

    // -----------------------------------------------------------------------
    // Riemann antisymmetry: R^ρ_{σμν} = −R^ρ_{σνμ}
    // -----------------------------------------------------------------------

    #[test]
    fn schwarzschild_riemann_antisymmetry() {
        let c = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];
        let (_, _, gamma) = schwarzschild_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &c, 1e-5);
        let riem = riemann(&gamma, &dgamma);

        for rho in 0..4 {
            for sigma in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        let sum = riem.component(&[rho, sigma, mu, nu])
                            + riem.component(&[rho, sigma, nu, mu]);
                        assert!(
                            sum.abs() < FD_TOL,
                            "R^{}_{}{}{}  + R^{}_{}{}{}  = {}",
                            rho,
                            sigma,
                            mu,
                            nu,
                            rho,
                            sigma,
                            nu,
                            mu,
                            sum
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // First Bianchi identity: R^ρ_{σμν} + R^ρ_{μνσ} + R^ρ_{νσμ} = 0
    // -----------------------------------------------------------------------

    #[test]
    fn schwarzschild_first_bianchi() {
        let c = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];
        let (_, _, gamma) = schwarzschild_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &c, 1e-5);
        let riem = riemann(&gamma, &dgamma);

        for rho in 0..4 {
            for sigma in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        let sum = riem.component(&[rho, sigma, mu, nu])
                            + riem.component(&[rho, mu, nu, sigma])
                            + riem.component(&[rho, nu, sigma, mu]);
                        assert!(
                            sum.abs() < FD_TOL,
                            "First Bianchi: ρ={} σ={} μ={} ν={}, sum = {}",
                            rho,
                            sigma,
                            mu,
                            nu,
                            sum
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Einstein tensor symmetry: G_{μν} = G_{νμ}
    // -----------------------------------------------------------------------

    #[test]
    fn einstein_symmetry() {
        let c = [0.0, 10.0, std::f64::consts::FRAC_PI_4, 0.0];
        let (g, g_inv, gamma) = schwarzschild_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &c, 1e-5);
        let riem = riemann(&gamma, &dgamma);
        let ric = ricci_tensor(&riem);
        let scalar = ricci_scalar(&ric, &g_inv);
        let ein = einstein_tensor(&ric, scalar, &g);

        for mu in 0..4 {
            for nu in (mu + 1)..4 {
                let diff = (ein.component(&[mu, nu]) - ein.component(&[nu, mu])).abs();
                assert!(
                    diff < FD_TOL,
                    "G_{}{} − G_{}{} = {}",
                    mu,
                    nu,
                    nu,
                    mu,
                    diff
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // FLRW de Sitter: Einstein tensor matches analytic G_{μν}
    //
    //   ds² = −dt² + a(t)²(dx² + dy² + dz²),  a = e^{Ht}
    //   G_{00} = 3H²,  G_{ii} = −3H² a²,  off-diagonal = 0
    // -----------------------------------------------------------------------

    #[test]
    fn flrw_einstein_analytic() {
        let big_h = 0.1_f64;
        let t = 5.0;
        let c = [t, 1.0, 2.0, 3.0];

        let (g, g_inv, gamma) = flrw_pipeline(&c);
        let dgamma = ChristoffelDerivative::from_fd(flrw_christoffel_at, &c, 1e-5);
        let riem = riemann(&gamma, &dgamma);
        let ric = ricci_tensor(&riem);
        let scalar = ricci_scalar(&ric, &g_inv);
        let ein = einstein_tensor(&ric, scalar, &g);

        let a2 = (big_h * t).exp().powi(2);
        let expected_g00 = 3.0 * big_h * big_h;
        let expected_gii = -3.0 * big_h * big_h * a2;

        assert!(
            (ein.component(&[0, 0]) - expected_g00).abs() < FD_TOL,
            "G_00: got {}, expected {}",
            ein.component(&[0, 0]),
            expected_g00
        );
        for i in 1..4 {
            assert!(
                (ein.component(&[i, i]) - expected_gii).abs() < FD_TOL,
                "G_{}{}: got {}, expected {}",
                i,
                i,
                ein.component(&[i, i]),
                expected_gii
            );
        }
        for mu in 0..4 {
            for nu in 0..4 {
                if mu != nu {
                    assert!(
                        ein.component(&[mu, nu]).abs() < FD_TOL,
                        "G_{}{} = {} ≠ 0",
                        mu,
                        nu,
                        ein.component(&[mu, nu])
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Contracted Bianchi identity: ∇^μ G_{μν} = 0
    //
    // Uses FLRW (non-zero G) with FD for ∂_k G_{μν} and the covariant
    // derivative formula to check the divergence vanishes.
    // -----------------------------------------------------------------------

    #[test]
    fn bianchi_identity_flrw() {
        let coords = [5.0_f64, 1.0, 2.0, 3.0];
        let dim = 4;
        let h_outer = 1e-4;

        // Full pipeline at center point
        let (_, g_inv, gamma) = flrw_pipeline(&coords);
        let dgamma = ChristoffelDerivative::from_fd(flrw_christoffel_at, &coords, 1e-6);
        let riem = riemann(&gamma, &dgamma);
        let ric = ricci_tensor(&riem);
        let scalar = ricci_scalar(&ric, &g_inv);

        // Need g at center for einstein_tensor — re-evaluate
        let mut g_flat = [0.0f64; 16];
        flrw_metric(&coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(4, g_flat.to_vec());
        let ein = einstein_tensor(&ric, scalar, &g);

        // ∂_k G_{μν} via FD of the full pipeline at neighboring points
        let mut partial_ein: Vec<Tensor<0, 2>> =
            (0..dim).map(|_| Tensor::<0, 2>::new(dim)).collect();

        for k in 0..dim {
            let mut cp = coords;
            let mut cm = coords;
            cp[k] += h_outer;
            cm[k] -= h_outer;

            let ein_p = {
                let (gp, gp_inv, gamma_p) = flrw_pipeline(&cp);
                let dg = ChristoffelDerivative::from_fd(flrw_christoffel_at, &cp, 1e-6);
                let ri = riemann(&gamma_p, &dg);
                let rc = ricci_tensor(&ri);
                let sc = ricci_scalar(&rc, &gp_inv);
                einstein_tensor(&rc, sc, &gp)
            };

            let ein_m = {
                let (gm, gm_inv, gamma_m) = flrw_pipeline(&cm);
                let dg = ChristoffelDerivative::from_fd(flrw_christoffel_at, &cm, 1e-6);
                let ri = riemann(&gamma_m, &dg);
                let rc = ricci_tensor(&ri);
                let sc = ricci_scalar(&rc, &gm_inv);
                einstein_tensor(&rc, sc, &gm)
            };

            for mu in 0..dim {
                for nu in 0..dim {
                    partial_ein[k].set_component(
                        &[mu, nu],
                        (ein_p.component(&[mu, nu]) - ein_m.component(&[mu, nu]))
                            / (2.0 * h_outer),
                    );
                }
            }
        }

        // ∇^μ G_{μν} = g^{μα} (∂_α G_{μν} − Γ^λ_{αμ} G_{λν} − Γ^λ_{αν} G_{μλ})
        for nu in 0..dim {
            let mut div = 0.0;
            for mu in 0..dim {
                for alpha in 0..dim {
                    let mut nabla = partial_ein[alpha].component(&[mu, nu]);
                    for lam in 0..dim {
                        nabla -= gamma.component(lam, alpha, mu) * ein.component(&[lam, nu]);
                        nabla -= gamma.component(lam, alpha, nu) * ein.component(&[mu, lam]);
                    }
                    div += g_inv.component(&[mu, alpha]) * nabla;
                }
            }
            assert!(
                div.abs() < 1e-4,
                "∇^μ G_μ{} = {} ≠ 0",
                nu,
                div
            );
        }
    }

    // -----------------------------------------------------------------------
    // from_metric_analytic: flat metric → ∂Γ = 0
    // -----------------------------------------------------------------------

    #[test]
    fn analytic_deriv_flat_metric_zero() {
        let dim = 4;
        let mut g_inv = Tensor::<2, 0>::new(dim);
        g_inv.set_component(&[0, 0], -1.0);
        for i in 1..dim {
            g_inv.set_component(&[i, i], 1.0);
        }

        let zero_g: Vec<Tensor<0, 2>> = (0..dim).map(|_| Tensor::<0, 2>::new(dim)).collect();
        let zero_g2: Vec<Tensor<0, 2>> =
            (0..dim * dim).map(|_| Tensor::<0, 2>::new(dim)).collect();

        let dgamma = ChristoffelDerivative::from_metric_analytic(&g_inv, &zero_g, &zero_g2);
        for &v in dgamma.as_slice() {
            assert!(v.abs() < TOL, "∂Γ = {} ≠ 0 for flat metric", v);
        }
    }

    // -----------------------------------------------------------------------
    // from_metric_analytic agrees with from_fd for Schwarzschild
    // -----------------------------------------------------------------------

    #[test]
    fn analytic_deriv_matches_fd_schwarzschild() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];
        let dim = 4;

        // g and g_inv via Enzyme-evaluated metric
        let mut g_flat = [0.0f64; 16];
        schwarzschild_metric(&coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(dim, g_flat.to_vec());
        let g_inv = invert_metric(&g);

        // ∂_m g_{ab} via Enzyme (exact)
        let jac = jacobian_from_reverse(dim, dim * dim, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 16];
            let mut dout = [0.0f64; 16];
            dout[i] = 1.0;
            d_schwarzschild(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let partial_g = metric_partials_from_jacobian(&jac, dim);

        // ∂_m ∂_n g_{ab} via central FD of the metric function.
        // Step h_fd must balance truncation O(h²) against roundoff O(ε/h²);
        // h ~ 1e-3 is near-optimal for f64 second differences.
        let h_fd = 1e-3_f64;
        let h_sq = h_fd * h_fd;
        let eval_metric = |c: &[f64; 4]| -> [f64; 16] {
            let mut out = [0.0f64; 16];
            schwarzschild_metric(c, &mut out);
            out
        };

        let mut partial2_g: Vec<Tensor<0, 2>> =
            (0..dim * dim).map(|_| Tensor::<0, 2>::new(dim)).collect();

        for m in 0..dim {
            for n in 0..dim {
                if m == n {
                    let mut cp = coords; cp[m] += h_fd;
                    let mut cm = coords; cm[m] -= h_fd;
                    let gp = eval_metric(&cp);
                    let g0 = eval_metric(&coords);
                    let gm = eval_metric(&cm);
                    for a in 0..dim {
                        for b in 0..dim {
                            let row = a * dim + b;
                            partial2_g[m * dim + n].set_component(
                                &[a, b],
                                (gp[row] - 2.0 * g0[row] + gm[row]) / h_sq,
                            );
                        }
                    }
                } else {
                    let mut cpp = coords; cpp[m] += h_fd; cpp[n] += h_fd;
                    let mut cpm = coords; cpm[m] += h_fd; cpm[n] -= h_fd;
                    let mut cmp = coords; cmp[m] -= h_fd; cmp[n] += h_fd;
                    let mut cmm = coords; cmm[m] -= h_fd; cmm[n] -= h_fd;

                    let gpp = eval_metric(&cpp);
                    let gpm = eval_metric(&cpm);
                    let gmp = eval_metric(&cmp);
                    let gmm = eval_metric(&cmm);

                    for a in 0..dim {
                        for b in 0..dim {
                            let row = a * dim + b;
                            partial2_g[m * dim + n].set_component(
                                &[a, b],
                                (gpp[row] - gpm[row] - gmp[row] + gmm[row]) / (4.0 * h_sq),
                            );
                        }
                    }
                }
            }
        }

        // Analytic ∂Γ
        let dgamma_analytic =
            ChristoffelDerivative::from_metric_analytic(&g_inv, &partial_g, &partial2_g);

        // FD ∂Γ (reference)
        let dgamma_fd = ChristoffelDerivative::from_fd(schwarzschild_christoffel_at, &coords, 1e-5);

        // Both methods use FD for the second-derivative layer, so expect
        // agreement at the ~1e-6 level (FD truncation + roundoff).
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let a = dgamma_analytic.component(i, j, k, l);
                        let f = dgamma_fd.component(i, j, k, l);
                        assert!(
                            (a - f).abs() < 1e-5,
                            "∂_{} Γ^{}_{}{}: analytic={}, fd={}, diff={}",
                            l, i, j, k, a, f, (a - f).abs()
                        );
                    }
                }
            }
        }

        // Ricci tensor should vanish (vacuum) using the analytic ∂Γ.
        let gamma = Christoffel::from_metric(&g_inv, &partial_g);
        let riem = riemann(&gamma, &dgamma_analytic);
        let ric = ricci_tensor(&riem);
        for mu in 0..dim {
            for nu in 0..dim {
                assert!(
                    ric.component(&[mu, nu]).abs() < FD_TOL,
                    "R_{}{} = {} ≠ 0 (analytic ∂Γ, Schwarzschild vacuum)",
                    mu, nu, ric.component(&[mu, nu])
                );
            }
        }
    }
}
