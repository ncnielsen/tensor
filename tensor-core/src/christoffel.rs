use crate::tensor::Tensor;

/// Christoffel symbols of the second kind: Gamma^i_{jk}.
///
/// Not a tensor — transforms inhomogeneously under coordinate changes.
/// Symmetric in the lower indices: Gamma^i_{jk} = Gamma^i_{kj}.
///
/// Storage: flat `Vec<f64>` of `dim^3` components.
/// Layout: `data[i * dim^2 + j * dim + k]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Christoffel {
    dim: usize,
    data: Vec<f64>,
}

impl Christoffel {
    /// Compute from metric, inverse metric, and partial derivatives.
    ///
    /// Gamma^i_{jk} = (1/2) g^{i lam} (d_j g_{lam k} + d_k g_{lam j} - d_lam g_{jk})
    ///
    /// - `g_inv`: contravariant metric g^{mu nu} (`Tensor<2,0>`)
    /// - `partial_g`: `partial_g[k]` = dg/dx^k (`Tensor<0,2>` per coordinate)
    pub fn from_metric(g_inv: &Tensor<2, 0>, partial_g: &[Tensor<0, 2>]) -> Self {
        let dim = g_inv.dim();
        assert_eq!(partial_g.len(), dim);

        let mut data = vec![0.0; dim * dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                for k in j..dim {
                    let mut val = 0.0;
                    for lam in 0..dim {
                        let term = partial_g[j].component(&[lam, k])
                            + partial_g[k].component(&[lam, j])
                            - partial_g[lam].component(&[j, k]);
                        val += g_inv.component(&[i, lam]) * term;
                    }
                    val *= 0.5;
                    let d2 = dim * dim;
                    data[i * d2 + j * dim + k] = val;
                    data[i * d2 + k * dim + j] = val;
                }
            }
        }

        Self { dim, data }
    }

    /// Create from a flat data vector (`dim^3` components).
    pub fn from_flat(dim: usize, data: Vec<f64>) -> Self {
        let expected = dim * dim * dim;
        assert_eq!(data.len(), expected,
            "expected {} components for dim={}, got {}", expected, dim, data.len());
        Self { dim, data }
    }

    /// Zero Christoffel symbols.
    pub fn new(dim: usize) -> Self {
        Self { dim, data: vec![0.0; dim * dim * dim] }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get Gamma^i_{jk}.
    pub fn component(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i * self.dim * self.dim + j * self.dim + k]
    }

    /// Set Gamma^i_{jk} (and Gamma^i_{kj} for symmetry).
    pub fn set_component(&mut self, i: usize, j: usize, k: usize, value: f64) {
        let d2 = self.dim * self.dim;
        self.data[i * d2 + j * self.dim + k] = value;
        self.data[i * d2 + k * self.dim + j] = value;
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deriv::{jacobian_from_reverse, metric_partials_from_jacobian};
    use crate::metric::invert_metric;
    use core::autodiff::autodiff_reverse;

    const TOL: f64 = 1e-10;

    // -- Flat metric -> all zero ---------------------------------------------

    #[test]
    fn flat_metric_christoffel_zero() {
        // Minkowski: constant metric, all partials zero
        let mut eta_inv = Tensor::<2, 0>::new(4);
        eta_inv.set_component(&[0, 0], -1.0);
        for i in 1..4 {
            eta_inv.set_component(&[i, i], 1.0);
        }

        let zero_partials: Vec<Tensor<0, 2>> =
            (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();

        let gamma = Christoffel::from_metric(&eta_inv, &zero_partials);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert!(gamma.component(i, j, k).abs() < TOL,
                        "Gamma^{}_{}{} = {} != 0", i, j, k, gamma.component(i, j, k));
                }
            }
        }
    }

    // -- Lower-index symmetry ------------------------------------------------

    #[test]
    fn symmetry_arbitrary_input() {
        // Build from_metric with random-looking partials, verify symmetry
        let dim = 3;
        let mut g_inv = Tensor::<2, 0>::new(dim);
        g_inv.set_component(&[0, 0], 1.0);
        g_inv.set_component(&[1, 1], 2.0);
        g_inv.set_component(&[2, 2], 3.0);
        g_inv.set_component(&[0, 1], 0.1);
        g_inv.set_component(&[1, 0], 0.1);

        let mut partials = Vec::new();
        for k in 0..dim {
            let mut dg = Tensor::<0, 2>::new(dim);
            for mu in 0..dim {
                for nu in 0..dim {
                    let val = ((k * 9 + mu * 3 + nu + 1) as f64) * 0.1;
                    dg.set_component(&[mu, nu], val);
                }
            }
            partials.push(dg);
        }

        let gamma = Christoffel::from_metric(&g_inv, &partials);

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    assert!((gamma.component(i, j, k) - gamma.component(i, k, j)).abs() < TOL,
                        "Gamma^{}_{}{} != Gamma^{}_{}{}", i, j, k, i, k, j);
                }
            }
        }
    }

    // -- Schwarzschild: analytic values --------------------------------------

    fn schwarzschild_christoffel_analytic(r: f64, theta: f64) -> Christoffel {
        // M=1 Schwarzschild in (t, r, θ, φ)
        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let cos_th = theta.cos();

        let mut gamma = Christoffel::new(4);

        // Gamma^t_{tr} = 1/(r^2 f)
        gamma.set_component(0, 0, 1, 1.0 / (r * r * f));
        // Gamma^r_{tt} = f / r^2
        gamma.set_component(1, 0, 0, f / (r * r));
        // Gamma^r_{rr} = -1/(r^2 f)
        gamma.set_component(1, 1, 1, -1.0 / (r * r * f));
        // Gamma^r_{θθ} = -r f
        gamma.set_component(1, 2, 2, -r * f);
        // Gamma^r_{φφ} = -r f sin^2(θ)
        gamma.set_component(1, 3, 3, -r * f * sin_th * sin_th);
        // Gamma^θ_{rθ} = 1/r
        gamma.set_component(2, 1, 2, 1.0 / r);
        // Gamma^θ_{φφ} = -sin(θ) cos(θ)
        gamma.set_component(2, 3, 3, -sin_th * cos_th);
        // Gamma^φ_{rφ} = 1/r
        gamma.set_component(3, 1, 3, 1.0 / r);
        // Gamma^φ_{θφ} = cos(θ)/sin(θ)
        gamma.set_component(3, 2, 3, cos_th / sin_th);

        gamma
    }

    #[test]
    fn schwarzschild_christoffel_from_analytic_partials() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let cos_th = theta.cos();

        // Build metric and inverse
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r * sin_th * sin_th);
        let g_inv = invert_metric(&g);

        // Build analytic partials
        let mut partials: Vec<Tensor<0, 2>> =
            (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();

        // dg/dr (index 1)
        partials[1].set_component(&[0, 0], -2.0 / (r * r));
        partials[1].set_component(&[1, 1], -2.0 / (r * r * f * f));
        partials[1].set_component(&[2, 2], 2.0 * r);
        partials[1].set_component(&[3, 3], 2.0 * r * sin_th * sin_th);
        // dg/dθ (index 2)
        partials[2].set_component(&[3, 3], 2.0 * r * r * sin_th * cos_th);

        let gamma = Christoffel::from_metric(&g_inv, &partials);
        let expected = schwarzschild_christoffel_analytic(r, theta);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert!(
                        (gamma.component(i, j, k) - expected.component(i, j, k)).abs() < TOL,
                        "Gamma^{}_{}{}: got {}, expected {}",
                        i, j, k,
                        gamma.component(i, j, k),
                        expected.component(i, j, k)
                    );
                }
            }
        }
    }

    // -- Enzyme integration: metric -> partials -> Christoffel ---------------

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

    #[test]
    fn schwarzschild_christoffel_via_enzyme() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];

        // Step 1: evaluate metric
        let mut g_flat = [0.0f64; 16];
        schwarzschild_metric(&coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(4, g_flat.to_vec());
        let g_inv = invert_metric(&g);

        // Step 2: Enzyme derivatives -> metric partials
        let jac = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 16];
            let mut dout = [0.0f64; 16];
            dout[i] = 1.0;
            d_schwarzschild(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let partials = metric_partials_from_jacobian(&jac, 4);

        // Step 3: Christoffel from metric + Enzyme partials
        let gamma = Christoffel::from_metric(&g_inv, &partials);

        // Step 4: compare against known analytic values
        let expected = schwarzschild_christoffel_analytic(r, theta);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert!(
                        (gamma.component(i, j, k) - expected.component(i, j, k)).abs() < TOL,
                        "Enzyme Gamma^{}_{}{}: got {}, expected {}",
                        i, j, k,
                        gamma.component(i, j, k),
                        expected.component(i, j, k)
                    );
                }
            }
        }
    }

    // -- from_flat / set_component -------------------------------------------

    #[test]
    fn from_flat_and_access() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let gamma = Christoffel::from_flat(2, data);
        // dim=2: data[i*4 + j*2 + k]
        assert_eq!(gamma.component(0, 0, 0), 0.0);
        assert_eq!(gamma.component(0, 0, 1), 1.0);
        assert_eq!(gamma.component(0, 1, 0), 2.0);
        assert_eq!(gamma.component(1, 1, 1), 7.0);
    }

    #[test]
    fn set_component_symmetric() {
        let mut gamma = Christoffel::new(3);
        gamma.set_component(0, 1, 2, 5.0);
        assert_eq!(gamma.component(0, 1, 2), 5.0);
        assert_eq!(gamma.component(0, 2, 1), 5.0); // symmetry enforced
    }
}
