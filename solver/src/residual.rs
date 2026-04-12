use core::autodiff::autodiff_reverse;

use tensor_core::{
    christoffel::Christoffel,
    curvature::{einstein_tensor, ricci_scalar, ricci_tensor, riemann, ChristoffelDerivative},
    deriv::jacobian_from_reverse,
    tensor::Tensor,
};

// ---------------------------------------------------------------------------
// coords → Christoffel (Enzyme-differentiable)
// ---------------------------------------------------------------------------

/// Schwarzschild metric (M=1): coordinates (t, r, θ, φ) → Christoffel Γ^i_{jk}.
///
/// Layout: `out[i * 16 + j * 4 + k]` = Γ^i_{jk}.
///
/// The metric formula **and** its analytic partial derivatives are inlined as
/// plain `f64` arithmetic. `#[autodiff_reverse]` therefore gives Enzyme exact
/// ∂Γ^i_{jk}/∂coords^l through the compiled LLVM IR — no finite differences,
/// no step-size parameter.
#[autodiff_reverse(d_schwarzschild_christoffel, Duplicated, Duplicated)]
pub fn schwarzschild_christoffel(coords: &[f64; 4], out: &mut [f64; 64]) {
    let r = coords[1];
    let theta = coords[2];
    let f = 1.0 - 2.0 / r;
    let sin_th = theta.sin();
    let cos_th = theta.cos();

    // g^{μν} — inverse metric (diagonal for Schwarzschild).
    // Flat row-major [16]; off-diagonal entries remain 0.0.
    let mut g_inv = [0.0f64; 16];
    g_inv[0] = -1.0 / f;                          // g^{tt}
    g_inv[5] = f;                                  // g^{rr}
    g_inv[10] = 1.0 / (r * r);                    // g^{θθ}
    g_inv[15] = 1.0 / (r * r * sin_th * sin_th);  // g^{φφ}

    // ∂_k g_{μν} — flat layout: dg[k * 16 + mu * 4 + nu].
    // Only ∂_r (k=1) and ∂_θ (k=2) are non-zero for Schwarzschild.
    let mut dg = [0.0f64; 64];
    dg[1 * 16 + 0 * 4 + 0] = -2.0 / (r * r);                  // ∂_r g_{tt}
    dg[1 * 16 + 1 * 4 + 1] = -2.0 / (r * r * f * f);          // ∂_r g_{rr}
    dg[1 * 16 + 2 * 4 + 2] = 2.0 * r;                          // ∂_r g_{θθ}
    dg[1 * 16 + 3 * 4 + 3] = 2.0 * r * sin_th * sin_th;       // ∂_r g_{φφ}
    dg[2 * 16 + 3 * 4 + 3] = 2.0 * r * r * sin_th * cos_th;  // ∂_θ g_{φφ}

    // Zero out before accumulating.
    // Use slice iteration (Enzyme handles slice iterators; Range<usize> causes
    // insertvalue issues with Option<usize> internal state).
    for v in out.iter_mut() {
        *v = 0.0;
    }

    // Γ^i_{jk} = (1/2) g^{il} (∂_j g_{lk} + ∂_k g_{lj} − ∂_l g_{jk})
    // Use while loops — Enzyme cannot type Range<usize>::next (Option enum).
    let mut i = 0usize;
    while i < 4 {
        let mut j = 0usize;
        while j < 4 {
            let mut k = 0usize;
            while k < 4 {
                let mut val = 0.0f64;
                let mut l = 0usize;
                while l < 4 {
                    val += g_inv[i * 4 + l]
                        * (dg[j * 16 + l * 4 + k]
                         + dg[k * 16 + l * 4 + j]
                         - dg[l * 16 + j * 4 + k]);
                    l += 1;
                }
                out[i * 16 + j * 4 + k] = 0.5 * val;
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Utility: Enzyme Jacobian → ChristoffelDerivative
// ---------------------------------------------------------------------------

/// Build a `ChristoffelDerivative` from the Jacobian of a `coords → Γ_flat`
/// function produced by `jacobian_from_reverse`.
///
/// `jac` is row-major, shape (dim³, dim):
///   `jac[(i*dim²+j*dim+k) * dim + l]` = ∂_l Γ^i_{jk}
pub fn christoffel_deriv_from_jacobian(jac: &[f64], dim: usize) -> ChristoffelDerivative {
    let d2 = dim * dim;
    let mut dgamma = ChristoffelDerivative::new(dim);
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                let row = i * d2 + j * dim + k;
                for l in 0..dim {
                    dgamma.set_component(i, j, k, l, jac[row * dim + l]);
                }
            }
        }
    }
    dgamma
}

// ---------------------------------------------------------------------------
// Einstein residual
// ---------------------------------------------------------------------------

/// Compute G_{μν} − κ T_{μν} at a spacetime point.
///
/// - `christoffel_fn`: plain-f64 function `coords → Γ_flat` (metric inlined).
/// - `d_christoffel_fn`: Enzyme derivative of `christoffel_fn`.
/// - `g`, `g_inv`: metric and its inverse at `coords`.
/// - `kappa_t`: κ T_{μν} (zero for vacuum).
pub fn einstein_residual(
    coords: &[f64; 4],
    christoffel_fn: fn(&[f64; 4], &mut [f64; 64]),
    d_christoffel_fn: fn(&[f64; 4], &mut [f64; 4], &mut [f64; 64], &mut [f64; 64]),
    g: &Tensor<0, 2>,
    g_inv: &Tensor<2, 0>,
    kappa_t: &Tensor<0, 2>,
) -> Tensor<0, 2> {
    // Christoffel at the point
    let mut gamma_flat = [0.0f64; 64];
    christoffel_fn(coords, &mut gamma_flat);
    let gamma = Christoffel::from_flat(4, gamma_flat.to_vec());

    // Christoffel derivative via Enzyme — exact, no finite differences
    let jac = jacobian_from_reverse(4, 64, |i| {
        let mut dx = [0.0f64; 4];
        let mut out = [0.0f64; 64];
        let mut dout = [0.0f64; 64];
        dout[i] = 1.0;
        d_christoffel_fn(coords, &mut dx, &mut out, &mut dout);
        dx.to_vec()
    });
    let dgamma = christoffel_deriv_from_jacobian(&jac, 4);

    // G_{μν}
    let riem = riemann(&gamma, &dgamma);
    let ric = ricci_tensor(&riem);
    let scalar = ricci_scalar(&ric, g_inv);
    let g_ein = einstein_tensor(&ric, scalar, g);

    // G_{μν} − κ T_{μν}
    &g_ein - kappa_t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_core::metric::invert_metric;

    const TOL: f64 = 1e-10;
    const RESIDUAL_TOL: f64 = 1e-6;

    fn schwarzschild_metric_and_inverse(r: f64, theta: f64) -> (Tensor<0, 2>, Tensor<2, 0>) {
        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r * sin_th * sin_th);
        let g_inv = invert_metric(&g);
        (g, g_inv)
    }

    // -- Flat vacuum: G = 0 -------------------------------------------------

    #[test]
    fn flat_vacuum_zero_residual() {
        // Minkowski: Γ = 0, ∂Γ = 0 ⟹ Riemann = 0 ⟹ G = 0
        let dim = 4;
        let gamma = Christoffel::new(dim);
        let dgamma = ChristoffelDerivative::new(dim);

        let mut g = Tensor::<0, 2>::new(dim);
        g.set_component(&[0, 0], -1.0);
        for i in 1..dim {
            g.set_component(&[i, i], 1.0);
        }
        let mut g_inv = Tensor::<2, 0>::new(dim);
        g_inv.set_component(&[0, 0], -1.0);
        for i in 1..dim {
            g_inv.set_component(&[i, i], 1.0);
        }

        let riem = riemann(&gamma, &dgamma);
        let ric = ricci_tensor(&riem);
        let scalar = ricci_scalar(&ric, &g_inv);
        let ein = einstein_tensor(&ric, scalar, &g);

        for &v in ein.as_slice() {
            assert!(v.abs() < TOL, "G_{} = {} ≠ 0 for flat vacuum", v, v);
        }
    }

    // -- schwarzschild_christoffel matches tensor-core reference ------------

    #[test]
    fn schwarzschild_christoffel_matches_reference() {
        // Reference: compute Christoffel via analytic partials
        // (same approach used in tensor-core's christoffel tests)
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let (_, g_inv_ref) = schwarzschild_metric_and_inverse(r, theta);

        // Build analytic partials (same as in tensor-core tests)
        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let cos_th = theta.cos();
        let mut partials: Vec<Tensor<0, 2>> =
            (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();
        partials[1].set_component(&[0, 0], -2.0 / (r * r));
        partials[1].set_component(&[1, 1], -2.0 / (r * r * f * f));
        partials[1].set_component(&[2, 2], 2.0 * r);
        partials[1].set_component(&[3, 3], 2.0 * r * sin_th * sin_th);
        partials[2].set_component(&[3, 3], 2.0 * r * r * sin_th * cos_th);

        let gamma_ref = Christoffel::from_metric(&g_inv_ref, &partials);

        // Our function
        let coords = [0.0, r, theta, 0.0];
        let mut gamma_flat = [0.0f64; 64];
        schwarzschild_christoffel(&coords, &mut gamma_flat);

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let ours = gamma_flat[i * 16 + j * 4 + k];
                    let reference = gamma_ref.component(i, j, k);
                    assert!(
                        (ours - reference).abs() < TOL,
                        "Γ^{}_{}{}: ours={}, ref={}",
                        i, j, k, ours, reference
                    );
                }
            }
        }
    }

    // -- Enzyme gives exact ∂Γ vs FD reference ------------------------------

    #[test]
    fn enzyme_christoffel_deriv_vs_fd() {
        use tensor_core::curvature::ChristoffelDerivative as CD;

        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];

        // Enzyme-based derivative (exact)
        let jac = jacobian_from_reverse(4, 64, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 64];
            let mut dout = [0.0f64; 64];
            dout[i] = 1.0;
            d_schwarzschild_christoffel(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let dgamma_enzyme = christoffel_deriv_from_jacobian(&jac, 4);

        // FD reference
        let h = 1e-5_f64;
        let dgamma_fd = CD::from_fd(
            |x| {
                let c = [x[0], x[1], x[2], x[3]];
                let mut out = [0.0f64; 64];
                schwarzschild_christoffel(&c, &mut out);
                Christoffel::from_flat(4, out.to_vec())
            },
            &coords,
            h,
        );

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        let enzyme = dgamma_enzyme.component(i, j, k, l);
                        let fd = dgamma_fd.component(i, j, k, l);
                        let err = (enzyme - fd).abs();
                        // FD truncation error is O(h²) ≈ 1e-10; allow 1e-6
                        assert!(
                            err < 1e-6,
                            "∂_{} Γ^{}_{}{}: enzyme={}, fd={}, diff={}",
                            l, i, j, k, enzyme, fd, err
                        );
                    }
                }
            }
        }
    }

    // -- Schwarzschild vacuum: G ≈ 0 using Enzyme for ∂Γ -------------------

    #[test]
    fn schwarzschild_vacuum_residual_zero() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];
        let (g, g_inv) = schwarzschild_metric_and_inverse(r, theta);
        let zero_t = Tensor::<0, 2>::new(4);

        let residual = einstein_residual(
            &coords,
            schwarzschild_christoffel,
            d_schwarzschild_christoffel,
            &g,
            &g_inv,
            &zero_t,
        );

        for mu in 0..4 {
            for nu in 0..4 {
                assert!(
                    residual.component(&[mu, nu]).abs() < RESIDUAL_TOL,
                    "G_{}{} = {} ≠ 0 for Schwarzschild vacuum",
                    mu, nu, residual.component(&[mu, nu])
                );
            }
        }
    }
}
