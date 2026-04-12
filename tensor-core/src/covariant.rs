use crate::christoffel::Christoffel;
use crate::tensor::Tensor;

/// Covariant derivative of a rank-(M,N) tensor field.
///
/// ∇_k T^{i₁…iₘ}_{j₁…jₙ} = ∂_k T^{i₁…iₘ}_{j₁…jₙ}
///     + Σ_p Γ^{iₚ}_{kl} T^{i₁…l…iₘ}_{j₁…jₙ}   (+Γ per upper index)
///     - Σ_q Γ^l_{k jq} T^{i₁…iₘ}_{j₁…l…jₙ}     (-Γ per lower index)
///
/// - `tensor`: field value T at the point
/// - `partial_deriv`: ∂_k T with k appended as last lower index (rank (M, N+1))
/// - `christoffel`: Γ^i_{jk} at the point
///
/// Output type specified by caller (Rust can't express N+1 in const generics):
/// ```ignore
/// let nabla_v: Tensor<1, 1> = covariant_derivative(&v, &dv, &gamma);
/// ```
pub fn covariant_derivative<
    const M: usize, const N: usize,
    const MO: usize, const NO: usize,
>(
    tensor: &Tensor<M, N>,
    partial_deriv: &Tensor<MO, NO>,
    christoffel: &Christoffel,
) -> Tensor<MO, NO> {
    assert_eq!(MO, M, "output upper rank must equal input: MO={} != M={}", MO, M);
    assert_eq!(NO, N + 1, "output lower rank must be N+1: NO={} != N+1={}", NO, N + 1);
    assert_eq!(tensor.dim(), partial_deriv.dim(), "dimension mismatch: tensor vs partial_deriv");
    assert_eq!(tensor.dim(), christoffel.dim(), "dimension mismatch: tensor vs christoffel");

    let dim = tensor.dim();
    let mut result = Tensor::<MO, NO>::new(dim);

    for flat in 0..result.len() {
        let idx = result.decode_flat_index(flat);
        // idx = [u_0, ..., u_{M-1}, l_0, ..., l_{N-1}, k]
        let k = idx[M + N];

        let mut val = partial_deriv.component(&idx);

        // +Γ correction for each upper index
        for p in 0..M {
            let u_p = idx[p];
            for l in 0..dim {
                let mut t_idx = Vec::with_capacity(M + N);
                for i in 0..M {
                    t_idx.push(if i == p { l } else { idx[i] });
                }
                for j in 0..N {
                    t_idx.push(idx[M + j]);
                }
                val += christoffel.component(u_p, k, l) * tensor.component(&t_idx);
            }
        }

        // -Γ correction for each original lower index
        for q in 0..N {
            let l_q = idx[M + q];
            for l in 0..dim {
                let mut t_idx = Vec::with_capacity(M + N);
                for i in 0..M {
                    t_idx.push(idx[i]);
                }
                for j in 0..N {
                    t_idx.push(if j == q { l } else { idx[M + j] });
                }
                val -= christoffel.component(l, k, l_q) * tensor.component(&t_idx);
            }
        }

        result.as_mut_slice()[flat] = val;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deriv::{jacobian_from_reverse, metric_partials_from_jacobian};
    use crate::metric::invert_metric;
    use core::autodiff::autodiff_reverse;

    const TOL: f64 = 1e-10;

    // -- Scalar gradient = partial derivative ------------------------------------

    #[test]
    fn scalar_gradient_equals_partial() {
        let dim = 4;
        let phi = Tensor::<0, 0>::from_vec(dim, vec![42.0]);
        let d_phi = Tensor::<0, 1>::from_vec(dim, vec![1.0, 2.0, 3.0, 4.0]);

        // Non-zero Christoffel — shouldn't affect scalar
        let mut gamma = Christoffel::new(dim);
        gamma.set_component(0, 1, 2, 5.0);
        gamma.set_component(1, 0, 0, 3.0);

        let nabla_phi: Tensor<0, 1> = covariant_derivative(&phi, &d_phi, &gamma);

        for k in 0..dim {
            assert!(
                (nabla_phi.component(&[k]) - d_phi.component(&[k])).abs() < TOL,
                "∇_{} φ = {}, expected {}",
                k, nabla_phi.component(&[k]), d_phi.component(&[k])
            );
        }
    }

    // -- Vector in flat space = partial derivative -------------------------------

    #[test]
    fn vector_flat_space_equals_partial() {
        let dim = 3;
        let v = Tensor::<1, 0>::from_vec(dim, vec![1.0, 2.0, 3.0]);

        // ∂_k V^i as Tensor<1,1> with indices [i, k]
        let mut dv = Tensor::<1, 1>::new(dim);
        for i in 0..dim {
            for k in 0..dim {
                dv.set_component(&[i, k], (i * dim + k + 1) as f64 * 0.5);
            }
        }

        let gamma = Christoffel::new(dim); // all zero in flat space

        let nabla_v: Tensor<1, 1> = covariant_derivative(&v, &dv, &gamma);

        for i in 0..dim {
            for k in 0..dim {
                assert!(
                    (nabla_v.component(&[i, k]) - dv.component(&[i, k])).abs() < TOL,
                    "∇_{} V^{} = {}, expected {}",
                    k, i, nabla_v.component(&[i, k]), dv.component(&[i, k])
                );
            }
        }
    }

    // -- Metric compatibility: ∇g = 0 ------------------------------------------

    #[test]
    fn metric_compatibility_schwarzschild() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let cos_th = theta.cos();

        // Schwarzschild metric
        let mut g = Tensor::<0, 2>::new(4);
        g.set_component(&[0, 0], -f);
        g.set_component(&[1, 1], 1.0 / f);
        g.set_component(&[2, 2], r * r);
        g.set_component(&[3, 3], r * r * sin_th * sin_th);

        let g_inv = invert_metric(&g);

        // Analytic partials ∂_k g_{μν}
        let mut partials: Vec<Tensor<0, 2>> =
            (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();
        partials[1].set_component(&[0, 0], -2.0 / (r * r));
        partials[1].set_component(&[1, 1], -2.0 / (r * r * f * f));
        partials[1].set_component(&[2, 2], 2.0 * r);
        partials[1].set_component(&[3, 3], 2.0 * r * sin_th * sin_th);
        partials[2].set_component(&[3, 3], 2.0 * r * r * sin_th * cos_th);

        let gamma = Christoffel::from_metric(&g_inv, &partials);

        // Build Tensor<0,3>: component(&[μ, ν, k]) = ∂_k g_{μν}
        let mut dg = Tensor::<0, 3>::new(4);
        for k in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    dg.set_component(&[mu, nu, k], partials[k].component(&[mu, nu]));
                }
            }
        }

        let nabla_g: Tensor<0, 3> = covariant_derivative(&g, &dg, &gamma);

        for mu in 0..4 {
            for nu in 0..4 {
                for k in 0..4 {
                    assert!(
                        nabla_g.component(&[mu, nu, k]).abs() < TOL,
                        "∇_{} g_{}{} = {} ≠ 0",
                        k, mu, nu, nabla_g.component(&[mu, nu, k])
                    );
                }
            }
        }
    }

    // -- Enzyme integration: full pipeline → ∇g = 0 -----------------------------

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
    fn metric_compatibility_via_enzyme() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];

        // Evaluate metric
        let mut g_flat = [0.0f64; 16];
        schwarzschild_metric(&coords, &mut g_flat);
        let g = Tensor::<0, 2>::from_vec(4, g_flat.to_vec());
        let g_inv = invert_metric(&g);

        // Enzyme derivatives → metric partials
        let jac = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0f64; 4];
            let mut out = [0.0f64; 16];
            let mut dout = [0.0f64; 16];
            dout[i] = 1.0;
            d_schwarzschild(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });
        let partials = metric_partials_from_jacobian(&jac, 4);

        // Christoffel from Enzyme partials
        let gamma = Christoffel::from_metric(&g_inv, &partials);

        // Build partial_deriv Tensor<0,3>
        let mut dg = Tensor::<0, 3>::new(4);
        for k in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    dg.set_component(&[mu, nu, k], partials[k].component(&[mu, nu]));
                }
            }
        }

        // ∇g = 0
        let nabla_g: Tensor<0, 3> = covariant_derivative(&g, &dg, &gamma);

        for mu in 0..4 {
            for nu in 0..4 {
                for k in 0..4 {
                    assert!(
                        nabla_g.component(&[mu, nu, k]).abs() < TOL,
                        "Enzyme ∇_{} g_{}{} = {} ≠ 0",
                        k, mu, nu, nabla_g.component(&[mu, nu, k])
                    );
                }
            }
        }
    }
}
