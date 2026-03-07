use crate::christoffel::Christoffel;
use crate::christoffel_derivative::ChristoffelDerivative;
use crate::tensor::{decode_flat_index, Tensor};

/// Numerical partial derivatives of a tensor field, using central differences.
///
/// Given a smooth function f: ℝ^D → Tensor<M,N> and a point x ∈ ℝ^D,
/// returns Tensor<M, N+1> whose component at [i1..iM, j1..jN, μ] is:
///
///   (∂_μ T)_{i1..iM j1..jN}(x) ≈ (f(x + h·eμ) − f(x − h·eμ)) / (2h)
///
/// The new covariant index μ is appended last, matching the convention used
/// by `Christoffel::from_metric` (`partial_g` layout [i,j,k] = ∂_k g_{ij})
/// and `covariant_derivative`.
///
/// `D = point.len()` sets the number of derivative directions and the `dim`
/// of the output tensor. Panics if `f` returns a tensor whose `dim` differs
/// from D.
///
/// Central differences are second-order accurate: the truncation error is
/// O(h²) for smooth f.  A value h ≈ 1e-5 gives good accuracy for most
/// double-precision computations.
pub fn partial_deriv<const M: usize, const N: usize>(
    f: &dyn Fn(&[f64]) -> Tensor<M, N>,
    point: &[f64],
    h: f64,
) -> Tensor<M, { N + 1 }>
where
    [(); N + 1]:,
    [(); M + { N + 1 }]:,
{
    let dim = point.len();
    assert!(dim >= 1, "Point must have at least one coordinate");

    let rank_in = M + N;
    let rank_out = rank_in + 1;
    let two_h_inv = 1.0 / (2.0 * h);

    // Evaluate f at x ± h·eμ once per direction (2·dim calls total).
    let perturbed: Vec<(Tensor<M, N>, Tensor<M, N>)> = (0..dim)
        .map(|mu| {
            let mut xp = point.to_vec();
            xp[mu] += h;
            let mut xm = point.to_vec();
            xm[mu] -= h;
            let fp = f(&xp);
            let fm = f(&xm);
            assert_eq!(
                fp.dim, dim,
                "f must return tensors with dim == point.len() (got {} vs {})",
                fp.dim, dim
            );
            (fp, fm)
        })
        .collect();

    let n_out = dim.pow(rank_out as u32);
    let components = (0..n_out)
        .map(|flat_out| {
            let out_idx = decode_flat_index(flat_out, dim, rank_out);
            let mu = out_idx[rank_out - 1]; // derivative direction is last index
            let tensor_idx = &out_idx[..rank_in];
            let (fp, fm) = &perturbed[mu];
            (fp.component(tensor_idx) - fm.component(tensor_idx)) * two_h_inv
        })
        .collect();

    Tensor::new(dim, components)
}

/// Numerical partial derivatives of Christoffel symbols, using central differences.
///
/// Given a function Γ: ℝ^D → Christoffel and a point x ∈ ℝ^D, returns a
/// `ChristoffelDerivative` whose component at (ρ, κ, μ, ν) is:
///
///   ∂_ν Γ^ρ_{κμ}(x) ≈ (Γ^ρ_{κμ}(x + h·eν) − Γ^ρ_{κμ}(x − h·eν)) / (2h)
///
/// The lower-index symmetry ∂_ν Γ^ρ_{κμ} = ∂_ν Γ^ρ_{μκ} is preserved
/// exactly because Christoffel symbols themselves are symmetric in their
/// lower indices.
///
/// `D = point.len()` determines the dimension.
pub fn christoffel_partial_deriv(
    f: &dyn Fn(&[f64]) -> Christoffel,
    point: &[f64],
    h: f64,
) -> ChristoffelDerivative {
    let dim = point.len();
    assert!(dim >= 1, "Point must have at least one coordinate");

    let two_h_inv = 1.0 / (2.0 * h);

    // Evaluate Γ at x ± h·eν once per direction (2·dim calls total).
    let perturbed: Vec<(Christoffel, Christoffel)> = (0..dim)
        .map(|nu| {
            let mut xp = point.to_vec();
            xp[nu] += h;
            let mut xm = point.to_vec();
            xm[nu] -= h;
            (f(&xp), f(&xm))
        })
        .collect();

    // Layout [ρ, κ, μ, ν] = ∂_ν Γ^ρ_{κμ}.
    let mut components = Vec::with_capacity(dim.pow(4));
    for rho in 0..dim {
        for kappa in 0..dim {
            for mu in 0..dim {
                for nu in 0..dim {
                    let (gp, gm) = &perturbed[nu];
                    let val = (gp.component(rho, kappa, mu)
                        - gm.component(rho, kappa, mu))
                        * two_h_inv;
                    components.push(val);
                }
            }
        }
    }

    ChristoffelDerivative::new(dim, components)
}
