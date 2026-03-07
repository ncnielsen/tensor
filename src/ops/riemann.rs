use crate::christoffel::Christoffel;
use crate::christoffel_derivative::ChristoffelDerivative;
use crate::tensor::{decode_flat_index, Tensor};

/// Riemann curvature tensor R^ρ_{σμν}.
///
/// Given Christoffel symbols Γ and their partial derivatives ∂Γ, computes:
///
///   R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ}
///             + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}
///
/// Output is a Tensor<1,3> with layout [ρ, σ, μ, ν]:
///   - ρ       : upper (contravariant) index
///   - σ, μ, ν : lower (covariant) indices
///
/// A zero Riemann tensor indicates flat spacetime; non-zero components
/// encode intrinsic curvature.
pub fn riemann(gamma: &Christoffel, partial_gamma: &ChristoffelDerivative) -> Tensor<1, 3> {
    assert_eq!(
        gamma.dim, partial_gamma.dim,
        "Dimension mismatch: gamma ({}) vs partial_gamma ({})",
        gamma.dim, partial_gamma.dim
    );

    let dim = gamma.dim;
    let n_out = dim.pow(4);

    let components = (0..n_out)
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, 4);
            let rho   = out[0];
            let sigma = out[1];
            let mu    = out[2];
            let nu    = out[3];

            // ∂_μ Γ^ρ_{νσ}
            let term1 = partial_gamma.component(rho, nu, sigma, mu);

            // ∂_ν Γ^ρ_{μσ}
            let term2 = partial_gamma.component(rho, mu, sigma, nu);

            // Σ_λ Γ^ρ_{μλ} Γ^λ_{νσ}
            let mut iter3 = (0..dim).map(|lambda| {
                gamma.component(rho, mu, lambda) * gamma.component(lambda, nu, sigma)
            });
            let first3 = iter3.next().unwrap();
            let term3 = iter3.fold(first3, |acc, x| acc + x);

            // Σ_λ Γ^ρ_{νλ} Γ^λ_{μσ}
            let mut iter4 = (0..dim).map(|lambda| {
                gamma.component(rho, nu, lambda) * gamma.component(lambda, mu, sigma)
            });
            let first4 = iter4.next().unwrap();
            let term4 = iter4.fold(first4, |acc, x| acc + x);

            term1 - term2 + term3 - term4
        })
        .collect();

    Tensor::new(dim, components)
}
