use crate::tensor::{decode_flat_index, Tensor};

/// Faraday (electromagnetic field) tensor F_{μν}.
///
/// Defined from the 4-potential A_μ as:
///
///   F_{μν} = ∂_μ A_ν − ∂_ν A_μ
///
/// The input `partial_a` is a Tensor<0,2> holding the partial derivatives of
/// the 4-potential with the derivative index appended last (consistent with
/// the rest of the library):
///
///   partial_a.component(&[ν, μ]) = ∂_μ A_ν
///
/// The output is a Tensor<0,2> with layout [μ, ν].
///
/// F_{μν} is antisymmetric by construction — F_{μν} = −F_{νμ} — so the
/// diagonal components always vanish. In 4D Minkowski spacetime the
/// independent components encode the electric and magnetic fields:
///
///   F_{01} = E_x,  F_{02} = E_y,  F_{03} = E_z
///   F_{12} = B_z,  F_{31} = B_y,  F_{23} = B_x
///
/// Unlike the Christoffel symbols, F_{μν} is a genuine tensor: it transforms
/// covariantly under coordinate changes.
pub fn faraday(partial_a: &Tensor<0, 2>) -> Tensor<0, 2> {
    let dim = partial_a.dim;
    let n_out = dim.pow(2);

    let components = (0..n_out)
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, 2);
            let mu = out[0];
            let nu = out[1];

            // ∂_μ A_ν = partial_a[ν, μ]
            // ∂_ν A_μ = partial_a[μ, ν]
            partial_a.component(&[nu, mu]) - partial_a.component(&[mu, nu])
        })
        .collect();

    Tensor::new(dim, components)
}
