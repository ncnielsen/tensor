use crate::tensor::Tensor;

/// Einstein tensor G_{μν}.
///
/// Defined as:
///
///   G_{μν} = R_{μν} − ½ g_{μν} R
///
/// Takes the Ricci tensor R_{μν} (Tensor<0,2>), the covariant metric g_{μν}
/// (Tensor<0,2>), and the Ricci scalar R (Tensor<0,0>), and returns a
/// Tensor<0,2>.
///
/// The Einstein tensor is the left-hand side of the Einstein field equations:
///
///   G_{μν} = (8πG / c⁴) T_{μν}
///
/// Its vanishing covariant divergence (∇^μ G_{μν} = 0) encodes the
/// conservation of energy and momentum.
pub fn einstein_tensor(
    ric: &Tensor<0, 2>,
    g: &Tensor<0, 2>,
    scalar: &Tensor<0, 0>,
) -> Tensor<0, 2> {
    assert_eq!(
        ric.dim, g.dim,
        "Dimension mismatch: ric ({}) vs g ({})",
        ric.dim, g.dim
    );

    let dim = ric.dim;
    let r = scalar.components[0]; // Ricci scalar (Number, Copy)

    let components = ric
        .components
        .iter()
        .zip(g.components.iter())
        .map(|(&r_mn, &g_mn)| r_mn - g_mn * r * 0.5)
        .collect();

    Tensor::new(dim, components)
}
