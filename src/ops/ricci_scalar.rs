use crate::tensor::Tensor;

/// Ricci scalar R.
///
/// The full contraction of the inverse metric with the Ricci tensor:
///
///   R = g^{μν} R_{μν} = Σ_{μ,ν} g^{μν} R_{μν}
///
/// Takes the inverse metric g^{μν} (Tensor<2,0>) and the Ricci tensor R_{μν}
/// (Tensor<0,2>), and returns a scalar Tensor<0,0>.
///
/// The Ricci scalar is a single number encoding the average curvature at a
/// point in spacetime. It appears in the Einstein tensor and, through the
/// field equations, relates directly to the local energy density.
pub fn ricci_scalar(g_inv: &Tensor<2, 0>, ric: &Tensor<0, 2>) -> Tensor<0, 0> {
    assert_eq!(
        g_inv.dim, ric.dim,
        "Dimension mismatch: g_inv ({}) vs ric ({})",
        g_inv.dim, ric.dim
    );
    assert!(g_inv.dim >= 1, "Dimension must be at least 1");

    let dim = g_inv.dim;

    // R = Σ_{μν} g^{μν} R_{μν}
    let mut iter = (0..dim).flat_map(|mu| {
        (0..dim).map(move |nu| {
            g_inv.component(&[mu, nu]) * ric.component(&[mu, nu])
        })
    });
    let first = iter.next().unwrap();
    let sum = iter.fold(first, |acc, x| acc + x);

    Tensor::new(dim, vec![sum])
}
