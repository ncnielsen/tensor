use crate::christoffel::Christoffel;
use crate::tensor::{decode_flat_index, Tensor};

/// Covariant derivative of a tensor field.
///
/// Given a tensor T^{i1..iM}_{j1..jN} at a point, its partial derivatives
/// ∂_k T (as a Tensor<M, N+1>), and Christoffel symbols Γ^i_{jk} (as a
/// Tensor<1, 2>), returns the covariant derivative:
///
///   ∇_k T^{i1..iM}_{j1..jN} =
///       ∂_k T^{i1..iM}_{j1..jN}
///     + Σ_{p=0}^{M-1} Σ_l Γ^{ip}_{kl} T^{i1..(l at p)..iM}_{j1..jN}   [upper correction]
///     - Σ_{q=0}^{N-1} Σ_l Γ^l_{kjq}   T^{i1..iM}_{j1..(l at q)..jN}   [lower correction]
///
/// Index conventions:
///   - `partial_deriv` and the output use layout [i1..iM, j1..jN, k];
///     the derivative index k is appended as the last lower index.
///   - `christoffel.component(&[i, j, k])` = Γ^i_{jk}.
///
/// The output rank is (M, N+1): same contravariant rank, one additional
/// covariant index for the direction of differentiation.
pub fn covariant_derivative<const M: usize, const N: usize>(
    tensor: &Tensor<M, N>,
    partial_deriv: &Tensor<M, { N + 1 }>,
    christoffel: &Christoffel,
) -> Tensor<M, { N + 1 }>
where
    [(); N + 1]:,
    [(); M + { N + 1 }]:,
{
    assert_eq!(
        tensor.dim, partial_deriv.dim,
        "Dimension mismatch between tensor and partial_deriv: {} vs {}",
        tensor.dim, partial_deriv.dim
    );
    assert_eq!(
        tensor.dim, christoffel.dim,
        "Dimension mismatch between tensor and christoffel: {} vs {}",
        tensor.dim,
        christoffel.dim
    );

    let dim = tensor.dim;
    let rank_out = M + N + 1;
    let n_out = if rank_out == 0 { 1 } else { dim.pow(rank_out as u32) };

    let components = (0..n_out)
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, rank_out);
            // Index layout: [i1,...,iM, j1,...,jN, k]
            let upper = &out[..M];
            let lower = &out[M..M + N];
            let k = out[M + N];

            // Start with the partial derivative ∂_k T^{i1..iM}_{j1..jN}
            let mut result = partial_deriv.component(&out);

            // Upper index corrections: +Σ_l Γ^{ip}_{kl} T^{..l..}_{j1..jN}
            for p in 0..M {
                let ip = upper[p];
                let mut iter = (0..dim).map(|l| {
                    let gamma = christoffel.component(ip, k, l);
                    let mut t_idx: Vec<usize> = upper.to_vec();
                    t_idx[p] = l;
                    t_idx.extend_from_slice(lower);
                    gamma * tensor.component(&t_idx)
                });
                let first = iter.next().unwrap();
                let correction = iter.fold(first, |acc, x| acc + x);
                result = result + correction;
            }

            // Lower index corrections: -Σ_l Γ^l_{kjq} T^{i1..iM}_{..l..}
            for q in 0..N {
                let jq = lower[q];
                let mut iter = (0..dim).map(|l| {
                    let gamma = christoffel.component(l, k, jq);
                    let mut t_idx: Vec<usize> = upper.to_vec();
                    t_idx.extend_from_slice(lower);
                    t_idx[M + q] = l;
                    gamma * tensor.component(&t_idx)
                });
                let first = iter.next().unwrap();
                let correction = iter.fold(first, |acc, x| acc + x);
                result = result - correction;
            }

            result
        })
        .collect();

    Tensor::new(dim, components)
}
