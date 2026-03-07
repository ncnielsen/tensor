use crate::tensor::{decode_flat_index, Tensor};

/// Contract one upper index with one lower index of a tensor.
///
/// Given T^{i0..ip..iM-1}_{j0..jq..jN-1}, contract upper index p with lower
/// index q by summing over the shared index k:
///
///   C^{i0..i_{p-1} i_{p+1}..iM-1}_{j0..j_{q-1} j_{q+1}..jN-1}
///     = Σ_k  T^{..k..}_{..k..}
///
/// This is the only legal index summation in tensor calculus: contracting an
/// upper index with a lower index produces a tensor of rank (M-1, N-1).
/// Contracting two upper or two lower indices does not yield a tensor.
///
/// Requires M >= 1 and N >= 1 (enforced at compile time via the where clauses).
pub fn contract<const M: usize, const N: usize>(
    tensor: &Tensor<M, N>,
    upper_idx: usize,
    lower_idx: usize,
) -> Tensor<{ M - 1 }, { N - 1 }>
where
    [(); M - 1]:,
    [(); N - 1]:,
    [(); { M - 1 } + { N - 1 }]:,
{
    assert!(
        upper_idx < M,
        "upper_idx {} out of range for M={}",
        upper_idx,
        M
    );
    assert!(
        lower_idx < N,
        "lower_idx {} out of range for N={}",
        lower_idx,
        N
    );
    assert!(tensor.dim >= 1, "Tensor dimension must be at least 1");

    let dim = tensor.dim;
    let rank_out = (M - 1) + (N - 1);
    let n_out = if rank_out == 0 { 1 } else { dim.pow(rank_out as u32) };

    let components = (0..n_out)
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, rank_out);
            let upper_out = &out[..M - 1];
            let lower_out = &out[M - 1..];

            // Sum over the contracted index k
            let mut iter = (0..dim).map(|k| {
                // Re-insert k at position upper_idx in the upper indices
                let mut upper_full: Vec<usize> = Vec::with_capacity(M);
                upper_full.extend_from_slice(&upper_out[..upper_idx]);
                upper_full.push(k);
                upper_full.extend_from_slice(&upper_out[upper_idx..]);

                // Re-insert k at position lower_idx in the lower indices
                let mut lower_full: Vec<usize> = Vec::with_capacity(N);
                lower_full.extend_from_slice(&lower_out[..lower_idx]);
                lower_full.push(k);
                lower_full.extend_from_slice(&lower_out[lower_idx..]);

                let mut full_indices = upper_full;
                full_indices.extend(lower_full);
                tensor.component(&full_indices)
            });

            // Fold without a spurious 0.0 leaf node: use the first term as seed
            let first = iter.next().unwrap();
            iter.fold(first, |acc, x| acc + x)
        })
        .collect();

    Tensor::new(dim, components)
}
