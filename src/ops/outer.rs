use crate::tensor::{decode_flat_index, flat_index, Tensor};

/// Outer product of two tensors.
///
/// For T^{i1..iM1}_{j1..jN1} and S^{k1..kM2}_{l1..lN2}, the result is:
///   R^{i1..iM1 k1..kM2}_{j1..jN1 l1..lN2}
///
/// Output index layout (row-major): [upper_a, upper_b, lower_a, lower_b]
pub fn outer<const M1: usize, const N1: usize, const M2: usize, const N2: usize>(
    a: &Tensor<M1, N1>,
    b: &Tensor<M2, N2>,
) -> Tensor<{ M1 + M2 }, { N1 + N2 }>
where
    [(); M1 + M2]:,
    [(); N1 + N2]:,
    [(); { M1 + M2 } + { N1 + N2 }]:,
{
    assert_eq!(
        a.dim, b.dim,
        "Dimension mismatch in outer product: {} vs {}",
        a.dim, b.dim
    );
    let dim = a.dim;
    let rank_out = M1 + M2 + N1 + N2;
    let n_out = if rank_out == 0 { 1 } else { dim.pow(rank_out as u32) };

    let components = (0..n_out)
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, rank_out);

            // Split output indices back into the four groups
            let upper_a = &out[..M1];
            let upper_b = &out[M1..M1 + M2];
            let lower_a = &out[M1 + M2..M1 + M2 + N1];
            let lower_b = &out[M1 + M2 + N1..];

            // Reconstruct the original flat indices for a and b
            let mut a_idx: Vec<usize> = upper_a.to_vec();
            a_idx.extend_from_slice(lower_a);

            let mut b_idx: Vec<usize> = upper_b.to_vec();
            b_idx.extend_from_slice(lower_b);

            a.components[flat_index(&a_idx, dim)] * b.components[flat_index(&b_idx, dim)]
        })
        .collect();

    Tensor::new(dim, components)
}
