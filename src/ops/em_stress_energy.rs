use crate::tensor::{decode_flat_index, Tensor};

/// Electromagnetic stress-energy tensor T_{μν}.
///
/// Given the Faraday tensor F_{μν}, the covariant metric g_{μν}, and the
/// inverse metric g^{μν}, computes:
///
///   T_{μν} = (1/μ₀) [ F_{μλ} g^{λρ} F_{νρ} − ¼ g_{μν} F_{λρ} g^{λα} g^{ρβ} F_{αβ} ]
///
/// The first term encodes momentum flux; the second subtracts the EM
/// invariant I = F_{λρ} F^{λρ} (computed once and reused for all components).
///
/// Convention: this formula is correct for the (−,+,+,+) metric signature.
/// With that signature a pure electric field E gives T_{00} = E²/(2μ₀) > 0
/// (positive energy density) and T_{ii} = ±E²/(2μ₀) for spatial components.
///
/// T_{μν} is symmetric by construction (scalar multiplication of Number
/// is commutative, so F_{μλ} g^{λρ} F_{νρ} = F_{νλ} g^{λρ} F_{μρ}).
pub fn em_stress_energy(
    f: &Tensor<0, 2>,
    g: &Tensor<0, 2>,
    g_inv: &Tensor<2, 0>,
    mu_0: f64,
) -> Tensor<0, 2> {
    assert_eq!(f.dim, g.dim,     "Dimension mismatch: f ({}) vs g ({})",     f.dim, g.dim);
    assert_eq!(f.dim, g_inv.dim, "Dimension mismatch: f ({}) vs g_inv ({})", f.dim, g_inv.dim);
    assert!(f.dim >= 1, "Dimension must be at least 1");

    let dim = f.dim;

    // I = F_{λρ} g^{λα} g^{ρβ} F_{αβ}  (EM invariant, scalar)
    let mut inv_iter = (0..dim.pow(4)).map(|flat| {
        let i = decode_flat_index(flat, dim, 4);
        f.component(&[i[0], i[1]])
            * g_inv.component(&[i[0], i[2]])
            * g_inv.component(&[i[1], i[3]])
            * f.component(&[i[2], i[3]])
    });
    let first_inv = inv_iter.next().unwrap();
    let invariant = inv_iter.fold(first_inv, |acc, x| acc + x);

    // T_{μν} for each output component
    let components = (0..dim.pow(2))
        .map(|flat_out| {
            let out = decode_flat_index(flat_out, dim, 2);
            let mu = out[0];
            let nu = out[1];

            // A_{μν} = Σ_{λ,ρ} F_{μλ} g^{λρ} F_{νρ}
            let mut a_iter = (0..dim.pow(2)).map(|flat| {
                let i = decode_flat_index(flat, dim, 2);
                f.component(&[mu, i[0]])
                    * g_inv.component(&[i[0], i[1]])
                    * f.component(&[nu, i[1]])
            });
            let first_a = a_iter.next().unwrap();
            let a_mn = a_iter.fold(first_a, |acc, x| acc + x);

            // T_{μν} = (A_{μν} − ¼ g_{μν} I) / μ₀
            (a_mn - g.component(&[mu, nu]) * invariant * 0.25) * (1.0 / mu_0)
        })
        .collect();

    Tensor::new(dim, components)
}
