use crate::tensor::Tensor;

/// 3+1 decomposition of the stress-energy tensor T_{μν} at one grid point.
///
/// Under geodesic slicing (α = 1, β^i = 0) the unit normal to the spatial
/// hypersurface is n^μ = (1, 0, 0, 0), so the standard projections reduce to:
///
///   ρ     = T_{μν} n^μ n^ν = T_{00}          (energy density)
///   J^i   = −γ^{ij} T_{0j}                   (momentum density, upper index)
///   S_{ij} = T_{ij}  (3D spatial block)       (spatial stress tensor)
///   S     = γ^{ij} S_{ij}                     (trace of S)
///
/// Index convention for the 4D tensor T: 0 = time, 1,2,3 = x,y,z.
/// The 3D tensor S uses indices 0,1,2 = x,y,z.
#[derive(Debug, Clone, Copy)]
pub struct AdmMatter {
    /// Energy density ρ = T_{00}.
    pub rho: f64,
    /// Momentum density J^i (upper index), i = 0..2.
    pub j: [f64; 3],
    /// Spatial stress S_{ij} (lower indices), flat row-major [i*3+j].
    pub s_ij: [f64; 9],
    /// Trace S = γ^{ij} S_{ij}.
    pub s_trace: f64,
}

impl AdmMatter {
    /// Vacuum: ρ = 0, J = 0, S = 0.
    pub fn vacuum() -> Self {
        AdmMatter { rho: 0.0, j: [0.0; 3], s_ij: [0.0; 9], s_trace: 0.0 }
    }

    /// Decompose a 4D T_{μν} (dim=4 Tensor<0,2>) under geodesic slicing.
    ///
    /// `t4` — EM stress-energy tensor with dim=4 layout [μ, ν] (μ=0 is time).
    /// `gamma_inv` — 3D inverse spatial metric γ^{ij}  (Tensor<2,0>, dim=3).
    pub fn from_t4d(t4: &Tensor<0, 2>, gamma_inv: &Tensor<2, 0>) -> Self {
        assert_eq!(t4.dim, 4, "T_{{μν}} must be 4D for ADM decomposition");
        assert_eq!(gamma_inv.dim, 3);

        // ρ = T_{00}
        let rho = t4.component(&[0, 0]).result;

        // S_{ij} = T_{i+1, j+1}  (4D spatial indices 1,2,3 → 3D indices 0,1,2)
        let mut s_ij = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                s_ij[i * 3 + j] = t4.component(&[i + 1, j + 1]).result;
            }
        }

        // S = γ^{ij} S_{ij}
        let s_trace = (0..3)
            .flat_map(|i| (0..3).map(move |j| gamma_inv.component(&[i, j]).result * s_ij[i * 3 + j]))
            .sum::<f64>();

        // J^i = −γ^{ij} T_{0,j+1}   (upper-index, 3D)
        let mut j = [0.0_f64; 3];
        for i in 0..3 {
            j[i] = -(0..3)
                .map(|jj| gamma_inv.component(&[i, jj]).result * t4.component(&[0, jj + 1]).result)
                .sum::<f64>();
        }

        AdmMatter { rho, j, s_ij, s_trace }
    }
}

/// Vacuum ADM K_{ij} right-hand side extended with a matter source.
///
/// Adds the 8πG matter terms to the geodesic-slicing RHS:
///
///   ∂_t K_{ij}|_matter = −8π S_{ij} + 4π γ_{ij} (S − ρ)
///
/// Call this and add it to the vacuum RHS from `adm_rhs_geodesic`.
/// Uses geometric units G = c = 1, so the coupling constant is 8π.
pub fn matter_dk_correction(
    matter: &AdmMatter,
    gamma: &Tensor<0, 2>,
) -> [f64; 9] {
    let mut delta = [0.0_f64; 9];
    let s_minus_rho = matter.s_trace - matter.rho;
    for i in 0..3 {
        for j in 0..3 {
            let gamma_ij = gamma.component(&[i, j]).result;
            delta[i * 3 + j] =
                -8.0 * std::f64::consts::PI * matter.s_ij[i * 3 + j]
                + 4.0 * std::f64::consts::PI * gamma_ij * s_minus_rho;
        }
    }
    delta
}
