use aad::number::Number;

use crate::tensor::{flat_index, Tensor};

// ── ExtrinsicCurvature ────────────────────────────────────────────────────────

/// Extrinsic curvature K_{ij} of the spatial hypersurface.
///
/// K_{ij} measures how the 3D spatial slice is embedded in 4D spacetime — it
/// is the "shape tensor" of the slice as seen from the ambient spacetime.
///
/// K_{ij} is a symmetric covariant 2-tensor on the 3D spatial slice.
/// Symmetry K_{ij} = K_{ji} is enforced at construction.
///
/// Components are stored flat in row-major order [i, j], each running 0..dim.
/// For the standard 3+1 ADM decomposition, dim = 3.
#[derive(Debug, Clone)]
pub struct ExtrinsicCurvature {
    pub components: Vec<Number>,
    pub dim: usize,
}

impl ExtrinsicCurvature {
    /// Construct from a pre-built `Number` component vector.
    ///
    /// Panics if `components.len() != dim²` or if K_{ij} ≠ K_{ji}.
    pub fn new(dim: usize, components: Vec<Number>) -> Self {
        let expected = dim * dim;
        assert_eq!(
            components.len(),
            expected,
            "Expected {} components for ExtrinsicCurvature in dim {}, got {}",
            expected,
            dim,
            components.len()
        );
        for i in 0..dim {
            for j in i + 1..dim {
                let kij = components[flat_index(&[i, j], dim)].result;
                let kji = components[flat_index(&[j, i], dim)].result;
                assert!(
                    (kij - kji).abs() < 1e-12,
                    "ExtrinsicCurvature must be symmetric: K_{{{}{}}} = {} but K_{{{}{}}} = {}",
                    i, j, kij, j, i, kji
                );
            }
        }
        ExtrinsicCurvature { components, dim }
    }

    /// Convenience constructor: wraps plain f64 values as leaf Number nodes.
    pub fn from_f64(dim: usize, values: Vec<f64>) -> Self {
        let components = values.into_iter().map(Number::new).collect();
        Self::new(dim, components)
    }

    /// Return K_{ij}.
    pub fn component(&self, i: usize, j: usize) -> Number {
        self.components[flat_index(&[i, j], self.dim)]
    }

    /// Scalar trace K = γ^{ij} K_{ij}.
    pub fn trace(&self, gamma_inv: &Tensor<2, 0>) -> Number {
        assert_eq!(self.dim, gamma_inv.dim, "Dimension mismatch in ExtrinsicCurvature::trace");
        let dim = self.dim;
        let mut iter = (0..dim).flat_map(|i| {
            (0..dim).map(move |j| gamma_inv.component(&[i, j]) * self.component(i, j))
        });
        let first = iter.next().unwrap();
        iter.fold(first, |acc, x| acc + x)
    }

    /// Raise one index: K^i_j = γ^{ik} K_{kj}.
    ///
    /// Returns a `Tensor<1,1>` with layout [i, j].
    pub fn raise_first(&self, gamma_inv: &Tensor<2, 0>) -> Tensor<1, 1> {
        assert_eq!(self.dim, gamma_inv.dim);
        let dim = self.dim;
        let mut components = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                let mut iter =
                    (0..dim).map(|k| gamma_inv.component(&[i, k]) * self.component(k, j));
                let first = iter.next().unwrap();
                components.push(iter.fold(first, |acc, x| acc + x));
            }
        }
        Tensor::new(dim, components)
    }
}

// ── AdmState ──────────────────────────────────────────────────────────────────

/// ADM state variables at a single grid point.
///
/// The 3+1 ADM decomposition splits the 4-metric as:
///
///   ds² = −α² dt² + γ_{ij}(dx^i + β^i dt)(dx^j + β^j dt)
///
/// Fields:
///   `gamma`  — 3D spatial metric γ_{ij}         (Tensor<0,2>, dim = 3)
///   `k`      — extrinsic curvature K_{ij}        (symmetric, dim = 3)
///   `alpha`  — lapse function α (gauge)
///   `beta`   — shift vector β^i (upper index, gauge)
#[derive(Debug, Clone)]
pub struct AdmState {
    pub gamma: Tensor<0, 2>,
    pub k: ExtrinsicCurvature,
    pub alpha: f64,
    pub beta: [f64; 3],
}

impl AdmState {
    /// Flat Minkowski initial data: γ_{ij} = δ_{ij}, K_{ij} = 0, α = 1, β^i = 0.
    ///
    /// This is the correct initial condition for a time-symmetric slice of flat
    /// spacetime in Cartesian coordinates with geodesic slicing.
    pub fn flat() -> Self {
        let dim = 3;
        let gamma_vals = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        AdmState {
            gamma: Tensor::from_f64(dim, gamma_vals),
            k: ExtrinsicCurvature::from_f64(dim, vec![0.0; dim * dim]),
            alpha: 1.0,
            beta: [0.0; 3],
        }
    }

    /// Compute the inverse spatial metric γ^{ij} via Gauss-Jordan elimination.
    ///
    /// Returns `None` if γ is singular.
    pub fn gamma_inv(&self) -> Option<Tensor<2, 0>> {
        let dim = self.gamma.dim;
        let flat: Vec<f64> = self.gamma.components.iter().map(|n| n.result).collect();
        let inv_flat = crate::solver::invert_matrix(&flat, dim)?;
        let components = inv_flat.into_iter().map(Number::new).collect();
        Some(Tensor::new(dim, components))
    }
}
