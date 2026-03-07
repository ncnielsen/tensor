use aad::number::Number;

use crate::tensor::{flat_index, Tensor};

/// Christoffel symbols of the second kind, Γ^i_{jk}.
///
/// Unlike tensors, Christoffel symbols are NOT coordinate-independent — they
/// acquire an extra inhomogeneous term under a coordinate change:
///
///   Γ'^i_{jk} = (∂x'^i/∂x^m)(∂x^n/∂x'^j)(∂x^p/∂x'^k) Γ^m_{np}
///             + (∂x'^i/∂x^m)(∂²x^m / ∂x'^j ∂x'^k)
///
/// This dedicated type makes that mathematical distinction explicit at the
/// type level, preventing Christoffel symbols from being silently treated as
/// ordinary (1,2) tensors.
///
/// For a torsion-free (Levi-Civita) connection the symbols are symmetric in
/// their two lower indices, Γ^i_{jk} = Γ^i_{kj}, which is enforced at
/// construction time.
///
/// Components are stored flat in row-major order over [i, j, k], each running
/// from 0 to dim-1. Use `component(i, j, k)` to access Γ^i_{jk}.
#[derive(Debug, Clone)]
pub struct Christoffel {
    pub components: Vec<Number>,
    pub dim: usize,
}

impl Christoffel {
    /// Construct from a pre-built `Number` component vector.
    ///
    /// Panics if `components.len() != dim³` or if the lower-index symmetry
    /// Γ^i_{jk} = Γ^i_{kj} is violated.
    pub fn new(dim: usize, components: Vec<Number>) -> Self {
        let expected = dim.pow(3);
        assert_eq!(
            components.len(),
            expected,
            "Expected {} components for Christoffel in dim {}, got {}",
            expected,
            dim,
            components.len()
        );
        for i in 0..dim {
            for j in 0..dim {
                for k in j + 1..dim {
                    let gjk = components[flat_index(&[i, j, k], dim)].result;
                    let gkj = components[flat_index(&[i, k, j], dim)].result;
                    assert!(
                        (gjk - gkj).abs() < 1e-12,
                        "Christoffel symbols must be symmetric in lower indices: \
                         Γ^{}_{{{},{}}} = {} but Γ^{}_{{{},{}}} = {}",
                        i,
                        j,
                        k,
                        gjk,
                        i,
                        k,
                        j,
                        gkj
                    );
                }
            }
        }
        Christoffel { components, dim }
    }

    /// Convenience constructor from plain `f64` values (wraps each as a leaf `Number`).
    pub fn from_f64(dim: usize, values: Vec<f64>) -> Self {
        let components = values.into_iter().map(Number::new).collect();
        Self::new(dim, components)
    }

    /// Compute Christoffel symbols from a metric tensor and its partial derivatives.
    ///
    /// Uses the Levi-Civita formula:
    ///   Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})
    ///
    /// Arguments:
    ///   - `g`         — covariant metric tensor g_{ij}           (Tensor<0,2>)
    ///   - `g_inv`     — contravariant inverse metric g^{ij}      (Tensor<2,0>)
    ///   - `partial_g` — partial derivatives of the metric        (Tensor<0,3>)
    ///                   layout [i, j, k]: component(&[i,j,k]) = ∂_k g_{ij}
    ///
    /// Panics if dimensions do not all match.
    pub fn from_metric(
        g: &Tensor<0, 2>,
        g_inv: &Tensor<2, 0>,
        partial_g: &Tensor<0, 3>,
    ) -> Self {
        assert_eq!(
            g.dim, g_inv.dim,
            "Dimension mismatch: g ({}) vs g_inv ({})",
            g.dim, g_inv.dim
        );
        assert_eq!(
            g.dim, partial_g.dim,
            "Dimension mismatch: g ({}) vs partial_g ({})",
            g.dim, partial_g.dim
        );

        let dim = g.dim;
        assert!(dim >= 1, "Dimension must be at least 1");

        // Build components in row-major layout [k, i, j] = Γ^k_{ij}
        let mut components = Vec::with_capacity(dim.pow(3));
        for k in 0..dim {
            for i in 0..dim {
                for j in 0..dim {
                    // Γ^k_{ij} = ½ Σ_l g^{kl} (∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})
                    let mut iter = (0..dim).map(|l| {
                        let g_kl    = g_inv.component(&[k, l]);
                        let d_i_gjl = partial_g.component(&[j, l, i]);
                        let d_j_gil = partial_g.component(&[i, l, j]);
                        let d_l_gij = partial_g.component(&[i, j, l]);
                        g_kl * (d_i_gjl + d_j_gil - d_l_gij)
                    });
                    let first = iter.next().unwrap();
                    let sum = iter.fold(first, |acc, x| acc + x);
                    components.push(sum * 0.5);
                }
            }
        }

        // Self::new validates the lower-index symmetry
        Self::new(dim, components)
    }

    /// Return Γ^i_{jk}.
    pub fn component(&self, i: usize, j: usize, k: usize) -> Number {
        self.components[flat_index(&[i, j, k], self.dim)]
    }
}
