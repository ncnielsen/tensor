use aad::number::Number;

use crate::tensor::flat_index;

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

    /// Return Γ^i_{jk}.
    pub fn component(&self, i: usize, j: usize, k: usize) -> Number {
        self.components[flat_index(&[i, j, k], self.dim)]
    }
}
