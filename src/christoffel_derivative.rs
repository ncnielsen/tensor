use aad::number::Number;

use crate::tensor::flat_index;

/// Partial derivatives of the Christoffel symbols, ∂_ν Γ^ρ_{κμ}.
///
/// Because Γ^ρ_{κμ} = Γ^ρ_{μκ} (torsion-free connection), differentiating
/// preserves that symmetry: ∂_ν Γ^ρ_{κμ} = ∂_ν Γ^ρ_{μκ}, which is enforced
/// at construction time.
///
/// Components are stored flat in row-major order over [ρ, κ, μ, ν], each
/// index running from 0 to dim-1.  Access via `component(rho, kappa, mu, nu)`.
#[derive(Debug, Clone)]
pub struct ChristoffelDerivative {
    pub components: Vec<Number>,
    pub dim: usize,
}

impl ChristoffelDerivative {
    /// Construct from a pre-built `Number` component vector.
    ///
    /// Panics if `components.len() != dim⁴` or if the symmetry
    /// ∂_ν Γ^ρ_{κμ} = ∂_ν Γ^ρ_{μκ} is violated.
    pub fn new(dim: usize, components: Vec<Number>) -> Self {
        let expected = dim.pow(4);
        assert_eq!(
            components.len(),
            expected,
            "Expected {} components for ChristoffelDerivative in dim {}, got {}",
            expected,
            dim,
            components.len()
        );
        for rho in 0..dim {
            for kappa in 0..dim {
                for mu in kappa + 1..dim {
                    for nu in 0..dim {
                        let a = components[flat_index(&[rho, kappa, mu, nu], dim)].result;
                        let b = components[flat_index(&[rho, mu, kappa, nu], dim)].result;
                        assert!(
                            (a - b).abs() < 1e-12,
                            "ChristoffelDerivative must satisfy ∂_{}Γ^{}_{{{},{}}} = ∂_{}Γ^{}_{{{},{}}}: \
                             {} != {}",
                            nu, rho, kappa, mu, nu, rho, mu, kappa, a, b
                        );
                    }
                }
            }
        }
        ChristoffelDerivative { components, dim }
    }

    /// Convenience constructor from plain `f64` values (wraps each as a leaf `Number`).
    pub fn from_f64(dim: usize, values: Vec<f64>) -> Self {
        let components = values.into_iter().map(Number::new).collect();
        Self::new(dim, components)
    }

    /// Return ∂_ν Γ^ρ_{κμ}.
    pub fn component(&self, rho: usize, kappa: usize, mu: usize, nu: usize) -> Number {
        self.components[flat_index(&[rho, kappa, mu, nu], self.dim)]
    }
}
