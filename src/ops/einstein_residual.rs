use crate::christoffel::Christoffel;
use crate::ops::einstein_tensor::einstein_tensor;
use crate::ops::partial_deriv::{christoffel_partial_deriv, partial_deriv};
use crate::ops::ricci_scalar::ricci_scalar;
use crate::ops::ricci_tensor::ricci_tensor;
use crate::ops::riemann::riemann;
use crate::tensor::Tensor;

/// Residual of the Einstein field equations at a single spacetime point.
///
/// Given the metric g_{μν} (as a smooth function of position), the inverse
/// metric g^{μν}, the stress-energy tensor T_{μν}, and the coupling constant
/// κ = 8πG/c⁴, computes:
///
///   ℛ_{μν} = G_{μν} − κ T_{μν}
///
/// This vanishes exactly when the Einstein field equations G_{μν} = κ T_{μν}
/// are satisfied.  The residual is a Tensor<0,2> in the same dimension as the
/// metric.
///
/// # Computation pipeline
///
/// All geometric quantities are derived numerically at `point` using central
/// differences with step size `h`:
///
///   1. ∂_k g_{ij}                          via `partial_deriv`
///   2. Γ^k_{ij}                            via `Christoffel::from_metric`
///   3. ∂_ν Γ^ρ_{κμ}                        via `christoffel_partial_deriv`
///      (nested finite differences: O(h²) accurate)
///   4. R^ρ_{σμν}                           via `riemann`
///   5. R_{σν} = Σ_ρ R^ρ_{σρν}             via `ricci_tensor`
///   6. R = g^{μν} R_{μν}                   via `ricci_scalar`
///   7. G_{μν} = R_{μν} − ½ g_{μν} R       via `einstein_tensor`
///   8. ℛ_{μν} = G_{μν} − κ T_{μν}
///
/// # Arguments
/// - `g_fn`     — covariant metric as a function of spacetime coordinates
/// - `g_inv_fn` — inverse metric as a function of spacetime coordinates
/// - `t`        — stress-energy tensor T_{μν} at `point`
/// - `point`    — spacetime coordinates at which to evaluate the residual
/// - `h`        — finite-difference step size (≈ 1e-5 is a good default)
/// - `kappa`    — coupling constant 8πG/c⁴ (use 1.0 for geometric units)
pub fn einstein_residual(
    g_fn: &dyn Fn(&[f64]) -> Tensor<0, 2>,
    g_inv_fn: &dyn Fn(&[f64]) -> Tensor<2, 0>,
    t: &Tensor<0, 2>,
    point: &[f64],
    h: f64,
    kappa: f64,
) -> Tensor<0, 2> {
    let dim = point.len();

    let g = g_fn(point);
    let g_inv = g_inv_fn(point);

    assert_eq!(g.dim, dim, "metric dim must equal point dimension");
    assert_eq!(t.dim, dim, "stress-energy dim must equal point dimension");

    // Step 1-2: metric derivatives + Christoffel at the point.
    let partial_g = partial_deriv(g_fn, point, h);
    let gamma = Christoffel::from_metric(&g, &g_inv, &partial_g);

    // Step 3: ∂_ν Γ^ρ_{κμ} via nested central differences.
    // Each call to gamma_fn internally runs partial_deriv (2·dim evaluations
    // of g_fn), and christoffel_partial_deriv runs gamma_fn 2·dim times, so
    // the total cost is O(dim²) evaluations of g_fn. The nested truncation
    // errors cancel in the outer difference, preserving O(h²) accuracy.
    let gamma_fn = |x: &[f64]| {
        let gx = g_fn(x);
        let gi = g_inv_fn(x);
        let pg = partial_deriv(g_fn, x, h);
        Christoffel::from_metric(&gx, &gi, &pg)
    };
    let partial_gamma = christoffel_partial_deriv(&gamma_fn, point, h);

    // Steps 4-7: Riemann → Ricci tensor → Ricci scalar → Einstein tensor.
    let r = riemann(&gamma, &partial_gamma);
    let ric = ricci_tensor(&r);
    let scalar = ricci_scalar(&g_inv, &ric);
    let capital_g = einstein_tensor(&ric, &g, &scalar);

    // Step 8: residual ℛ_{μν} = G_{μν} − κ T_{μν}.
    let components = capital_g
        .components
        .iter()
        .zip(t.components.iter())
        .map(|(&g_mn, &t_mn)| g_mn - t_mn * kappa)
        .collect();

    Tensor::new(dim, components)
}
