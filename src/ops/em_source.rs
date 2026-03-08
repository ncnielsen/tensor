use aad::automatic_differentiator::AutomaticDifferentiator;
use aad::number::Number;

use crate::ops::em_stress_energy::em_stress_energy;
use crate::ops::faraday::faraday;
use crate::ops::partial_deriv::partial_deriv;
use crate::solver::invert_matrix;
use crate::tensor::Tensor;

fn clear_tape() {
    let arg = Number::new(0.0);
    AutomaticDifferentiator::new().derivatives(|_| Number::new(0.0), &[arg]);
}

/// Compute the EM stress-energy tensor T_{μν} at every point of a 3-D spatial grid.
///
/// # EM pipeline per grid point
///
/// 1. Evaluate ∂_μ A_ν numerically:
///    `partial_a[ν, μ] = (A_ν(x + eps·eμ) − A_ν(x − eps·eμ)) / (2·eps)`
/// 2. Faraday tensor: `F_{μν} = ∂_μ A_ν − ∂_ν A_μ`.
/// 3. EM stress-energy:
///    `T_{μν} = (F_{μλ} g^{λρ} F_{νρ} − ¼ g_{μν} F_{λρ} F^{λρ}) / μ₀`.
///
/// # Tornado field prescription
///
/// For a magnetic vortex aligned with coordinate-2 (z-axis) centred at
/// (x₀, y₀) in the coordinate-0/1 plane, use the 4-potential
///
/// ```text
/// A₀(x) = −½ B₀ (x₁ − y₀) exp(−r²/2σ²)
/// A₁(x) =  ½ B₀ (x₀ − x₀) exp(−r²/2σ²)
/// A₂ = A₃ = 0
/// ```
///
/// This produces F_{01} = B₀ exp(−r²/2σ²)(1 − r²/2σ²) — a circulating axial
/// magnetic flux tube that sources angular-momentum-carrying T_{μν}, the seed
/// of the spacetime tornado.
///
/// # Arguments
///
/// - `a_fn`   — 4-potential A_μ at a spacetime point; called as `a_fn(x)` where
///              `x` is the `dim`-dimensional coordinate; returns `Vec<f64>` of
///              `dim` components
/// - `g_grid` — metric g_{μν} at every grid point (flat `[nx·ny·nz]`, each
///              entry has `dim²` row-major components)
/// - `nx`, `ny`, `nz` — grid dimensions (≥ 1 each)
/// - `h`      — spatial grid spacing (used to set the coordinate of each point)
/// - `mu_0`   — magnetic permeability (use 1.0 for geometric units)
/// - `eps`    — finite-difference step for computing ∂_μ A_ν  (use ≈ 1e-5 for
///              smooth potentials; independent of the grid spacing `h`)
///
/// # Returns
///
/// Flat `[nx·ny·nz]` vector of `Tensor<0,2>` — one T_{μν} per grid point.
/// The AAD tape is cleared after each point so memory stays bounded.
pub fn em_t_grid(
    a_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    g_grid: &[Vec<f64>],
    nx: usize,
    ny: usize,
    nz: usize,
    h: f64,
    mu_0: f64,
    eps: f64,
) -> Vec<Tensor<0, 2>> {
    assert_eq!(g_grid.len(), nx * ny * nz, "g_grid must have nx*ny*nz entries");

    let dim2 = g_grid[0].len();
    let dim = (dim2 as f64).sqrt() as usize;
    assert_eq!(dim * dim, dim2, "Metric must have dim² components");
    assert!(dim >= 3, "dim must be ≥ 3 for the 3-D EM source");

    let identity: Vec<f64> = (0..dim2)
        .map(|k| if k / dim == k % dim { 1.0 } else { 0.0 })
        .collect();

    (0..nx * ny * nz)
        .map(|flat| {
            let ix = flat / (ny * nz);
            let iy = (flat / nz) % ny;
            let iz = flat % nz;

            let g_vals = &g_grid[flat];
            let g = Tensor::<0, 2>::from_f64(dim, g_vals.clone());
            let g_inv_vals = invert_matrix(g_vals, dim).unwrap_or_else(|| identity.clone());
            let g_inv = Tensor::<2, 0>::from_f64(dim, g_inv_vals);

            // Spatial coordinates; remaining coordinates (e.g. time) set to 0.
            let mut point = vec![0.0f64; dim];
            point[0] = ix as f64 * h;
            point[1] = iy as f64 * h;
            point[2] = iz as f64 * h;

            // Wrap the scalar-valued A_fn into a Tensor<0,1>-valued function
            // so partial_deriv can compute ∂_μ A_ν.
            let a_tensor_fn = |x: &[f64]| -> Tensor<0, 1> {
                Tensor::from_f64(dim, a_fn(x))
            };

            let partial_a: Tensor<0, 2> = partial_deriv(&a_tensor_fn, &point, eps);
            let f_tensor = faraday(&partial_a);
            let t = em_stress_energy(&f_tensor, &g, &g_inv, mu_0);

            // Prevent unbounded AAD tape growth — we only need the f64 results.
            clear_tape();

            t
        })
        .collect()
}
