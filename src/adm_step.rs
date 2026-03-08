use aad::number::Number;

use crate::adm::{AdmState, ExtrinsicCurvature};
use crate::adm_grid::{AdmGrid, FIELDS_PER_PT};
use crate::adm_matter::{matter_dk_correction, AdmMatter};
use crate::adm_rhs::adm_rhs_geodesic;
use crate::christoffel::Christoffel;
use crate::christoffel_derivative::ChristoffelDerivative;
use crate::ops::ricci_tensor::ricci_tensor;
use crate::ops::riemann::riemann;
use crate::solver::invert_matrix;
use crate::tensor::Tensor;

// ── Per-point geometry ────────────────────────────────────────────────────────

/// Centered finite-difference ∂_k γ_{ij} at grid point (ix, iy, iz).
///
/// Output layout: `Tensor<0,3>` with component([i, j, k]) = ∂_k γ_{ij}.
fn partial_gamma_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> Tensor<0, 3> {
    // 27 components: flat index = i*9 + j*3 + k
    let mut comps = [0.0_f64; 27];

    let gp_x = grid.gamma_flat(ix + 1, iy, iz);
    let gm_x = grid.gamma_flat(ix - 1, iy, iz);
    let gp_y = grid.gamma_flat(ix, iy + 1, iz);
    let gm_y = grid.gamma_flat(ix, iy - 1, iz);
    let gp_z = grid.gamma_flat(ix, iy, iz + 1);
    let gm_z = grid.gamma_flat(ix, iy, iz - 1);

    let inv_2dx = 0.5 / grid.dx;
    let inv_2dy = 0.5 / grid.dy;
    let inv_2dz = 0.5 / grid.dz;

    for i in 0..3 {
        for j in 0..3 {
            let ij = i * 3 + j;
            comps[i * 9 + j * 3 + 0] = (gp_x[ij] - gm_x[ij]) * inv_2dx;
            comps[i * 9 + j * 3 + 1] = (gp_y[ij] - gm_y[ij]) * inv_2dy;
            comps[i * 9 + j * 3 + 2] = (gp_z[ij] - gm_z[ij]) * inv_2dz;
        }
    }

    Tensor::from_f64(3, comps.to_vec())
}

/// 3D spatial Christoffel Γ^k_{ij} at grid point (ix, iy, iz).
fn christoffel_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> Christoffel {
    let gamma_flat = grid.gamma_flat(ix, iy, iz);
    let gamma_inv_flat = invert_matrix(&gamma_flat, 3).expect("singular metric at grid point");

    let gamma: Tensor<0, 2> = Tensor::from_f64(3, gamma_flat.to_vec());
    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);
    let partial_g = partial_gamma_at(grid, ix, iy, iz);

    Christoffel::from_metric(&gamma, &gamma_inv, &partial_g)
}

/// ∂_ν Γ^ρ_{κμ} at (ix, iy, iz) via centered FD of Christoffel at ±1 neighbors.
fn christoffel_deriv_at(
    grid: &AdmGrid,
    ix: usize,
    iy: usize,
    iz: usize,
) -> ChristoffelDerivative {
    let gp_x = christoffel_at(grid, ix + 1, iy, iz);
    let gm_x = christoffel_at(grid, ix - 1, iy, iz);
    let gp_y = christoffel_at(grid, ix, iy + 1, iz);
    let gm_y = christoffel_at(grid, ix, iy - 1, iz);
    let gp_z = christoffel_at(grid, ix, iy, iz + 1);
    let gm_z = christoffel_at(grid, ix, iy, iz - 1);

    let inv_2dx = 0.5 / grid.dx;
    let inv_2dy = 0.5 / grid.dy;
    let inv_2dz = 0.5 / grid.dz;

    // Layout [ρ, κ, μ, ν]: flat = ρ*27 + κ*9 + μ*3 + ν
    let mut comps = Vec::with_capacity(81);
    for rho in 0..3 {
        for kappa in 0..3 {
            for mu in 0..3 {
                comps.push(
                    (gp_x.component(rho, kappa, mu) - gm_x.component(rho, kappa, mu))
                        * inv_2dx,
                );
                comps.push(
                    (gp_y.component(rho, kappa, mu) - gm_y.component(rho, kappa, mu))
                        * inv_2dy,
                );
                comps.push(
                    (gp_z.component(rho, kappa, mu) - gm_z.component(rho, kappa, mu))
                        * inv_2dz,
                );
            }
        }
    }

    ChristoffelDerivative::new(3, comps)
}

/// Compute (∂_t γ_{ij}, ∂_t K_{ij}) at one interior grid point under geodesic slicing.
///
/// Returns flat [f64; 9] arrays for dgamma/dt and dK/dt.
/// All geometry is computed inside `aad::no_tape` — no AAD overhead.
fn rhs_at_geodesic(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> ([f64; 9], [f64; 9]) {
    aad::no_tape(|| {
        let gamma_flat = grid.gamma_flat(ix, iy, iz);
        let k_flat = grid.k_flat(ix, iy, iz);

        let gamma_inv_flat =
            invert_matrix(&gamma_flat, 3).expect("singular spatial metric in RHS");
        let gamma: Tensor<0, 2> = Tensor::from_f64(3, gamma_flat.to_vec());
        let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);

        let christoffel = christoffel_at(grid, ix, iy, iz);
        let partial_christoffel = christoffel_deriv_at(grid, ix, iy, iz);

        let riemann_tensor = riemann(&christoffel, &partial_christoffel);
        let ricci = ricci_tensor(&riemann_tensor);

        let state = AdmState {
            gamma,
            k: ExtrinsicCurvature::new(
                3,
                k_flat.iter().map(|&v| Number::new(v)).collect(),
            ),
            alpha: 1.0,
            beta: [0.0; 3],
        };

        let rhs = adm_rhs_geodesic(&state, &gamma_inv, &ricci);

        let dgamma: [f64; 9] = rhs
            .dgamma_dt
            .components
            .iter()
            .map(|n| n.result)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let dk: [f64; 9] = rhs
            .dk_dt
            .components
            .iter()
            .map(|n| n.result)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        (dgamma, dk)
    })
}

// ── Full-grid RHS ─────────────────────────────────────────────────────────────

/// Compute the RHS field vector for the entire grid (geodesic slicing).
///
/// Returns a `Vec<f64>` with the same layout as `grid.fields`.
/// Boundary entries (within 2 cells of the edge) are set to zero.
pub fn geodesic_rhs(grid: &AdmGrid) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; grid.fields.len()];

    for ix in 2..grid.nx.saturating_sub(2) {
        for iy in 2..grid.ny.saturating_sub(2) {
            for iz in 2..grid.nz.saturating_sub(2) {
                let (dgamma, dk) = rhs_at_geodesic(grid, ix, iy, iz);
                let base = grid.flat_pt(ix, iy, iz) * FIELDS_PER_PT;
                rhs[base..base + 9].copy_from_slice(&dgamma);
                rhs[base + 9..base + 18].copy_from_slice(&dk);
            }
        }
    }

    rhs
}

// ── RK4 time step ─────────────────────────────────────────────────────────────

/// Advance the grid one time step of size `dt` using 4th-order Runge-Kutta.
///
/// Uses geodesic slicing (α = 1, β = 0) throughout.
/// Boundary values are frozen; only interior points `[2..n-2]` are evolved.
///
/// The CFL condition for stability (heuristic) is:
///   dt ≤ 0.5 * min(dx, dy, dz)
pub fn adm_step_rk4(grid: &AdmGrid, dt: f64) -> AdmGrid {
    // k1 = RHS(y_n)
    let k1 = geodesic_rhs(grid);

    // k2 = RHS(y_n + dt/2 * k1)
    let g2 = grid.with_rhs(&k1, dt * 0.5);
    let k2 = geodesic_rhs(&g2);

    // k3 = RHS(y_n + dt/2 * k2)
    let g3 = grid.with_rhs(&k2, dt * 0.5);
    let k3 = geodesic_rhs(&g3);

    // k4 = RHS(y_n + dt * k3)
    let g4 = grid.with_rhs(&k3, dt);
    let k4 = geodesic_rhs(&g4);

    // y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    let mut combined = vec![0.0_f64; grid.fields.len()];
    for i in 0..combined.len() {
        combined[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }

    grid.with_rhs(&combined, dt)
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

/// L2 norm of the Hamiltonian constraint violation across all interior points.
///
/// For vacuum flat initial data this starts near machine precision and grows
/// due to numerical truncation error. Useful for monitoring solution quality.
pub fn hamiltonian_l2(grid: &AdmGrid) -> f64 {
    use crate::adm_rhs::{hamiltonian_constraint, k_squared};
    use crate::ops::ricci_scalar::ricci_scalar;

    let mut sum = 0.0;
    let mut count = 0;

    for ix in 2..grid.nx.saturating_sub(2) {
        for iy in 2..grid.ny.saturating_sub(2) {
            for iz in 2..grid.nz.saturating_sub(2) {
                aad::no_tape(|| {
                    let gamma_flat = grid.gamma_flat(ix, iy, iz);
                    let gamma_inv_flat = invert_matrix(&gamma_flat, 3).unwrap();
                    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);

                    let christoffel = christoffel_at(grid, ix, iy, iz);
                    let partial_christoffel = christoffel_deriv_at(grid, ix, iy, iz);
                    let riemann_tensor = riemann(&christoffel, &partial_christoffel);
                    let ricci = ricci_tensor(&riemann_tensor);
                    let r3_tensor = ricci_scalar(&gamma_inv, &ricci);
                    let r3 = r3_tensor.component(&[]);

                    let k = ExtrinsicCurvature::new(
                        3,
                        grid.k_flat(ix, iy, iz)
                            .iter()
                            .map(|&v| Number::new(v))
                            .collect(),
                    );
                    let k_tr = k.trace(&gamma_inv);
                    let k_sq = k_squared(&k, &gamma_inv);
                    let h = hamiltonian_constraint(r3, k_tr, k_sq, 0.0);
                    sum += h.result * h.result;
                });
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f64).sqrt()
    }
}

// ── Source-coupled RHS and RK4 ────────────────────────────────────────────────

/// Compute (∂_t γ_{ij}, ∂_t K_{ij}) at one interior point, including EM matter source.
///
/// The vacuum geodesic RHS is augmented with:
///   ∂_t K_{ij}|_matter = −8π S_{ij} + 4π γ_{ij}(S − ρ)
fn rhs_at_with_matter(
    grid: &AdmGrid,
    ix: usize,
    iy: usize,
    iz: usize,
    matter: &AdmMatter,
) -> ([f64; 9], [f64; 9]) {
    aad::no_tape(|| {
        let gamma_flat = grid.gamma_flat(ix, iy, iz);
        let k_flat = grid.k_flat(ix, iy, iz);

        let gamma_inv_flat =
            invert_matrix(&gamma_flat, 3).expect("singular spatial metric in matter RHS");
        let gamma: Tensor<0, 2> = Tensor::from_f64(3, gamma_flat.to_vec());
        let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);

        let christoffel = christoffel_at(grid, ix, iy, iz);
        let partial_christoffel = christoffel_deriv_at(grid, ix, iy, iz);
        let riemann_tensor = riemann(&christoffel, &partial_christoffel);
        let ricci = ricci_tensor(&riemann_tensor);

        let state = AdmState {
            gamma: gamma.clone(),
            k: ExtrinsicCurvature::new(3, k_flat.iter().map(|&v| Number::new(v)).collect()),
            alpha: 1.0,
            beta: [0.0; 3],
        };

        let rhs = adm_rhs_geodesic(&state, &gamma_inv, &ricci);

        let dgamma: [f64; 9] = rhs
            .dgamma_dt
            .components
            .iter()
            .map(|n| n.result)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Add matter correction to dk
        let dk_vacuum: [f64; 9] = rhs
            .dk_dt
            .components
            .iter()
            .map(|n| n.result)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let correction = matter_dk_correction(matter, &gamma);
        let mut dk = dk_vacuum;
        for i in 0..9 {
            dk[i] += correction[i];
        }

        (dgamma, dk)
    })
}

/// Compute the full-grid RHS including EM matter sources.
///
/// `matters` must have length `grid.n_pts()` with the same flat ordering as the grid.
/// Boundary entries are zeroed; only interior points `[2..n-2]` are evolved.
pub fn geodesic_rhs_with_matter(grid: &AdmGrid, matters: &[AdmMatter]) -> Vec<f64> {
    assert_eq!(
        matters.len(),
        grid.n_pts(),
        "matters must have one entry per grid point"
    );
    let mut rhs = vec![0.0_f64; grid.fields.len()];

    for ix in 2..grid.nx.saturating_sub(2) {
        for iy in 2..grid.ny.saturating_sub(2) {
            for iz in 2..grid.nz.saturating_sub(2) {
                let pt = grid.flat_pt(ix, iy, iz);
                let (dgamma, dk) = rhs_at_with_matter(grid, ix, iy, iz, &matters[pt]);
                let base = pt * FIELDS_PER_PT;
                rhs[base..base + 9].copy_from_slice(&dgamma);
                rhs[base + 9..base + 18].copy_from_slice(&dk);
            }
        }
    }

    rhs
}

/// RK4 step with prescribed EM matter sources.
///
/// `matters` is evaluated once at the current time and held fixed across all
/// RK4 stages — appropriate for an externally driven source field.
///
/// For a self-consistent coupled Maxwell+GR evolution (Phase 4), the source
/// should be re-evaluated at each stage using the updated metric.
pub fn adm_step_rk4_with_source(grid: &AdmGrid, dt: f64, matters: &[AdmMatter]) -> AdmGrid {
    let k1 = geodesic_rhs_with_matter(grid, matters);
    let g2 = grid.with_rhs(&k1, dt * 0.5);
    let k2 = geodesic_rhs_with_matter(&g2, matters);
    let g3 = grid.with_rhs(&k2, dt * 0.5);
    let k3 = geodesic_rhs_with_matter(&g3, matters);
    let g4 = grid.with_rhs(&k3, dt);
    let k4 = geodesic_rhs_with_matter(&g4, matters);

    let mut combined = vec![0.0_f64; grid.fields.len()];
    for i in 0..combined.len() {
        combined[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    grid.with_rhs(&combined, dt)
}
