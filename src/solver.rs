use crate::ops::einstein_residual::einstein_residual;
use crate::ops::newton_step::newton_step;
use crate::tensor::Tensor;

/// Invert a dim×dim matrix stored row-major as a flat `Vec<f64>`.
///
/// Uses Gauss-Jordan elimination with partial pivoting.
/// Returns `None` if any pivot is smaller than 1e-14 (singular matrix).
pub fn invert_matrix(a: &[f64], dim: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), dim * dim);

    // Build the augmented matrix [A | I].
    let mut aug: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            let mut row = a[i * dim..(i + 1) * dim].to_vec();
            row.extend((0..dim).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    let wide = 2 * dim;

    for col in 0..dim {
        // Partial pivot.
        let pivot_row = (col..dim)
            .max_by(|&i, &j| {
                aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap()
            })
            .unwrap();
        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            return None;
        }

        // Normalise pivot row.
        let inv = 1.0 / pivot;
        for c in 0..wide {
            aug[col][c] *= inv;
        }

        // Eliminate this column from every other row.
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for c in 0..wide {
                let t = aug[col][c];
                aug[row][c] -= factor * t;
            }
        }
    }

    // Extract the right half (the inverse).
    Some(
        (0..dim)
            .flat_map(|i| aug[i][dim..].to_vec())
            .collect(),
    )
}

/// Symmetrize a flat dim×dim matrix in place: g_{μν} ← (g_{μν} + g_{νμ}) / 2.
///
/// The physical metric must be symmetric.  When the NR Jacobian perturbs one
/// off-diagonal component, this restores symmetry before the geometry is evaluated.
fn symmetrize(g: &mut [f64], dim: usize) {
    for mu in 0..dim {
        for nu in (mu + 1)..dim {
            let avg = (g[mu * dim + nu] + g[nu * dim + mu]) / 2.0;
            g[mu * dim + nu] = avg;
            g[nu * dim + mu] = avg;
        }
    }
}

// ─── Public result type ────────────────────────────────────────────────────────

/// Returned by [`solve_1d`].
pub struct SolveResult {
    /// Metric at every grid point (boundaries unchanged, interior updated).
    pub g_grid: Vec<Vec<f64>>,
    /// Number of Newton-Raphson steps performed.
    pub iterations: usize,
    /// L∞ norm of the residual G_{μν} − κ T_{μν} at the final iterate.
    pub residual_norm: f64,
    /// `true` if `residual_norm < tol` was achieved within `max_iter` steps.
    pub converged: bool,
}

// ─── Solver ───────────────────────────────────────────────────────────────────

/// Newton-Raphson solver for G_{μν} = κ T_{μν} on a uniform 1-D spatial grid.
///
/// # Grid layout
///
/// N points at positions x_i = i·h, i = 0…N−1.  The **first and last** points
/// are Dirichlet boundary conditions (held fixed throughout).  The N−2 interior
/// points are the unknowns.
///
/// The metric varies along coordinate direction 0 only; all other coordinates
/// are held at 0.
///
/// # Algorithm — per outer iteration
///
/// 1. Symmetrize all interior metric values  (g_{μν} ← (g_{μν}+g_{νμ})/2).
/// 2. Compute g_inv = g⁻¹ at every grid point via Gauss-Jordan elimination.
/// 3. Evaluate the full residual vector (n_interior × dim² components).
/// 4. If ‖ℛ‖∞ < `tol`, declare convergence and stop.
/// 5. One Newton-Raphson step: numerical Jacobian (central differences,
///    step `eps`) + Gaussian elimination.
///
/// # Arguments
///
/// - `g_grid`   — initial metric \[N\]\[dim²\], row-major flat storage
/// - `t_grid`   — stress-energy T_{μν} at each grid point \[N\]
/// - `h`        — uniform grid spacing (also used as the spatial FD step)
/// - `kappa`    — coupling constant 8πG/c⁴ (use 1.0 for geometric units)
/// - `tol`      — convergence criterion: stop when max|ℛ_{μν}| < tol
/// - `max_iter` — maximum number of NR iterations before giving up
/// - `eps`      — step size for the numerical Jacobian inside `newton_step`
pub fn solve_1d(
    g_grid: &[Vec<f64>],
    t_grid: &[Tensor<0, 2>],
    h: f64,
    kappa: f64,
    tol: f64,
    max_iter: usize,
    eps: f64,
) -> SolveResult {
    let n_points = g_grid.len();
    assert!(n_points >= 3, "Need at least 3 grid points (2 boundary + ≥1 interior)");
    assert_eq!(t_grid.len(), n_points, "t_grid must have one entry per grid point");

    let n_interior = n_points - 2;
    let dim2 = g_grid[0].len();
    let dim = (dim2 as f64).sqrt() as usize;
    assert_eq!(dim * dim, dim2, "Metric must have dim² components for some integer dim");

    let mut grid = g_grid.to_vec();
    let boundary_left = g_grid[0].clone();
    let boundary_right = g_grid[n_points - 1].clone();

    let mut iterations = 0usize;
    let mut residual_norm = f64::INFINITY;
    let mut converged = false;

    // +1 so we always evaluate the residual after the last step.
    for iter in 0..=max_iter {
        // ── 1. Symmetrize interior metrics ────────────────────────────────
        for i in 1..n_points - 1 {
            symmetrize(&mut grid[i], dim);
        }

        // ── 2. Compute g_inv at every grid point ──────────────────────────
        let identity: Vec<f64> = (0..dim2)
            .map(|k| if k / dim == k % dim { 1.0 } else { 0.0 })
            .collect();

        let g_inv_grid: Vec<Vec<f64>> = grid
            .iter()
            .map(|g| invert_matrix(g, dim).unwrap_or_else(|| identity.clone()))
            .collect();

        // ── 3. Flatten interior values ────────────────────────────────────
        let x0: Vec<f64> = (1..n_points - 1)
            .flat_map(|i| grid[i].iter().copied())
            .collect();

        // ── Build residual closure for this iteration ─────────────────────
        //
        // Captures: boundary values, current g_inv, t_grid reference.
        // The closure is recreated each iteration so g_inv stays consistent.
        let bl = boundary_left.clone();
        let br = boundary_right.clone();
        let gi = g_inv_grid; // moved in

        let residual_fn = move |g_interior: &[f64]| -> Vec<f64> {
            // Reconstruct full grid with symmetrized interior.
            let mut full: Vec<Vec<f64>> = vec![bl.clone()];
            for k in 0..n_interior {
                let off = k * dim2;
                let mut g = g_interior[off..off + dim2].to_vec();
                symmetrize(&mut g, dim);
                full.push(g);
            }
            full.push(br.clone());

            // Residual at each interior point.
            (1..n_points - 1)
                .flat_map(|i| {
                    let g_fn = |x: &[f64]| -> Tensor<0, 2> {
                        let j = (x[0] / h).round() as usize;
                        let j = j.clamp(0, n_points - 1);
                        Tensor::from_f64(dim, full[j].clone())
                    };
                    let g_inv_fn = |x: &[f64]| -> Tensor<2, 0> {
                        let j = (x[0] / h).round() as usize;
                        let j = j.clamp(0, n_points - 1);
                        Tensor::from_f64(dim, gi[j].clone())
                    };

                    let mut point = vec![0.0f64; dim];
                    point[0] = i as f64 * h;

                    einstein_residual(
                        &g_fn,
                        &g_inv_fn,
                        &t_grid[i],
                        &point,
                        h,
                        kappa,
                    )
                    .components
                    .iter()
                    .map(|c| c.result)
                    .collect::<Vec<_>>()
                })
                .collect()
        };

        // ── 4. Check convergence ──────────────────────────────────────────
        let r = residual_fn(&x0);
        residual_norm = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        if residual_norm < tol {
            converged = true;
            break;
        }

        if iter == max_iter {
            break; // Reached limit without converging.
        }

        // ── 5. Newton-Raphson step ────────────────────────────────────────
        let x_new = newton_step(&residual_fn, &x0, eps);

        for k in 0..n_interior {
            let off = k * dim2;
            grid[k + 1] = x_new[off..off + dim2].to_vec();
        }

        iterations += 1;
    }

    SolveResult {
        g_grid: grid,
        iterations,
        residual_norm,
        converged,
    }
}
