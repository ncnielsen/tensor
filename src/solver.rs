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

// ─── Public result types ───────────────────────────────────────────────────────

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
            // clear_tape() is called after each point to prevent unbounded tape growth.
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

                    aad::no_tape(|| {
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
                        .collect::<Vec<f64>>()
                    })
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

// ─── 3-D solver ───────────────────────────────────────────────────────────────

/// Returned by [`solve_3d`].
pub struct SolveResult3D {
    /// Metric at every grid point in row-major order.
    /// Flat index of `(ix, iy, iz)` is `ix * ny * nz + iy * nz + iz`.
    pub g_grid: Vec<Vec<f64>>,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Number of Newton-Raphson steps performed.
    pub iterations: usize,
    /// L∞ norm of the residual G_{μν} − κ T_{μν} at the final iterate.
    pub residual_norm: f64,
    /// `true` if `residual_norm < tol` was achieved within `max_iter` steps.
    pub converged: bool,
}

/// Newton-Raphson solver for G_{μν} = κ T_{μν} on a uniform 3-D Cartesian grid.
///
/// # Grid layout
///
/// `nx × ny × nz` points.  Flat index of `(ix, iy, iz)` is
/// `ix * ny * nz + iy * nz + iz`.  All six faces of the bounding box are
/// Dirichlet boundary conditions (held fixed).  The `(nx−2)·(ny−2)·(nz−2)`
/// interior points are the unknowns.
///
/// The metric at cell `(ix, iy, iz)` is evaluated at coordinate
/// `xᵅ = (ix·h, iy·h, iz·h, 0, …)`.  Requires `dim ≥ 3`; any extra
/// spacetime dimensions beyond the three spatial ones are held at zero
/// (static solution).
///
/// # Algorithm — same as `solve_1d`, generalized to 3-D
///
/// Per outer iteration:
/// 1. Symmetrize all interior metric values.
/// 2. Recompute g_inv = g⁻¹ via Gauss-Jordan at every grid point.
/// 3. Evaluate the residual vector (n_interior × dim² components).
/// 4. If ‖ℛ‖∞ < `tol`, declare convergence and stop.
/// 5. One Newton-Raphson step (numerical Jacobian + Gaussian elimination).
///
/// # Arguments
///
/// - `g_grid`  — initial metric `[nx·ny·nz]`, each entry has `dim²` components
/// - `t_grid`  — stress-energy T_{μν} at each point, same flat indexing
/// - `nx`, `ny`, `nz` — grid dimensions (each ≥ 3)
/// - `h`       — uniform grid spacing (and FD step for derivatives)
/// - `kappa`   — coupling constant 8πG/c⁴
/// - `tol`     — convergence criterion
/// - `max_iter`— maximum NR iterations before giving up
/// - `eps`     — step size for the numerical Jacobian
pub fn solve_3d(
    g_grid: &[Vec<f64>],
    t_grid: &[Tensor<0, 2>],
    nx: usize,
    ny: usize,
    nz: usize,
    h: f64,
    kappa: f64,
    tol: f64,
    max_iter: usize,
    eps: f64,
) -> SolveResult3D {
    let n_points = nx * ny * nz;
    assert_eq!(g_grid.len(), n_points, "g_grid must have nx*ny*nz entries");
    assert_eq!(t_grid.len(), n_points, "t_grid must have nx*ny*nz entries");
    assert!(nx >= 3 && ny >= 3 && nz >= 3, "Need ≥ 3 points per axis");

    let dim2 = g_grid[0].len();
    let dim = (dim2 as f64).sqrt() as usize;
    assert_eq!(dim * dim, dim2, "Metric must have dim² components");
    assert!(dim >= 3, "dim must be ≥ 3 for the 3-D spatial solver");

    // Pre-enumerate interior cells in row-major order.
    let interior: Vec<(usize, usize, usize)> = (1..nx - 1)
        .flat_map(|ix| (1..ny - 1).flat_map(move |iy| (1..nz - 1).map(move |iz| (ix, iy, iz))))
        .collect();

    let identity: Vec<f64> = (0..dim2)
        .map(|k| if k / dim == k % dim { 1.0 } else { 0.0 })
        .collect();

    let mut grid = g_grid.to_vec();
    let mut iterations = 0usize;
    let mut residual_norm = f64::INFINITY;
    let mut converged = false;

    // +1 so we always evaluate the residual after the last step.
    for iter in 0..=max_iter {
        // ── 1. Symmetrize interior ────────────────────────────────────────
        for &(ix, iy, iz) in &interior {
            symmetrize(&mut grid[ix * ny * nz + iy * nz + iz], dim);
        }

        // ── 2. g_inv at every grid point ──────────────────────────────────
        let g_inv_grid: Vec<Vec<f64>> = grid
            .iter()
            .map(|g| invert_matrix(g, dim).unwrap_or_else(|| identity.clone()))
            .collect();

        // ── 3. Flatten interior unknowns ──────────────────────────────────
        let x0: Vec<f64> = interior
            .iter()
            .flat_map(|&(ix, iy, iz)| grid[ix * ny * nz + iy * nz + iz].iter().copied())
            .collect();

        // ── Build residual closure ────────────────────────────────────────
        let gi = g_inv_grid;
        let grid_snap = grid.clone();
        let interior_cl = interior.clone();

        let residual_fn = move |g_interior: &[f64]| -> Vec<f64> {
            // Flat-index helper (captures ny, nz via the move).
            let fi = |ix: usize, iy: usize, iz: usize| ix * ny * nz + iy * nz + iz;

            // Reconstruct full grid from interior unknowns.
            let mut full = grid_snap.clone();
            for (k, &(ix, iy, iz)) in interior_cl.iter().enumerate() {
                let off = k * dim2;
                let mut g = g_interior[off..off + dim2].to_vec();
                symmetrize(&mut g, dim);
                full[fi(ix, iy, iz)] = g;
            }

            // Residual at every interior point.
            // clear_tape() after each point keeps the AAD tape bounded.
            interior_cl
                .iter()
                .flat_map(|&(iix, iiy, iiz)| {
                    let g_fn = |x: &[f64]| -> Tensor<0, 2> {
                        let jx = ((x[0] / h).round() as usize).clamp(0, nx - 1);
                        let jy = ((x[1] / h).round() as usize).clamp(0, ny - 1);
                        let jz = ((x[2] / h).round() as usize).clamp(0, nz - 1);
                        Tensor::from_f64(dim, full[fi(jx, jy, jz)].clone())
                    };
                    let g_inv_fn = |x: &[f64]| -> Tensor<2, 0> {
                        let jx = ((x[0] / h).round() as usize).clamp(0, nx - 1);
                        let jy = ((x[1] / h).round() as usize).clamp(0, ny - 1);
                        let jz = ((x[2] / h).round() as usize).clamp(0, nz - 1);
                        Tensor::from_f64(dim, gi[fi(jx, jy, jz)].clone())
                    };

                    let mut point = vec![0.0f64; dim];
                    point[0] = iix as f64 * h;
                    point[1] = iiy as f64 * h;
                    point[2] = iiz as f64 * h;
                    // point[3..] remain 0 (static solution)

                    aad::no_tape(|| {
                        einstein_residual(
                            &g_fn,
                            &g_inv_fn,
                            &t_grid[fi(iix, iiy, iiz)],
                            &point,
                            h,
                            kappa,
                        )
                        .components
                        .iter()
                        .map(|c| c.result)
                        .collect::<Vec<f64>>()
                    })
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
            break;
        }

        // ── 5. Newton-Raphson step ────────────────────────────────────────
        let x_new = newton_step(&residual_fn, &x0, eps);

        for (k, &(ix, iy, iz)) in interior.iter().enumerate() {
            let off = k * dim2;
            grid[ix * ny * nz + iy * nz + iz] = x_new[off..off + dim2].to_vec();
        }

        iterations += 1;
    }

    SolveResult3D {
        g_grid: grid,
        nx,
        ny,
        nz,
        iterations,
        residual_norm,
        converged,
    }
}
