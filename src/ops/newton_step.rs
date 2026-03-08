/// One Newton-Raphson step for a system F(x) = 0.
///
/// Given a smooth function F: ℝⁿ → ℝⁿ and a current point x ∈ ℝⁿ, computes:
///
///   1. The Jacobian J = dF/dx numerically via central differences:
///      J[:,j] = (F(x + ε·eⱼ) − F(x − ε·eⱼ)) / (2ε)   (2n evaluations of F)
///
///   2. The Newton direction δx by solving the linear system J δx = −F(x)
///      using Gaussian elimination with partial pivoting.
///
///   3. The updated point x + δx.
///
/// Near-zero pivots (|pivot| < 1e-14) are skipped, so the update is zero for
/// directions in which the Jacobian is singular.
///
/// # Arguments
/// - `f`   — system to solve; must satisfy `f(x).len() == x.len()`
/// - `x`   — current iterate, length n
/// - `eps` — step for the numerical Jacobian (≈ 1e-5 for smooth F)
///
/// # Application to the Einstein field equations
///
/// Wrap `einstein_residual` in a closure to obtain the residual as a flat
/// `Vec<f64>`, then call `newton_step` to advance the metric one NR step:
///
/// ```ignore
/// let residual_fn = |g_flat: &[f64]| {
///     // reconstruct g_fn from g_flat, then:
///     einstein_residual(&g_fn, &g_inv_fn, &t, &point, h, kappa)
///         .components.iter().map(|c| c.result).collect()
/// };
/// let g_new = newton_step(&residual_fn, &g_flat, 1e-5);
/// ```
pub fn newton_step(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let fx = f(x);
    assert_eq!(fx.len(), n, "F must map ℝⁿ → ℝⁿ (got {} outputs for {} inputs)", fx.len(), n);

    // Jacobian: column j = (F(x + ε eⱼ) − F(x − ε eⱼ)) / (2ε).
    let two_eps_inv = 1.0 / (2.0 * eps);
    let mut j = vec![vec![0.0f64; n]; n];
    for col in 0..n {
        let mut xp = x.to_vec();
        xp[col] += eps;
        let fp = f(&xp);

        let mut xm = x.to_vec();
        xm[col] -= eps;
        let fm = f(&xm);

        for row in 0..n {
            j[row][col] = (fp[row] - fm[row]) * two_eps_inv;
        }
    }

    // Solve J δx = −F(x), then return x + δx.
    let rhs: Vec<f64> = fx.iter().map(|&r| -r).collect();
    let delta = solve_linear(&mut j, rhs);
    x.iter().zip(delta.iter()).map(|(&xi, &di)| xi + di).collect()
}

/// Gaussian elimination with partial pivoting.  Solves A x = b; modifies A in place.
fn solve_linear(a: &mut Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();

    for col in 0..n {
        // Find the row with the largest absolute value in this column (partial pivot).
        let pivot_row = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-14 {
            continue; // Singular direction; leave δx = 0 here.
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for c in col..n {
                let t = a[col][c];
                a[row][c] -= factor * t;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-14 {
            continue;
        }
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    x
}
