use tensor_core::deriv::jacobian_from_reverse;

// ---------------------------------------------------------------------------
// Gaussian elimination (runtime n×n)
// ---------------------------------------------------------------------------

/// Solve A x = b in-place via Gaussian elimination with partial pivoting.
/// Modifies `a` and `b`; returns `x`.
///
/// # Panics
/// If the system is singular (zero pivot).
fn gaussian_elim(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    assert_eq!(a.len(), n);

    for col in 0..n {
        // Partial pivot
        let (mut max_val, mut max_row) = (a[col][col].abs(), col);
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        assert!(max_val > 1e-14, "singular system at column {}", col);
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Newton step
// ---------------------------------------------------------------------------

/// Compute one Newton-Raphson step: x_new = x − J⁻¹ F(x).
///
/// - `x`: current iterate
/// - `residual(x, out)`: evaluates F(x) into `out`
/// - `reverse_pass(i)`: seeds output i with 1.0 and returns the gradient
///   of F_i w.r.t. inputs — i.e., row i of the Jacobian. This is the pattern
///   produced by Enzyme's `#[autodiff_reverse]` with `Duplicated` args.
///
/// The Jacobian J is assembled from `n` reverse passes and then used to solve
/// J Δx = −F(x) for the step Δx.
pub fn newton_step(
    x: &[f64],
    residual: &dyn Fn(&[f64], &mut [f64]),
    reverse_pass: &dyn Fn(usize) -> Vec<f64>,
) -> Vec<f64> {
    let n = x.len();

    // Evaluate F(x)
    let mut fx = vec![0.0; n];
    residual(x, &mut fx);

    // Jacobian J[i][j] = ∂F_i/∂x_j via n reverse passes
    let jac_flat = jacobian_from_reverse(n, n, reverse_pass);

    // Build mutable J for Gaussian elimination, rhs = -F(x)
    let mut j_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| jac_flat[i * n..(i + 1) * n].to_vec())
        .collect();
    let mut rhs: Vec<f64> = fx.iter().map(|v| -v).collect();

    // Solve J Δx = −F(x)
    let delta = gaussian_elim(&mut j_rows, &mut rhs);

    // x_new = x + Δx
    x.iter().zip(&delta).map(|(xi, di)| xi + di).collect()
}

// ---------------------------------------------------------------------------
// Solve (iterate Newton steps)
// ---------------------------------------------------------------------------

/// Find a root of F by Newton-Raphson iteration starting from `x0`.
///
/// Returns the solution when ‖F(x)‖∞ < `tol`, or an error after `max_iter`.
///
/// `reverse_pass(x, i)` must evaluate row i of the Jacobian at point `x`
/// (i.e., seed output i and return the gradient).
pub fn solve(
    x0: Vec<f64>,
    residual: &dyn Fn(&[f64], &mut [f64]),
    reverse_pass: &dyn Fn(&[f64], usize) -> Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    let n = x0.len();
    let mut x = x0;

    for iter in 0..max_iter {
        let mut fx = vec![0.0; n];
        residual(&x, &mut fx);

        let norm = fx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if norm < tol {
            return Ok(x);
        }

        x = newton_step(&x, residual, &|i| reverse_pass(&x, i));

        // Check for divergence
        if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(format!("diverged at iteration {}", iter));
        }
    }

    Err(format!("did not converge in {} iterations", max_iter))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::autodiff::autodiff_reverse;

    const TOL: f64 = 1e-12;

    // -- Scalar quadratic: F(x) = x² − 4, root x = 2 -----------------------

    #[autodiff_reverse(d_quadratic, Duplicated, Duplicated)]
    fn quadratic(x: &[f64; 1], out: &mut [f64; 1]) {
        out[0] = x[0] * x[0] - 4.0;
    }

    #[test]
    fn newton_quadratic_converges() {
        let result = solve(
            vec![1.0],
            &|x, out| {
                let xa = [x[0]];
                let mut oa = [0.0f64];
                quadratic(&xa, &mut oa);
                out[0] = oa[0];
            },
            &|x, i| {
                let xa = [x[0]];
                let mut dx = [0.0f64];
                let mut out = [0.0f64];
                let mut dout = [0.0f64];
                dout[i] = 1.0;
                d_quadratic(&xa, &mut dx, &mut out, &mut dout);
                dx.to_vec()
            },
            1e-12,
            20,
        )
        .expect("quadratic solver failed");

        assert!((result[0] - 2.0).abs() < TOL, "root = {}, expected 2", result[0]);
    }

    // -- 2D system: F = [x² + y² − 5, x − y], root (1.5811, 1.5811) ------

    #[autodiff_reverse(d_circle_line, Duplicated, Duplicated)]
    fn circle_line(x: &[f64; 2], out: &mut [f64; 2]) {
        out[0] = x[0] * x[0] + x[1] * x[1] - 5.0;
        out[1] = x[0] - x[1];
    }

    #[test]
    fn newton_2d_converges() {
        let result = solve(
            vec![1.0, 1.0],
            &|x, out| {
                let xa = [x[0], x[1]];
                let mut oa = [0.0f64; 2];
                circle_line(&xa, &mut oa);
                out.copy_from_slice(&oa);
            },
            &|x, i| {
                let xa = [x[0], x[1]];
                let mut dx = [0.0f64; 2];
                let mut out = [0.0f64; 2];
                let mut dout = [0.0f64; 2];
                dout[i] = 1.0;
                d_circle_line(&xa, &mut dx, &mut out, &mut dout);
                dx.to_vec()
            },
            1e-12,
            20,
        )
        .expect("2D solver failed");

        let expected = (5.0_f64 / 2.0).sqrt();
        assert!((result[0] - expected).abs() < 1e-10, "x = {}, expected {}", result[0], expected);
        assert!((result[1] - expected).abs() < 1e-10, "y = {}, expected {}", result[1], expected);
    }

    // -- Gaussian elimination standalone ------------------------------------

    #[test]
    fn gaussian_elim_3x3() {
        // 2x + y + z = 8
        // 4x + 3y + 3z = 22
        // -2x + y - z = -4
        // Solution: x=1, y=2, z=4
        let mut a = vec![
            vec![2.0, 1.0, 1.0],
            vec![4.0, 3.0, 3.0],
            vec![-2.0, 1.0, -1.0],
        ];
        let mut b = vec![8.0, 22.0, -4.0];
        let x = gaussian_elim(&mut a, &mut b);
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-12, "x[1] = {}", x[1]);
        assert!((x[2] - 4.0).abs() < 1e-12, "x[2] = {}", x[2]);
    }
}
