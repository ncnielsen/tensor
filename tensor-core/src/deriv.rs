use crate::tensor::Tensor;

/// Build the full Jacobian J[i][j] = dF_i/dx_j using reverse-mode autodiff.
///
/// Calls `reverse_pass(i)` for each output index `i`. The closure should:
/// 1. Zero the input shadow buffer
/// 2. Seed the output shadow with unit vector e_i
/// 3. Call the Enzyme-generated reverse function
/// 4. Return the input gradient vector
///
/// Returns flat row-major Jacobian: `J[i * n_inputs + j]`.
pub fn jacobian_from_reverse(
    n_inputs: usize,
    n_outputs: usize,
    reverse_pass: impl Fn(usize) -> Vec<f64>,
) -> Vec<f64> {
    let mut jac = vec![0.0; n_outputs * n_inputs];
    for i in 0..n_outputs {
        let grad = reverse_pass(i);
        assert_eq!(grad.len(), n_inputs, "gradient length mismatch");
        jac[i * n_inputs..(i + 1) * n_inputs].copy_from_slice(&grad);
    }
    jac
}

/// Build the full Jacobian using forward-mode autodiff.
///
/// Calls `forward_pass(j)` for each input index `j`. The closure should:
/// 1. Seed the input tangent with unit vector e_j
/// 2. Call the Enzyme-generated forward function
/// 3. Return the output tangent vector
///
/// Returns flat row-major Jacobian: `J[i * n_inputs + j]`.
pub fn jacobian_from_forward(
    n_inputs: usize,
    n_outputs: usize,
    forward_pass: impl Fn(usize) -> Vec<f64>,
) -> Vec<f64> {
    let mut jac = vec![0.0; n_outputs * n_inputs];
    for j in 0..n_inputs {
        let tangent = forward_pass(j);
        assert_eq!(tangent.len(), n_outputs, "tangent length mismatch");
        for i in 0..n_outputs {
            jac[i * n_inputs + j] = tangent[i];
        }
    }
    jac
}

/// Convert a flat Jacobian of a metric function into per-coordinate partial
/// derivative tensors.
///
/// The metric maps coordinates (length `dim`) to flat metric components
/// (length `dim * dim`, row-major). The Jacobian therefore has `dim * dim`
/// rows and `dim` columns.
///
/// Returns `dim` tensors: `result[k]` = partial g / partial x^k as a `Tensor<0,2>`.
pub fn metric_partials_from_jacobian(jacobian: &[f64], dim: usize) -> Vec<Tensor<0, 2>> {
    let n_out = dim * dim;
    assert_eq!(jacobian.len(), n_out * dim, "Jacobian size mismatch");

    (0..dim)
        .map(|k| {
            let mut dg = Tensor::<0, 2>::new(dim);
            for mu in 0..dim {
                for nu in 0..dim {
                    let row = mu * dim + nu;
                    dg.set_component(&[mu, nu], jacobian[row * dim + k]);
                }
            }
            dg
        })
        .collect()
}

/// Jacobian via central finite differences (for validation).
///
/// `f` maps a coordinate slice (length `n_inputs`) into an output buffer
/// (length `n_outputs`). Step size `h` controls accuracy (O(h^2) truncation).
pub fn jacobian_fd(
    f: impl Fn(&[f64], &mut [f64]),
    x: &[f64],
    n_outputs: usize,
    h: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut jac = vec![0.0; n_outputs * n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let mut fp = vec![0.0; n_outputs];
    let mut fm = vec![0.0; n_outputs];

    for j in 0..n {
        xp[j] = x[j] + h;
        xm[j] = x[j] - h;
        f(&xp, &mut fp);
        f(&xm, &mut fm);
        for i in 0..n_outputs {
            jac[i * n + j] = (fp[i] - fm[i]) / (2.0 * h);
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }

    jac
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::autodiff::autodiff_reverse;

    const TOL: f64 = 1e-10;
    const FD_TOL: f64 = 1e-6;

    // -- Enzyme: scalar gradient ---------------------------------------------

    #[autodiff_reverse(d_quadratic, Duplicated, Active)]
    fn quadratic(x: &[f64; 3]) -> f64 {
        x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]
    }

    #[test]
    fn enzyme_scalar_gradient() {
        let x = [1.0f64, 2.0, 3.0];
        let mut dx = [0.0f64; 3];
        let _val = d_quadratic(&x, &mut dx, 1.0);
        // grad = [2x0, 4x1, 6x2] = [2, 8, 18]
        assert!((dx[0] - 2.0).abs() < TOL);
        assert!((dx[1] - 8.0).abs() < TOL);
        assert!((dx[2] - 18.0).abs() < TOL);
    }

    // -- Enzyme: Schwarzschild metric ----------------------------------------

    #[autodiff_reverse(d_schwarzschild, Duplicated, Duplicated)]
    fn schwarzschild_metric(coords: &[f64; 4], out: &mut [f64; 16]) {
        let r = coords[1];
        let theta = coords[2];
        let f = 1.0 - 2.0 / r;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        out[0] = -f;                                    // g_tt
        out[5] = 1.0 / f;                               // g_rr
        out[10] = r * r;                                 // g_θθ
        out[15] = r * r * theta.sin() * theta.sin();     // g_φφ
    }

    #[test]
    fn enzyme_schwarzschild_partials() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];

        let jac = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0; 4];
            let mut out = [0.0; 16];
            let mut dout = [0.0; 16];
            dout[i] = 1.0;
            d_schwarzschild(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });

        let partials = metric_partials_from_jacobian(&jac, 4);

        let f = 1.0 - 2.0 / r;
        let sin_th = theta.sin();
        let cos_th = theta.cos();

        // dg_tt/dr = -2/r^2
        let dg_tt_dr = -2.0 / (r * r);
        // dg_rr/dr = -2/(r^2 * f^2)
        let dg_rr_dr = -2.0 / (r * r * f * f);
        // dg_thth/dr = 2r
        let dg_thth_dr = 2.0 * r;
        // dg_phph/dr = 2r sin^2(th)
        let dg_phph_dr = 2.0 * r * sin_th * sin_th;
        // dg_phph/dth = 2 r^2 sin(th) cos(th)
        let dg_phph_dth = 2.0 * r * r * sin_th * cos_th;

        // partials[0] = dg/dt = 0
        for mu in 0..4 {
            for nu in 0..4 {
                assert!(partials[0].component(&[mu, nu]).abs() < TOL);
            }
        }

        // partials[1] = dg/dr
        assert!((partials[1].component(&[0, 0]) - dg_tt_dr).abs() < TOL,
            "dg_tt/dr: {} vs {}", partials[1].component(&[0, 0]), dg_tt_dr);
        assert!((partials[1].component(&[1, 1]) - dg_rr_dr).abs() < TOL,
            "dg_rr/dr: {} vs {}", partials[1].component(&[1, 1]), dg_rr_dr);
        assert!((partials[1].component(&[2, 2]) - dg_thth_dr).abs() < TOL);
        assert!((partials[1].component(&[3, 3]) - dg_phph_dr).abs() < TOL);

        // partials[2] = dg/dtheta
        assert!((partials[2].component(&[3, 3]) - dg_phph_dth).abs() < TOL);

        // partials[3] = dg/dphi = 0
        for mu in 0..4 {
            for nu in 0..4 {
                assert!(partials[3].component(&[mu, nu]).abs() < TOL);
            }
        }
    }

    // -- Enzyme vs FD --------------------------------------------------------

    #[test]
    fn enzyme_vs_fd_schwarzschild() {
        let r = 10.0_f64;
        let theta = std::f64::consts::FRAC_PI_4;
        let coords = [0.0, r, theta, 0.0];

        let jac_enzyme = jacobian_from_reverse(4, 16, |i| {
            let mut dx = [0.0; 4];
            let mut out = [0.0; 16];
            let mut dout = [0.0; 16];
            dout[i] = 1.0;
            d_schwarzschild(&coords, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });

        let jac_fd = jacobian_fd(
            |x, out| {
                let mut c = [0.0; 4];
                c.copy_from_slice(&x[..4]);
                let o: &mut [f64; 16] = out.try_into().unwrap();
                schwarzschild_metric(&c, o);
            },
            &coords,
            16,
            1e-6,
        );

        for (i, (e, f)) in jac_enzyme.iter().zip(jac_fd.iter()).enumerate() {
            assert!(
                (e - f).abs() < FD_TOL,
                "Jacobian[{}]: enzyme={}, fd={}, diff={}",
                i, e, f, (e - f).abs()
            );
        }
    }

    // -- Deep chain ----------------------------------------------------------

    fn inner(x: &[f64; 2]) -> [f64; 2] {
        [x[0] * x[1], x[0] + x[1] * x[1]]
    }

    fn middle(x: &[f64; 2]) -> [f64; 2] {
        let y = inner(x);
        [y[0] * y[0] + y[1], y[0] - y[1] * y[1]]
    }

    #[autodiff_reverse(d_deep_chain, Duplicated, Duplicated)]
    fn deep_chain(x: &[f64; 2], out: &mut [f64; 2]) {
        let z = middle(x);
        out[0] = z[0].sin() + z[1];
        out[1] = z[0] * z[1];
    }

    #[test]
    fn enzyme_deep_chain_vs_fd() {
        let x = [1.5, 2.0];

        let jac_enzyme = jacobian_from_reverse(2, 2, |i| {
            let mut dx = [0.0; 2];
            let mut out = [0.0; 2];
            let mut dout = [0.0; 2];
            dout[i] = 1.0;
            d_deep_chain(&x, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });

        let jac_fd = jacobian_fd(
            |xs, os| {
                let mut c = [0.0; 2];
                c.copy_from_slice(&xs[..2]);
                let o: &mut [f64; 2] = os.try_into().unwrap();
                deep_chain(&c, o);
            },
            &x,
            2,
            1e-7,
        );

        for (i, (e, f)) in jac_enzyme.iter().zip(jac_fd.iter()).enumerate() {
            assert!(
                (e - f).abs() < FD_TOL,
                "Deep chain Jacobian[{}]: enzyme={}, fd={}, diff={}",
                i, e, f, (e - f).abs()
            );
        }
    }

    // -- Jacobian of vector function -----------------------------------------

    #[autodiff_reverse(d_vec_fn, Duplicated, Duplicated)]
    fn vec_fn(x: &[f64; 3], out: &mut [f64; 3]) {
        out[0] = x[0] * x[1] + x[2];
        out[1] = x[0] * x[0] + x[1] * x[2];
        out[2] = x[0] * x[1] * x[2];
    }

    #[test]
    fn enzyme_jacobian_analytic() {
        let x = [2.0, 3.0, 5.0];

        let jac = jacobian_from_reverse(3, 3, |i| {
            let mut dx = [0.0; 3];
            let mut out = [0.0; 3];
            let mut dout = [0.0; 3];
            dout[i] = 1.0;
            d_vec_fn(&x, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });

        // f0 = x0*x1 + x2   -> [x1, x0, 1]       = [3, 2, 1]
        // f1 = x0^2 + x1*x2 -> [2*x0, x2, x1]     = [4, 5, 3]
        // f2 = x0*x1*x2     -> [x1*x2, x0*x2, x0*x1] = [15, 10, 6]
        #[rustfmt::skip]
        let expected = [
            3.0,  2.0, 1.0,
            4.0,  5.0, 3.0,
            15.0, 10.0, 6.0,
        ];

        for (i, (e, a)) in jac.iter().zip(expected.iter()).enumerate() {
            assert!((e - a).abs() < TOL, "J[{}]: enzyme={}, analytic={}", i, e, a);
        }
    }

    #[test]
    fn enzyme_jacobian_vs_fd() {
        let x = [2.0, 3.0, 5.0];

        let jac_enzyme = jacobian_from_reverse(3, 3, |i| {
            let mut dx = [0.0; 3];
            let mut out = [0.0; 3];
            let mut dout = [0.0; 3];
            dout[i] = 1.0;
            d_vec_fn(&x, &mut dx, &mut out, &mut dout);
            dx.to_vec()
        });

        let jac_fd = jacobian_fd(
            |xs, os| {
                let mut c = [0.0; 3];
                c.copy_from_slice(&xs[..3]);
                let o: &mut [f64; 3] = os.try_into().unwrap();
                vec_fn(&c, o);
            },
            &x,
            3,
            1e-7,
        );

        for (i, (e, f)) in jac_enzyme.iter().zip(jac_fd.iter()).enumerate() {
            assert!(
                (e - f).abs() < FD_TOL,
                "J[{}]: enzyme={}, fd={}", i, e, f
            );
        }
    }

    // -- Pure infrastructure tests (no Enzyme) -------------------------------

    #[test]
    fn metric_partials_restructuring() {
        // 2D metric: Jacobian is 4 outputs x 2 inputs = 8 entries
        #[rustfmt::skip]
        let jac = vec![
            1.0, 2.0,  // dg_00/dx0, dg_00/dx1
            3.0, 4.0,  // dg_01/dx0, dg_01/dx1
            5.0, 6.0,  // dg_10/dx0, dg_10/dx1
            7.0, 8.0,  // dg_11/dx0, dg_11/dx1
        ];

        let partials = metric_partials_from_jacobian(&jac, 2);
        assert_eq!(partials.len(), 2);

        // partials[0] = dg/dx0
        assert_eq!(partials[0].component(&[0, 0]), 1.0);
        assert_eq!(partials[0].component(&[0, 1]), 3.0);
        assert_eq!(partials[0].component(&[1, 0]), 5.0);
        assert_eq!(partials[0].component(&[1, 1]), 7.0);

        // partials[1] = dg/dx1
        assert_eq!(partials[1].component(&[0, 0]), 2.0);
        assert_eq!(partials[1].component(&[0, 1]), 4.0);
        assert_eq!(partials[1].component(&[1, 0]), 6.0);
        assert_eq!(partials[1].component(&[1, 1]), 8.0);
    }

    #[test]
    fn fd_jacobian_standalone() {
        let x = [2.0, 3.0, 5.0];
        let jac = jacobian_fd(
            |xs, os| {
                let mut c = [0.0; 3];
                c.copy_from_slice(&xs[..3]);
                let mut o = [0.0; 3];
                vec_fn(&c, &mut o);
                os.copy_from_slice(&o);
            },
            &x,
            3,
            1e-7,
        );

        #[rustfmt::skip]
        let expected = [
            3.0,  2.0, 1.0,
            4.0,  5.0, 3.0,
            15.0, 10.0, 6.0,
        ];

        for (i, (g, a)) in jac.iter().zip(expected.iter()).enumerate() {
            assert!((g - a).abs() < FD_TOL, "FD J[{}]: got={}, expected={}", i, g, a);
        }
    }
}
