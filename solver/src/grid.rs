use crate::adm::{
    adm_rhs_vacuum, gauge_rhs, hamiltonian_constraint, AdmState,
    ExtrinsicCurvature, Gauge, GaugeDeriv,
};
use rayon::prelude::*;
use tensor_core::{
    christoffel::Christoffel,
    curvature::ChristoffelDerivative,
    metric::invert_metric,
    tensor::Tensor,
};

const DIM: usize = 3;

// Boundary band width. The grid always carries a 3-cell ghost zone on every
// face, which supports:
//   - 4th-order central first differences (reach ±2)
//   - 4th-order central second differences, diagonal and mixed (reach ±2)
//   - 6th-order Kreiss-Oliger dissipation (reach ±3)
// Interior cells: [BAND, n − BAND). Minimum n = 2·BAND + 1 = 7.
const BAND: usize = 3;

// Component layout within a single grid point (22 f64 total):
//   γ_{ij}:  [i*3+j] → offsets 0..9
//   K_{ij}:  [i*3+j] → offsets 9..18
//   α:                 offset  18
//   β^i:     [i]     → offsets 19..22
const FIELDS: usize = 22;
const OFF_GAMMA: usize = 0;
const OFF_K: usize = 9;
const OFF_ALPHA: usize = 18;
const OFF_BETA: usize = 19;

/// Spatial derivative accuracy.
///
/// `Second` matches the original 2nd-order central stencils (reach ±1).
/// `Fourth` uses standard 5-point central stencils (reach ±2) and, for ∂Γ,
/// switches from FD-of-FD to the analytic chain rule
/// (`ChristoffelDerivative::from_metric_analytic`), eliminating one layer of
/// truncation-error compounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdOrder {
    Second,
    Fourth,
}

// ---------------------------------------------------------------------------
// AdmGrid
// ---------------------------------------------------------------------------

/// Flat 3D spatial grid for ADM time evolution.
///
/// Stores 22 `f64` per point: γ_{ij} (9), K_{ij} (9), α (1), β^i (3).
/// Layout: `data[(ix * n * n + iy * n + iz) * 22 + field]`.
///
/// **Boundary band:** 2-cell ghost zone on every face (indices 0,1 and n-2,n-1).
/// Interior cells: `2 ≤ ix,iy,iz ≤ n-3`. Minimum `n = 5` (one interior point).
///
/// Geodesic evolution (`adm_step_rk4`) does not modify boundary cells.
pub struct AdmGrid {
    n: usize,
    h: f64,
    data: Vec<f64>,
}

impl AdmGrid {
    /// Create an all-zero grid with `n` points per side and spacing `h`.
    ///
    /// Requires `n >= 2·BAND + 1 = 7` so that the 3-cell ghost zone leaves at
    /// least one interior point.
    pub fn new(n: usize, h: f64) -> Self {
        assert!(
            n > 2 * BAND,
            "minimum grid size is {} (3-cell band needs ≥1 interior), got n={}",
            2 * BAND + 1, n
        );
        Self { n, h, data: vec![0.0; n * n * n * FIELDS] }
    }

    pub fn n(&self) -> usize { self.n }
    pub fn h(&self) -> f64 { self.h }
    pub fn raw(&self) -> &[f64] { &self.data }
    pub fn raw_mut(&mut self) -> &mut [f64] { &mut self.data }

    fn idx(&self, ix: usize, iy: usize, iz: usize, f: usize) -> usize {
        ((ix * self.n + iy) * self.n + iz) * FIELDS + f
    }

    pub fn gamma(&self, ix: usize, iy: usize, iz: usize) -> Tensor<0, 2> {
        let mut g = Tensor::<0, 2>::new(DIM);
        for i in 0..DIM {
            for j in 0..DIM {
                g.set_component(&[i, j], self.data[self.idx(ix, iy, iz, OFF_GAMMA + i * DIM + j)]);
            }
        }
        g
    }

    pub fn set_gamma(&mut self, ix: usize, iy: usize, iz: usize, gamma: &Tensor<0, 2>) {
        for i in 0..DIM {
            for j in 0..DIM {
                let idx = self.idx(ix, iy, iz, OFF_GAMMA + i * DIM + j);
                self.data[idx] = gamma.component(&[i, j]);
            }
        }
    }

    pub fn k_tensor(&self, ix: usize, iy: usize, iz: usize) -> ExtrinsicCurvature {
        let mut k = ExtrinsicCurvature::new(DIM);
        for i in 0..DIM {
            for j in 0..DIM {
                k.set_component(i, j, self.data[self.idx(ix, iy, iz, OFF_K + i * DIM + j)]);
            }
        }
        k
    }

    pub fn set_k_tensor(&mut self, ix: usize, iy: usize, iz: usize, k: &ExtrinsicCurvature) {
        for i in 0..DIM {
            for j in 0..DIM {
                let idx = self.idx(ix, iy, iz, OFF_K + i * DIM + j);
                self.data[idx] = k.component(i, j);
            }
        }
    }

    pub fn alpha_val(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.data[self.idx(ix, iy, iz, OFF_ALPHA)]
    }

    pub fn set_alpha_val(&mut self, ix: usize, iy: usize, iz: usize, alpha: f64) {
        let idx = self.idx(ix, iy, iz, OFF_ALPHA);
        self.data[idx] = alpha;
    }

    pub fn beta_val(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        [
            self.data[self.idx(ix, iy, iz, OFF_BETA)],
            self.data[self.idx(ix, iy, iz, OFF_BETA + 1)],
            self.data[self.idx(ix, iy, iz, OFF_BETA + 2)],
        ]
    }

    pub fn set_beta_val(&mut self, ix: usize, iy: usize, iz: usize, beta: [f64; 3]) {
        for i in 0..3 {
            let idx = self.idx(ix, iy, iz, OFF_BETA + i);
            self.data[idx] = beta[i];
        }
    }

    /// Set all interior points to flat space: γ = I, K = 0, α = 1, β = 0.
    pub fn init_flat_interior(&mut self) {
        let n = self.n;
        for ix in BAND..(n - BAND) {
            for iy in BAND..(n - BAND) {
                for iz in BAND..(n - BAND) {
                    let mut g = Tensor::<0, 2>::new(DIM);
                    for i in 0..DIM { g.set_component(&[i, i], 1.0); }
                    self.set_gamma(ix, iy, iz, &g);
                    self.set_k_tensor(ix, iy, iz, &ExtrinsicCurvature::new(DIM));
                    self.set_alpha_val(ix, iy, iz, 1.0);
                    self.set_beta_val(ix, iy, iz, [0.0; 3]);
                }
            }
        }
    }

    /// Set every grid point (including boundary) to flat space.
    pub fn init_flat_all(&mut self) {
        let n = self.n;
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let mut g = Tensor::<0, 2>::new(DIM);
                    for i in 0..DIM { g.set_component(&[i, i], 1.0); }
                    self.set_gamma(ix, iy, iz, &g);
                    self.set_k_tensor(ix, iy, iz, &ExtrinsicCurvature::new(DIM));
                    self.set_alpha_val(ix, iy, iz, 1.0);
                    self.set_beta_val(ix, iy, iz, [0.0; 3]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FD operators — dispatch on FdOrder for all spatial derivatives
// ---------------------------------------------------------------------------
//
// 2nd-order central first derivative:  (f[+1] − f[−1]) / 2h          reach ±1
// 4th-order central first derivative:  (−f[+2] + 8f[+1] − 8f[−1] + f[−2]) / 12h   reach ±2
//
// 2nd-order diagonal 2nd derivative:   (f[+1] − 2f[0] + f[−1]) / h²  reach ±1
// 4th-order diagonal 2nd derivative:   (−f[+2] + 16f[+1] − 30f[0] + 16f[−1] − f[−2]) / 12h²  reach ±2
//
// 2nd-order mixed 2nd derivative:      (f[+,+] − f[+,−] − f[−,+] + f[−,−]) / 4h²  reach ±1 × ±1
// 4th-order mixed 2nd derivative:      factored D_x⁴[D_y⁴[f]] / (144 h²)          reach ±2 × ±2

/// Central FD of γ in each spatial direction: `out[k]` = ∂_k γ_{ij}.
fn partial_gamma_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> [Tensor<0, 2>; 3] {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let gbase = |x: usize, y: usize, z: usize| ((x * n + y) * n + z) * FIELDS + OFF_GAMMA;

    let mut out = [Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM)];

    let dirs = [
        (gbase(ix + 2, iy, iz), gbase(ix + 1, iy, iz), gbase(ix - 1, iy, iz), gbase(ix - 2, iy, iz)),
        (gbase(ix, iy + 2, iz), gbase(ix, iy + 1, iz), gbase(ix, iy - 1, iz), gbase(ix, iy - 2, iz)),
        (gbase(ix, iy, iz + 2), gbase(ix, iy, iz + 1), gbase(ix, iy, iz - 1), gbase(ix, iy, iz - 2)),
    ];

    for (d, &(p2, p1, m1, m2)) in dirs.iter().enumerate() {
        let dst = out[d].as_mut_slice();
        match order {
            FdOrder::Second => {
                let denom = 2.0 * h;
                for c in 0..(DIM * DIM) {
                    dst[c] = (raw[p1 + c] - raw[m1 + c]) / denom;
                }
            }
            FdOrder::Fourth => {
                let denom = 12.0 * h;
                for c in 0..(DIM * DIM) {
                    dst[c] = (-raw[p2 + c] + 8.0 * raw[p1 + c] - 8.0 * raw[m1 + c] + raw[m2 + c]) / denom;
                }
            }
        }
    }
    out
}

/// Central FD of K_{ij} in each spatial direction. Same stencil/layout as `partial_gamma_at`.
fn partial_k_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> [Tensor<0, 2>; 3] {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let kbase = |x: usize, y: usize, z: usize| ((x * n + y) * n + z) * FIELDS + OFF_K;

    let mut out = [Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM)];

    let dirs = [
        (kbase(ix + 2, iy, iz), kbase(ix + 1, iy, iz), kbase(ix - 1, iy, iz), kbase(ix - 2, iy, iz)),
        (kbase(ix, iy + 2, iz), kbase(ix, iy + 1, iz), kbase(ix, iy - 1, iz), kbase(ix, iy - 2, iz)),
        (kbase(ix, iy, iz + 2), kbase(ix, iy, iz + 1), kbase(ix, iy, iz - 1), kbase(ix, iy, iz - 2)),
    ];

    for (d, &(p2, p1, m1, m2)) in dirs.iter().enumerate() {
        let dst = out[d].as_mut_slice();
        match order {
            FdOrder::Second => {
                let denom = 2.0 * h;
                for c in 0..(DIM * DIM) {
                    dst[c] = (raw[p1 + c] - raw[m1 + c]) / denom;
                }
            }
            FdOrder::Fourth => {
                let denom = 12.0 * h;
                for c in 0..(DIM * DIM) {
                    dst[c] = (-raw[p2 + c] + 8.0 * raw[p1 + c] - 8.0 * raw[m1 + c] + raw[m2 + c]) / denom;
                }
            }
        }
    }
    out
}

/// Central FD of the lapse α: `out[k]` = ∂_k α.
fn partial_alpha_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize, order: FdOrder) -> [f64; 3] {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let alpha_at = |x: usize, y: usize, z: usize| {
        raw[((x * n + y) * n + z) * FIELDS + OFF_ALPHA]
    };
    match order {
        FdOrder::Second => {
            let d = 2.0 * h;
            [
                (alpha_at(ix + 1, iy, iz) - alpha_at(ix - 1, iy, iz)) / d,
                (alpha_at(ix, iy + 1, iz) - alpha_at(ix, iy - 1, iz)) / d,
                (alpha_at(ix, iy, iz + 1) - alpha_at(ix, iy, iz - 1)) / d,
            ]
        }
        FdOrder::Fourth => {
            let d = 12.0 * h;
            [
                (-alpha_at(ix + 2, iy, iz) + 8.0 * alpha_at(ix + 1, iy, iz) - 8.0 * alpha_at(ix - 1, iy, iz) + alpha_at(ix - 2, iy, iz)) / d,
                (-alpha_at(ix, iy + 2, iz) + 8.0 * alpha_at(ix, iy + 1, iz) - 8.0 * alpha_at(ix, iy - 1, iz) + alpha_at(ix, iy - 2, iz)) / d,
                (-alpha_at(ix, iy, iz + 2) + 8.0 * alpha_at(ix, iy, iz + 1) - 8.0 * alpha_at(ix, iy, iz - 1) + alpha_at(ix, iy, iz - 2)) / d,
            ]
        }
    }
}

/// Second partials of the lapse: `out[i][j]` = ∂_i ∂_j α. Symmetric by construction.
fn partial2_alpha_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize, order: FdOrder) -> [[f64; 3]; 3] {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let alpha_at = |x: usize, y: usize, z: usize| {
        raw[((x * n + y) * n + z) * FIELDS + OFF_ALPHA]
    };

    let a0 = alpha_at(ix, iy, iz);
    let h_sq = h * h;

    let diag = |p2: f64, p1: f64, m1: f64, m2: f64| -> f64 {
        match order {
            FdOrder::Second => (p1 - 2.0 * a0 + m1) / h_sq,
            FdOrder::Fourth => (-p2 + 16.0 * p1 - 30.0 * a0 + 16.0 * m1 - m2) / (12.0 * h_sq),
        }
    };
    let d_xx = diag(alpha_at(ix + 2, iy, iz), alpha_at(ix + 1, iy, iz), alpha_at(ix - 1, iy, iz), alpha_at(ix - 2, iy, iz));
    let d_yy = diag(alpha_at(ix, iy + 2, iz), alpha_at(ix, iy + 1, iz), alpha_at(ix, iy - 1, iz), alpha_at(ix, iy - 2, iz));
    let d_zz = diag(alpha_at(ix, iy, iz + 2), alpha_at(ix, iy, iz + 1), alpha_at(ix, iy, iz - 1), alpha_at(ix, iy, iz - 2));

    let dxy = compute_mixed_2nd(alpha_at, ix, iy, iz, 0, 1, order, h_sq);
    let dxz = compute_mixed_2nd(alpha_at, ix, iy, iz, 0, 2, order, h_sq);
    let dyz = compute_mixed_2nd(alpha_at, ix, iy, iz, 1, 2, order, h_sq);

    [
        [d_xx, dxy, dxz],
        [dxy, d_yy, dyz],
        [dxz, dyz, d_zz],
    ]
}

/// Helper: mixed second derivative ∂_{axis_a} ∂_{axis_b} of a scalar field
/// accessible via `f_at(x, y, z)`. `axis_a` ≠ `axis_b` ∈ {0, 1, 2} = {x, y, z}.
#[allow(clippy::too_many_arguments)]
fn compute_mixed_2nd(
    f_at: impl Fn(usize, usize, usize) -> f64,
    ix: usize, iy: usize, iz: usize,
    axis_a: usize, axis_b: usize,
    order: FdOrder,
    h_sq: f64,
) -> f64 {
    match order {
        FdOrder::Second => {
            let pp = at2(ix, iy, iz, axis_a, 1, axis_b, 1, &f_at);
            let pm = at2(ix, iy, iz, axis_a, 1, axis_b, -1, &f_at);
            let mp = at2(ix, iy, iz, axis_a, -1, axis_b, 1, &f_at);
            let mm = at2(ix, iy, iz, axis_a, -1, axis_b, -1, &f_at);
            (pp - pm - mp + mm) / (4.0 * h_sq)
        }
        FdOrder::Fourth => {
            let dy_raw = |i_off: isize| -> f64 {
                -at2(ix, iy, iz, axis_a, i_off, axis_b, 2, &f_at)
                + 8.0 * at2(ix, iy, iz, axis_a, i_off, axis_b, 1, &f_at)
                - 8.0 * at2(ix, iy, iz, axis_a, i_off, axis_b, -1, &f_at)
                + at2(ix, iy, iz, axis_a, i_off, axis_b, -2, &f_at)
            };
            (-dy_raw(2) + 8.0 * dy_raw(1) - 8.0 * dy_raw(-1) + dy_raw(-2)) / (144.0 * h_sq)
        }
    }
}

/// Read a scalar field at (ix + off_a along axis_a, iy + off_b along axis_b).
#[allow(clippy::too_many_arguments)]
fn at2(
    ix: usize, iy: usize, iz: usize,
    axis_a: usize, off_a: isize,
    axis_b: usize, off_b: isize,
    f_at: &impl Fn(usize, usize, usize) -> f64,
) -> f64 {
    let (mut x, mut y, mut z) = (ix as isize, iy as isize, iz as isize);
    match axis_a {
        0 => x += off_a,
        1 => y += off_a,
        2 => z += off_a,
        _ => unreachable!(),
    }
    match axis_b {
        0 => x += off_b,
        1 => y += off_b,
        2 => z += off_b,
        _ => unreachable!(),
    }
    f_at(x as usize, y as usize, z as usize)
}

/// Shift Jacobian: `out[k][j]` = ∂_j β^k. Central FD per component per direction.
fn partial_beta_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize, order: FdOrder) -> [[f64; 3]; 3] {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let beta_at = |x: usize, y: usize, z: usize| -> [f64; 3] {
        let base = ((x * n + y) * n + z) * FIELDS + OFF_BETA;
        [raw[base], raw[base + 1], raw[base + 2]]
    };

    let mut out = [[0.0f64; 3]; 3];
    match order {
        FdOrder::Second => {
            let d = 2.0 * h;
            let bp_x = beta_at(ix + 1, iy, iz);
            let bm_x = beta_at(ix - 1, iy, iz);
            let bp_y = beta_at(ix, iy + 1, iz);
            let bm_y = beta_at(ix, iy - 1, iz);
            let bp_z = beta_at(ix, iy, iz + 1);
            let bm_z = beta_at(ix, iy, iz - 1);
            for k in 0..3 {
                out[k][0] = (bp_x[k] - bm_x[k]) / d;
                out[k][1] = (bp_y[k] - bm_y[k]) / d;
                out[k][2] = (bp_z[k] - bm_z[k]) / d;
            }
        }
        FdOrder::Fourth => {
            let d = 12.0 * h;
            let b2x = beta_at(ix + 2, iy, iz);
            let b1x = beta_at(ix + 1, iy, iz);
            let bm1x = beta_at(ix - 1, iy, iz);
            let bm2x = beta_at(ix - 2, iy, iz);
            let b2y = beta_at(ix, iy + 2, iz);
            let b1y = beta_at(ix, iy + 1, iz);
            let bm1y = beta_at(ix, iy - 1, iz);
            let bm2y = beta_at(ix, iy - 2, iz);
            let b2z = beta_at(ix, iy, iz + 2);
            let b1z = beta_at(ix, iy, iz + 1);
            let bm1z = beta_at(ix, iy, iz - 1);
            let bm2z = beta_at(ix, iy, iz - 2);
            for k in 0..3 {
                out[k][0] = (-b2x[k] + 8.0 * b1x[k] - 8.0 * bm1x[k] + bm2x[k]) / d;
                out[k][1] = (-b2y[k] + 8.0 * b1y[k] - 8.0 * bm1y[k] + bm2y[k]) / d;
                out[k][2] = (-b2z[k] + 8.0 * b1z[k] - 8.0 * bm1z[k] + bm2z[k]) / d;
            }
        }
    }
    out
}

/// Assemble the full `GaugeDeriv` at a grid point via FD on stored α and β.
fn gauge_deriv_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize, order: FdOrder) -> GaugeDeriv {
    GaugeDeriv {
        partial_alpha: partial_alpha_at(grid, ix, iy, iz, order),
        partial2_alpha: partial2_alpha_at(grid, ix, iy, iz, order),
        partial_beta: partial_beta_at(grid, ix, iy, iz, order),
    }
}

/// Christoffel symbols at a grid point from the FD metric derivatives.
fn christoffel_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize, order: FdOrder) -> Christoffel {
    let gamma = grid.gamma(ix, iy, iz);
    let gamma_inv = invert_metric(&gamma);
    let pg = partial_gamma_at(grid, ix, iy, iz, order);
    Christoffel::from_metric(&gamma_inv, &pg)
}

/// Second partials of γ: `out[m * DIM + n]` = ∂_m ∂_n γ_{ij} as a `Tensor<0,2>`.
///
/// Diagonal uses the 3-point (2nd) or 5-point (4th) stencil. Mixed uses the
/// 4-point cross (2nd) or the factored 16-point D⁴_x[D⁴_y] stencil (4th).
fn partial2_gamma_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> Vec<Tensor<0, 2>> {
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let h_sq = h * h;
    let gbase = |x: usize, y: usize, z: usize| ((x * n + y) * n + z) * FIELDS + OFF_GAMMA;
    let g = |x: usize, y: usize, z: usize, c: usize| raw[gbase(x, y, z) + c];

    let mut out: Vec<Tensor<0, 2>> = (0..DIM * DIM).map(|_| Tensor::<0, 2>::new(DIM)).collect();

    for m in 0..DIM {
        for n_dir in 0..DIM {
            let idx = m * DIM + n_dir;
            for c in 0..(DIM * DIM) {
                let val = if m == n_dir {
                    // Diagonal second derivative along axis m
                    let (p2, p1, m1, m2) = neighbors4(ix, iy, iz, m);
                    match order {
                        FdOrder::Second => {
                            (g(p1.0, p1.1, p1.2, c) - 2.0 * g(ix, iy, iz, c) + g(m1.0, m1.1, m1.2, c)) / h_sq
                        }
                        FdOrder::Fourth => {
                            (-g(p2.0, p2.1, p2.2, c) + 16.0 * g(p1.0, p1.1, p1.2, c)
                             - 30.0 * g(ix, iy, iz, c) + 16.0 * g(m1.0, m1.1, m1.2, c)
                             - g(m2.0, m2.1, m2.2, c)) / (12.0 * h_sq)
                        }
                    }
                } else {
                    // Mixed second derivative along axes (m, n_dir)
                    let f_at = |x: usize, y: usize, z: usize| g(x, y, z, c);
                    compute_mixed_2nd(f_at, ix, iy, iz, m, n_dir, order, h_sq)
                };
                out[idx].as_mut_slice()[c] = val;
            }
        }
    }
    out
}

/// Return neighbours at offsets ±1, ±2 along `axis` from `(ix, iy, iz)`.
#[allow(clippy::type_complexity)]
fn neighbors4(ix: usize, iy: usize, iz: usize, axis: usize) -> ((usize, usize, usize), (usize, usize, usize), (usize, usize, usize), (usize, usize, usize)) {
    let (p2, p1, m1, m2) = match axis {
        0 => (ix + 2, ix + 1, ix - 1, ix - 2),
        1 => (iy + 2, iy + 1, iy - 1, iy - 2),
        2 => (iz + 2, iz + 1, iz - 1, iz - 2),
        _ => unreachable!(),
    };
    let mk = |a_val: usize| match axis {
        0 => (a_val, iy, iz),
        1 => (ix, a_val, iz),
        2 => (ix, iy, a_val),
        _ => unreachable!(),
    };
    (mk(p2), mk(p1), mk(m1), mk(m2))
}

/// ∂_l Γ^i_{jk} at a grid point.
///
/// Dispatches on `order`:
/// - `Second`: FD-of-FD (2nd-order central FD of Christoffel at neighbours).
/// - `Fourth`: analytic chain rule via `from_metric_analytic` (∂γ + ∂²γ), no
///   FD-of-FD. This is the key accuracy win: one derivative layer, not two.
fn christoffel_deriv_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> ChristoffelDerivative {
    match order {
        FdOrder::Second => christoffel_deriv_fd(grid, ix, iy, iz),
        FdOrder::Fourth => christoffel_deriv_analytic_at(grid, ix, iy, iz, order),
    }
}

/// 2nd-order FD-of-FD: (Γ[+1] − Γ[−1]) / 2h.
fn christoffel_deriv_fd(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> ChristoffelDerivative {
    let h2 = 2.0 * grid.h();
    let d3 = DIM * DIM * DIM;
    let d2 = DIM * DIM;
    let mut data = vec![0.0f64; DIM.pow(4)];

    let directions = [
        (christoffel_at(grid, ix + 1, iy, iz, FdOrder::Second), christoffel_at(grid, ix - 1, iy, iz, FdOrder::Second), 0usize),
        (christoffel_at(grid, ix, iy + 1, iz, FdOrder::Second), christoffel_at(grid, ix, iy - 1, iz, FdOrder::Second), 1usize),
        (christoffel_at(grid, ix, iy, iz + 1, FdOrder::Second), christoffel_at(grid, ix, iy, iz - 1, FdOrder::Second), 2usize),
    ];

    for (gp, gm, l) in &directions {
        for i in 0..DIM {
            for j in 0..DIM {
                for k in 0..DIM {
                    data[i * d3 + j * d2 + k * DIM + l] =
                        (gp.component(i, j, k) - gm.component(i, j, k)) / h2;
                }
            }
        }
    }

    ChristoffelDerivative::from_flat(DIM, data)
}

/// Analytic ∂Γ from ∂γ + ∂²γ (4th-order FD inputs, single derivative layer).
fn christoffel_deriv_analytic_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> ChristoffelDerivative {
    let gamma = grid.gamma(ix, iy, iz);
    let gamma_inv = invert_metric(&gamma);
    let pg = partial_gamma_at(grid, ix, iy, iz, order);
    let p2g = partial2_gamma_at(grid, ix, iy, iz, order);
    ChristoffelDerivative::from_metric_analytic(&gamma_inv, &pg, &p2g)
}

// ---------------------------------------------------------------------------
// Christoffel cache — compute Γ once per point, reuse for center + ∂Γ
// ---------------------------------------------------------------------------

/// Dense flat index into the Christoffel cache, which covers the cube
/// `[BAND-1, n-(BAND-1))³ = [2, n-2)³` (interior plus the 1-cell halo that
/// 2nd-order FD of ∂Γ reads; 4th-order analytic ∂Γ reads only the point itself).
#[inline]
fn cache_idx(n: usize, ix: usize, iy: usize, iz: usize) -> usize {
    let m = n - 2 * (BAND - 1); // n - 4 for BAND=3
    let base = BAND - 1;        // 2 for BAND=3
    ((ix - base) * m + (iy - base)) * m + (iz - base)
}

/// Compute Christoffel symbols once at every point in `[2, n-2)³`.
fn christoffel_cache(grid: &AdmGrid, order: FdOrder) -> Vec<Christoffel> {
    let n = grid.n();
    let m = n - 2 * (BAND - 1);
    let base = BAND - 1;
    (0..m * m * m)
        .into_par_iter()
        .map(|idx| {
            let iz = idx % m + base;
            let iy = (idx / m) % m + base;
            let ix = idx / (m * m) + base;
            christoffel_at(grid, ix, iy, iz, order)
        })
        .collect()
}

/// 2nd-order FD-of-FD ∂Γ from cached Christoffels (reach ±1 in cache).
fn christoffel_deriv_cached(
    cache: &[Christoffel],
    n: usize,
    h: f64,
    ix: usize, iy: usize, iz: usize,
) -> ChristoffelDerivative {
    let h2 = 2.0 * h;
    let d3 = DIM * DIM * DIM;
    let d2 = DIM * DIM;
    let mut data = vec![0.0f64; DIM.pow(4)];

    let directions = [
        (&cache[cache_idx(n, ix + 1, iy, iz)], &cache[cache_idx(n, ix - 1, iy, iz)], 0usize),
        (&cache[cache_idx(n, ix, iy + 1, iz)], &cache[cache_idx(n, ix, iy - 1, iz)], 1usize),
        (&cache[cache_idx(n, ix, iy, iz + 1)], &cache[cache_idx(n, ix, iy, iz - 1)], 2usize),
    ];

    for (gp, gm, l) in directions {
        for i in 0..DIM {
            for j in 0..DIM {
                for k in 0..DIM {
                    data[i * d3 + j * d2 + k * DIM + l] =
                        (gp.component(i, j, k) - gm.component(i, j, k)) / h2;
                }
            }
        }
    }

    ChristoffelDerivative::from_flat(DIM, data)
}

// ---------------------------------------------------------------------------
// Kreiss-Oliger 6th-order dissipation
// ---------------------------------------------------------------------------

/// 6th-order Kreiss-Oliger dissipation operator applied to a single scalar
/// field at (ix, iy, iz). Returns ε h⁵ (D₊D₋)³ u summed over the 3 axes,
/// where (D₊D₋)³ has the 7-point stencil [1, −6, 15, −20, 15, −6, 1].
///
/// Damps high-frequency modes (k ~ π/h) while leaving smooth modes
/// (k << 1/h) unaffected at O(h⁵). Reach ±3 per axis.
fn ko_dissipation_field(
    grid: &AdmGrid,
    field_offset: usize,
    ix: usize, iy: usize, iz: usize,
    eps_ko: f64,
) -> f64 {
    if eps_ko == 0.0 {
        return 0.0;
    }
    let n = grid.n();
    let h = grid.h();
    let raw = grid.raw();
    let at = |x: usize, y: usize, z: usize| {
        raw[((x * n + y) * n + z) * FIELDS + field_offset]
    };

    let d6 = |f3: f64, f2: f64, f1: f64, f0: f64, fm1: f64, fm2: f64, fm3: f64| {
        f3 - 6.0 * f2 + 15.0 * f1 - 20.0 * f0 + 15.0 * fm1 - 6.0 * fm2 + fm3
    };

    let d6_x = d6(
        at(ix + 3, iy, iz), at(ix + 2, iy, iz), at(ix + 1, iy, iz), at(ix, iy, iz),
        at(ix - 1, iy, iz), at(ix - 2, iy, iz), at(ix - 3, iy, iz),
    );
    let d6_y = d6(
        at(ix, iy + 3, iz), at(ix, iy + 2, iz), at(ix, iy + 1, iz), at(ix, iy, iz),
        at(ix, iy - 1, iz), at(ix, iy - 2, iz), at(ix, iy - 3, iz),
    );
    let d6_z = d6(
        at(ix, iy, iz + 3), at(ix, iy, iz + 2), at(ix, iy, iz + 1), at(ix, iy, iz),
        at(ix, iy, iz - 1), at(ix, iy, iz - 2), at(ix, iy, iz - 3),
    );

    eps_ko * h.powi(5) * (d6_x + d6_y + d6_z)
}

// ---------------------------------------------------------------------------
// Grid-level RHS
// ---------------------------------------------------------------------------

/// Evaluate the general-gauge vacuum ADM RHS at every interior point.
///
/// Writes the full 22-field RHS: ∂_t γ_{ij} (9), ∂_t K_{ij} (9), ∂_t α (1),
/// ∂_t β^i (3). Uses `adm_rhs_vacuum` for the metric/extrinsic-curvature
/// evolution and `gauge_rhs` for the gauge variables. KO dissipation is
/// applied to γ and K (the dynamical fields).
fn vacuum_rhs(grid: &AdmGrid, gauge: Gauge, order: FdOrder, eps_ko: f64) -> AdmGrid {
    let n = grid.n();
    let h = grid.h();
    let mut rhs = AdmGrid::new(n, h);

    let cache = christoffel_cache(grid, order);

    let m = n - 2 * BAND;
    let results: Vec<(usize, [f64; FIELDS])> = (0..m * m * m)
        .into_par_iter()
        .map(|idx| {
            let iz = idx % m + BAND;
            let iy = (idx / m) % m + BAND;
            let ix = idx / (m * m) + BAND;

            let gamma = grid.gamma(ix, iy, iz);
            let k = grid.k_tensor(ix, iy, iz);
            let alpha = grid.alpha_val(ix, iy, iz);
            let beta = grid.beta_val(ix, iy, iz);
            let state = AdmState::new(gamma, k, alpha, beta);

            let ch = &cache[cache_idx(n, ix, iy, iz)];
            let dch = christoffel_deriv_for_order(grid, &cache, n, h, ix, iy, iz, order);
            let pg = partial_gamma_at(grid, ix, iy, iz, order);
            let pk = partial_k_at(grid, ix, iy, iz, order);
            let gd = gauge_deriv_at(grid, ix, iy, iz, order);

            let (gamma_dot, k_dot) = adm_rhs_vacuum(&state, ch, &dch, &pg, &pk, &gd);
            let (alpha_dot, beta_dot) = gauge_rhs(&state, gauge);

            let base = ((ix * n + iy) * n + iz) * FIELDS;
            let mut row = [0.0f64; FIELDS];
            row[..9].copy_from_slice(gamma_dot.as_slice());
            row[9..18].copy_from_slice(k_dot.as_slice());
            row[OFF_ALPHA] = alpha_dot;
            row[OFF_BETA..OFF_BETA + 3].copy_from_slice(&beta_dot);

            if eps_ko != 0.0 {
                for c in 0..9 {
                    row[c] += ko_dissipation_field(grid, OFF_GAMMA + c, ix, iy, iz, eps_ko);
                    row[OFF_K + c] += ko_dissipation_field(grid, OFF_K + c, ix, iy, iz, eps_ko);
                }
            }

            (base, row)
        })
        .collect();

    let raw = rhs.raw_mut();
    for (base, row) in results {
        raw[base..base + FIELDS].copy_from_slice(&row);
    }

    rhs
}

/// Dispatch ∂Γ computation based on `order`:
/// - `Second`: FD-of-FD from the cached Christoffels (reach ±1 in cache).
/// - `Fourth`: analytic chain rule at the point (reach ±2 in γ, no cache needed
///   for ∂Γ itself, but Γ is read from the cache for the Riemann tensor).
#[allow(clippy::too_many_arguments)]
fn christoffel_deriv_for_order(
    grid: &AdmGrid,
    cache: &[Christoffel],
    n: usize,
    h: f64,
    ix: usize, iy: usize, iz: usize,
    order: FdOrder,
) -> ChristoffelDerivative {
    match order {
        FdOrder::Second => christoffel_deriv_cached(cache, n, h, ix, iy, iz),
        FdOrder::Fourth => christoffel_deriv_analytic_at(grid, ix, iy, iz, order),
    }
}

/// Return a new grid equal to `base + scale * delta` (elementwise on raw data).
fn scaled_add(base: &AdmGrid, scale: f64, delta: &AdmGrid) -> AdmGrid {
    let mut result = AdmGrid::new(base.n(), base.h());
    let b = base.raw();
    let d = delta.raw();
    result
        .raw_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, r)| *r = b[i] + scale * d[i]);
    result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Advance `grid` by one 4th-order Runge-Kutta step of size `dt`.
///
/// Uses geodesic slicing (α = 1, β = 0) with 2nd-order FD and no dissipation.
/// Equivalent to `adm_step_rk4_with_gauge(grid, dt, Gauge::Geodesic, FdOrder::Second, 0.0)`.
/// Boundary cells are frozen.
pub fn adm_step_rk4(grid: &mut AdmGrid, dt: f64) {
    adm_step_rk4_with_gauge(grid, dt, Gauge::Geodesic, FdOrder::Second, 0.0);
}

/// Advance `grid` by one 4th-order Runge-Kutta step of size `dt`.
///
/// - `gauge`: how α and β evolve (see [`Gauge`]).
/// - `order`: spatial FD accuracy for ∂γ, ∂K, ∂Γ (see [`FdOrder`]). `Fourth`
///   uses the analytic ∂Γ (no FD-of-FD) and 5-point stencils for first
///   derivatives.
/// - `eps_ko`: 6th-order Kreiss-Oliger dissipation strength. 0 disables it;
///   typical values are 0.01–0.5. Applied to γ and K only.
///
/// Boundary cells (the `BAND`-cell ghost zone) are frozen.
pub fn adm_step_rk4_with_gauge(
    grid: &mut AdmGrid,
    dt: f64,
    gauge: Gauge,
    order: FdOrder,
    eps_ko: f64,
) {
    let k1 = vacuum_rhs(grid, gauge, order, eps_ko);

    let y2 = scaled_add(grid, dt / 2.0, &k1);
    let k2 = vacuum_rhs(&y2, gauge, order, eps_ko);

    let y3 = scaled_add(grid, dt / 2.0, &k2);
    let k3 = vacuum_rhs(&y3, gauge, order, eps_ko);

    let y4 = scaled_add(grid, dt, &k3);
    let k4 = vacuum_rhs(&y4, gauge, order, eps_ko);

    let (r1, r2, r3, r4) = (k1.raw(), k2.raw(), k3.raw(), k4.raw());
    grid.raw_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, v)| {
            *v += dt / 6.0 * (r1[i] + 2.0 * r2[i] + 2.0 * r3[i] + r4[i]);
        });
}

/// RMS of the Hamiltonian constraint over all interior points.
///
/// Vanishes for physically consistent data; grows when constraints are violated.
pub fn hamiltonian_l2(grid: &AdmGrid, order: FdOrder) -> f64 {
    let n = grid.n();
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for ix in BAND..(n - BAND) {
        for iy in BAND..(n - BAND) {
            for iz in BAND..(n - BAND) {
                let gamma = grid.gamma(ix, iy, iz);
                let k = grid.k_tensor(ix, iy, iz);
                let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
                let ch = christoffel_at(grid, ix, iy, iz, order);
                let dch = christoffel_deriv_at(grid, ix, iy, iz, order);

                let h_val = hamiltonian_constraint(&state, &ch, &dch);
                sum_sq += h_val * h_val;
                count += 1;
            }
        }
    }

    if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    // -- Flat space: zero RHS, zero constraint, no drift --------------------

    #[test]
    fn flat_space_no_drift() {
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();

        let h0 = hamiltonian_l2(&grid, FdOrder::Second);
        assert!(h0.abs() < TOL, "initial H = {} ≠ 0", h0);

        let dt = 1e-4;
        for _ in 0..100 {
            adm_step_rk4(&mut grid, dt);
        }

        let g = grid.gamma(3, 3, 3);
        for i in 0..DIM {
            for j in 0..DIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g.component(&[i, j]) - expected).abs() < TOL,
                    "γ_{}{} = {}, expected {} after 100 steps", i, j, g.component(&[i, j]), expected
                );
            }
        }

        let h_final = hamiltonian_l2(&grid, FdOrder::Second);
        assert!(h_final.abs() < TOL, "final H = {} ≠ 0", h_final);
    }

    // -- Boundary cells unchanged after evolution ----------------------------

    #[test]
    fn boundary_cells_unchanged() {
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();

        let g_before = grid.gamma(0, 0, 0);
        let k_before = grid.k_tensor(0, 0, 0);

        adm_step_rk4(&mut grid, 0.01);

        let g_after = grid.gamma(0, 0, 0);
        let k_after = grid.k_tensor(0, 0, 0);

        for i in 0..DIM {
            for j in 0..DIM {
                assert_eq!(g_before.component(&[i, j]), g_after.component(&[i, j]),
                    "boundary γ_{}{} changed", i, j);
                assert_eq!(k_before.component(i, j), k_after.component(i, j),
                    "boundary K_{}{} changed", i, j);
            }
        }
    }

    // -- Isotropic K: ∂_t γ and ∂_t K match analytic ------------------------
    //
    // Uniform γ = I, K = ε I, geodesic slicing: ∂_t γ_{ij} = −2ε δ_{ij},
    // ∂_t K_{ij} = ε² δ_{ij}. Uniform fields → FD gives exact zero spatial
    // derivatives, so the grid RHS matches the point-wise analytic value.

    #[test]
    fn isotropic_k_rhs_analytic() {
        let eps = 0.1f64;
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();
        for ix in 0..7 {
            for iy in 0..7 {
                for iz in 0..7 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let rhs = vacuum_rhs(&grid, Gauge::Geodesic, FdOrder::Second, 0.0);

        let gd = rhs.gamma(3, 3, 3);
        let kd = rhs.k_tensor(3, 3, 3);

        for i in 0..DIM {
            for j in 0..DIM {
                let expected_gd = if i == j { -2.0 * eps } else { 0.0 };
                let expected_kd = if i == j { eps * eps } else { 0.0 };
                assert!((gd.component(&[i, j]) - expected_gd).abs() < TOL,
                    "∂_t γ_{}{} = {}, expected {}", i, j, gd.component(&[i, j]), expected_gd);
                assert!((kd.component(i, j) - expected_kd).abs() < TOL,
                    "∂_t K_{}{} = {}, expected {}", i, j, kd.component(i, j), expected_kd);
            }
        }
    }

    // -- RK4 convergence order (point-wise ODE, no grid) ---------------------

    #[test]
    fn rk4_convergence_order() {
        use crate::adm::adm_rhs_geodesic;
        use tensor_core::{christoffel::Christoffel, curvature::ChristoffelDerivative};

        let eps = 0.05f64;
        let gamma_exact = |t: f64| (1.0 - 3.0 * eps * t).powf(2.0 / 3.0);
        let k_exact = |t: f64| eps * (1.0 - 3.0 * eps * t).powf(-1.0 / 3.0);

        let point_rhs = |g: f64, k: f64| -> (f64, f64) {
            let mut gamma = Tensor::<0, 2>::new(DIM);
            for i in 0..DIM { gamma.set_component(&[i, i], g); }
            let mut kmat = ExtrinsicCurvature::new(DIM);
            for i in 0..DIM { kmat.set_component(i, i, k); }
            let state = AdmState::new(gamma, kmat, 1.0, [0.0; 3]);
            let ch = Christoffel::new(DIM);
            let dch = ChristoffelDerivative::new(DIM);
            let (gd, kd) = adm_rhs_geodesic(&state, &ch, &dch);
            (gd.component(&[0, 0]), kd.component(0, 0))
        };

        let rk4_step = |g: f64, k: f64, dt: f64| -> (f64, f64) {
            let (dg1, dk1) = point_rhs(g, k);
            let (dg2, dk2) = point_rhs(g + dt/2.0*dg1, k + dt/2.0*dk1);
            let (dg3, dk3) = point_rhs(g + dt/2.0*dg2, k + dt/2.0*dk2);
            let (dg4, dk4) = point_rhs(g + dt*dg3, k + dt*dk3);
            (g + dt/6.0*(dg1 + 2.0*dg2 + 2.0*dg3 + dg4),
             k + dt/6.0*(dk1 + 2.0*dk2 + 2.0*dk3 + dk4))
        };

        let dt1 = 0.1f64;
        let dt2 = dt1 / 2.0;
        let (g1, k1_val) = rk4_step(1.0, eps, dt1);
        let err1_gamma = (g1 - gamma_exact(dt1)).abs();
        let err1_k = (k1_val - k_exact(dt1)).abs();
        let (g2a, k2a) = rk4_step(1.0, eps, dt2);
        let (g2b, k2b) = rk4_step(g2a, k2a, dt2);
        let err2_gamma = (g2b - gamma_exact(dt1)).abs();
        let err2_k = (k2b - k_exact(dt1)).abs();

        assert!(err1_gamma / err2_gamma >= 10.0,
            "γ RK4 convergence: err(dt)={:.2e}, err(dt/2)={:.2e}, ratio={:.1}",
            err1_gamma, err2_gamma, err1_gamma / err2_gamma);
        assert!(err1_k / err2_k >= 10.0,
            "K RK4 convergence: err(dt)={:.2e}, err(dt/2)={:.2e}, ratio={:.1}",
            err1_k, err2_k, err1_k / err2_k);
    }

    // -- 2nd vs 4th order FD: both agree on uniform fields -------------------

    #[test]
    fn fd4_matches_fd2_on_uniform_fields() {
        let eps = 0.05f64;
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();
        for ix in 0..7 {
            for iy in 0..7 {
                for iz in 0..7 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let rhs2 = vacuum_rhs(&grid, Gauge::Geodesic, FdOrder::Second, 0.0);
        let rhs4 = vacuum_rhs(&grid, Gauge::Geodesic, FdOrder::Fourth, 0.0);

        // On uniform fields, both orders give the same RHS (spatial derivs = 0).
        for c in 0..FIELDS {
            let a = rhs2.raw()[((3 * 7 + 3) * 7 + 3) * FIELDS + c];
            let b = rhs4.raw()[((3 * 7 + 3) * 7 + 3) * FIELDS + c];
            assert!((a - b).abs() < TOL, "field {}: fd2={}, fd4={}", c, a, b);
        }
    }

    // -- 4th-order spatial convergence: ∂_x γ on a sinusoidal metric ---------
    //
    // γ_{00}(x) = 1 + ε sin(kx), other components = δ_{ij}. The analytic
    // ∂_x γ_{00} = εk cos(kx). We compare the FD ∂_x γ_{00} at the domain
    // centre for two grids (h and h/2) and check the error ratio:
    //   2nd-order: ratio ≈ 4  (error ∝ h²)
    //   4th-order: ratio ≈ 16 (error ∝ h⁴)

    #[test]
    fn fd_spatial_convergence_order() {
        let eps = 0.01f64;
        let k_wave = 2.0 * std::f64::consts::PI; // one full period on [0, 1]

        let setup_grid = |n: usize| -> AdmGrid {
            let h = 1.0 / (n - 1) as f64;
            let mut grid = AdmGrid::new(n, h);
            grid.init_flat_all();
            for ix in 0..n {
                let x = ix as f64 * h;
                let mut g = Tensor::<0, 2>::new(DIM);
                for i in 0..DIM { g.set_component(&[i, i], 1.0); }
                g.set_component(&[0, 0], 1.0 + eps * (k_wave * x).sin());
                for iy in 0..n {
                    for iz in 0..n {
                        grid.set_gamma(ix, iy, iz, &g);
                    }
                }
            }
            grid
        };

        let eval_dx_gamma00 = |n: usize, order: FdOrder| -> f64 {
            let grid = setup_grid(n);
            let center = n / 2;
            let pg = partial_gamma_at(&grid, center, center, center, order);
            // ∂_x γ_{00} is pg[0].component([0, 0])
            pg[0].component(&[0, 0])
        };

        let analytic = |n: usize| -> f64 {
            let h = 1.0 / (n - 1) as f64;
            let x = (n / 2) as f64 * h;
            eps * k_wave * (k_wave * x).cos()
        };

        // n=7 (h=1/6) and n=13 (h=1/12): ratio h1/h2 = 2.
        for order in [FdOrder::Second, FdOrder::Fourth] {
            let fd_7 = eval_dx_gamma00(7, order);
            let fd_13 = eval_dx_gamma00(13, order);
            let an_7 = analytic(7);
            let an_13 = analytic(13);
            let err_7 = (fd_7 - an_7).abs();
            let err_13 = (fd_13 - an_13).abs();
            let ratio = err_7 / err_13;

            let (expected_ratio, label) = match order {
                FdOrder::Second => (4.0, "2nd"),
                FdOrder::Fourth => (16.0, "4th"),
            };
            // Allow slack: ratio should be at least 60% of theoretical.
            assert!(
                ratio >= 0.6 * expected_ratio,
                "{}-order convergence: err(h)={:.2e}, err(h/2)={:.2e}, ratio={:.1} (expected ~{:.0})",
                label, err_7, err_13, ratio, expected_ratio
            );
        }
    }

    // -- 1+log at t=0 on uniform isotropic K: analytic ∂_t α -----------------

    #[test]
    fn one_plus_log_rhs_analytic_at_t0() {
        let eps = 0.1f64;
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();
        for ix in 0..7 {
            for iy in 0..7 {
                for iz in 0..7 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let rhs = vacuum_rhs(&grid, Gauge::OnePlusLog, FdOrder::Second, 0.0);

        let alpha_dot = rhs.alpha_val(3, 3, 3);
        assert!((alpha_dot - (-6.0 * eps)).abs() < TOL,
            "∂_t α = {}, expected -6ε = {}", alpha_dot, -6.0 * eps);

        let beta_dot = rhs.beta_val(3, 3, 3);
        for (i, &bd) in beta_dot.iter().enumerate() {
            assert!(bd.abs() < TOL, "∂_t β[{}] = {}, expected 0", i, bd);
        }

        let gd = rhs.gamma(3, 3, 3);
        let kd = rhs.k_tensor(3, 3, 3);
        for i in 0..DIM {
            for j in 0..DIM {
                let expected_gd = if i == j { -2.0 * eps } else { 0.0 };
                let expected_kd = if i == j { eps * eps } else { 0.0 };
                assert!((gd.component(&[i, j]) - expected_gd).abs() < TOL,
                    "∂_t γ_{}{} = {}, expected {}", i, j, gd.component(&[i, j]), expected_gd);
                assert!((kd.component(i, j) - expected_kd).abs() < TOL,
                    "∂_t K_{}{} = {}, expected {}", i, j, kd.component(i, j), expected_kd);
            }
        }
    }

    // -- 1+log actually decreases α when K > 0 --------------------------------

    #[test]
    fn one_plus_log_decreases_alpha_in_time() {
        let eps = 0.05f64;
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();
        for ix in 3..4 {
            for iy in 3..4 {
                for iz in 3..4 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let dt = 1e-3;
        for _ in 0..20 {
            adm_step_rk4_with_gauge(&mut grid, dt, Gauge::OnePlusLog, FdOrder::Second, 0.0);
        }

        let alpha = grid.alpha_val(3, 3, 3);
        assert!(alpha < 1.0 - 1e-6,
            "1+log should have decreased α below 1, got α = {}", alpha);
        assert!(alpha > 0.0, "α should still be positive, got {}", alpha);
    }

    // -- 1+log flat space (K=0): α frozen at 1, no drift ---------------------

    #[test]
    fn one_plus_log_flat_space_no_drift() {
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();

        let h0 = hamiltonian_l2(&grid, FdOrder::Second);
        assert!(h0.abs() < TOL, "initial H = {} ≠ 0", h0);

        let dt = 1e-4;
        for _ in 0..100 {
            adm_step_rk4_with_gauge(&mut grid, dt, Gauge::OnePlusLog, FdOrder::Second, 0.0);
        }

        let alpha = grid.alpha_val(3, 3, 3);
        assert!((alpha - 1.0).abs() < TOL, "α = {}, expected 1.0", alpha);

        let g = grid.gamma(3, 3, 3);
        for i in 0..DIM {
            for j in 0..DIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((g.component(&[i, j]) - expected).abs() < TOL,
                    "γ_{}{} = {}, expected {}", i, j, g.component(&[i, j]), expected);
            }
        }

        let h_final = hamiltonian_l2(&grid, FdOrder::Second);
        assert!(h_final.abs() < TOL, "final H = {} ≠ 0", h_final);
    }

    // -- Boundary cells (including α, β) stay frozen under 1+log -------------

    #[test]
    fn one_plus_log_boundary_cells_unchanged() {
        let eps = 0.05f64;
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();
        for ix in 3..4 {
            for iy in 3..4 {
                for iz in 3..4 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let g_before = grid.gamma(0, 0, 0);
        let k_before = grid.k_tensor(0, 0, 0);
        let alpha_before = grid.alpha_val(0, 0, 0);
        let beta_before = grid.beta_val(0, 0, 0);

        adm_step_rk4_with_gauge(&mut grid, 0.01, Gauge::OnePlusLog, FdOrder::Second, 0.0);

        let g_after = grid.gamma(0, 0, 0);
        let k_after = grid.k_tensor(0, 0, 0);
        let alpha_after = grid.alpha_val(0, 0, 0);
        let beta_after = grid.beta_val(0, 0, 0);

        for i in 0..DIM {
            for j in 0..DIM {
                assert_eq!(g_before.component(&[i, j]), g_after.component(&[i, j]),
                    "boundary γ_{}{} changed", i, j);
                assert_eq!(k_before.component(i, j), k_after.component(i, j),
                    "boundary K_{}{} changed", i, j);
            }
        }
        assert_eq!(alpha_before, alpha_after, "boundary α changed");
        for (i, &b) in beta_before.iter().enumerate() {
            assert_eq!(b, beta_after[i], "boundary β[{}] changed", i);
        }
    }

    // -- KO dissipation damps high-frequency noise ---------------------------
    //
    // Start with flat space + a high-frequency perturbation in K_{00}. Without
    // dissipation, the perturbation persists (or grows via nonlinear coupling).
    // With KO dissipation (eps_ko > 0), the perturbation should decay faster.

    #[test]
    fn ko_dissipation_damps_high_frequency() {
        let n = 11;
        let h = 0.1;
        let dt = 1e-4;
        let steps = 200;

        // High-frequency perturbation: alternate sign every cell (Nyquist mode).
        let perturbation = 0.01f64;

        let make_grid = || {
            let mut grid = AdmGrid::new(n, h);
            grid.init_flat_all();
            for ix in BAND..(n - BAND) {
                for iy in BAND..(n - BAND) {
                    for iz in BAND..(n - BAND) {
                        let mut k = ExtrinsicCurvature::new(DIM);
                        let sign = if (ix + iy + iz) % 2 == 0 { 1.0 } else { -1.0 };
                        k.set_component(0, 0, perturbation * sign);
                        grid.set_k_tensor(ix, iy, iz, &k);
                    }
                }
            }
            grid
        };

        // Run without dissipation
        let mut grid_no_ko = make_grid();
        for _ in 0..steps {
            adm_step_rk4_with_gauge(&mut grid_no_ko, dt, Gauge::Geodesic, FdOrder::Second, 0.0);
        }

        // Run with dissipation
        let mut grid_ko = make_grid();
        for _ in 0..steps {
            adm_step_rk4_with_gauge(&mut grid_ko, dt, Gauge::Geodesic, FdOrder::Second, 1.0);
        }

        // Measure the amplitude of the Nyquist mode in K_{00} at the centre.
        let center = n / 2;
        let k_no_ko = grid_no_ko.k_tensor(center, center, center).component(0, 0);
        let k_ko = grid_ko.k_tensor(center, center, center).component(0, 0);

        // The dissipated run should have smaller |K_{00}| than the undissipated.
        assert!(
            k_ko.abs() < k_no_ko.abs(),
            "KO dissipation should reduce |K_00|: no_ko={}, ko={}",
            k_no_ko, k_ko
        );
    }

    // -- 4th-order stepper preserves flat space ------------------------------
    //
    // Sanity: the 4th-order path with analytic ∂Γ should also give zero RHS
    // and zero drift on flat space (all spatial derivatives vanish).

    #[test]
    fn fd4_flat_space_no_drift() {
        let mut grid = AdmGrid::new(7, 0.1);
        grid.init_flat_all();

        let h0 = hamiltonian_l2(&grid, FdOrder::Fourth);
        assert!(h0.abs() < TOL, "fd4 initial H = {} ≠ 0", h0);

        let dt = 1e-4;
        for _ in 0..100 {
            adm_step_rk4_with_gauge(&mut grid, dt, Gauge::Geodesic, FdOrder::Fourth, 0.0);
        }

        let g = grid.gamma(3, 3, 3);
        for i in 0..DIM {
            for j in 0..DIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((g.component(&[i, j]) - expected).abs() < TOL,
                    "fd4 γ_{}{} = {}, expected {}", i, j, g.component(&[i, j]), expected);
            }
        }

        let h_final = hamiltonian_l2(&grid, FdOrder::Fourth);
        assert!(h_final.abs() < TOL, "fd4 final H = {} ≠ 0", h_final);
    }
}
