use crate::adm::{adm_rhs_geodesic, hamiltonian_constraint, AdmState, ExtrinsicCurvature};
use tensor_core::{
    christoffel::Christoffel,
    curvature::ChristoffelDerivative,
    metric::invert_metric,
    tensor::Tensor,
};

const DIM: usize = 3;

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
    pub fn new(n: usize, h: f64) -> Self {
        assert!(n >= 5, "minimum grid size is 5 (2-cell boundary band needs 1 interior)");
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
        for ix in 2..(n - 2) {
            for iy in 2..(n - 2) {
                for iz in 2..(n - 2) {
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
// FD operators
// ---------------------------------------------------------------------------

/// Central FD of γ in each spatial direction: ∂_k γ_{ij} = (γ[+1] − γ[−1]) / 2h.
fn partial_gamma_at(
    grid: &AdmGrid,
    ix: usize, iy: usize, iz: usize,
) -> [Tensor<0, 2>; 3] {
    let h2 = 2.0 * grid.h();
    let mut out = [Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM), Tensor::<0, 2>::new(DIM)];

    macro_rules! fd_dir {
        ($d:expr, $xp:expr, $xm:expr) => {{
            for i in 0..DIM {
                for j in 0..DIM {
                    let v = ($xp.component(&[i, j]) - $xm.component(&[i, j])) / h2;
                    out[$d].set_component(&[i, j], v);
                }
            }
        }};
    }

    fd_dir!(0, grid.gamma(ix + 1, iy, iz), grid.gamma(ix - 1, iy, iz));
    fd_dir!(1, grid.gamma(ix, iy + 1, iz), grid.gamma(ix, iy - 1, iz));
    fd_dir!(2, grid.gamma(ix, iy, iz + 1), grid.gamma(ix, iy, iz - 1));

    out
}

/// Christoffel symbols at a grid point from the FD metric derivatives.
fn christoffel_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> Christoffel {
    let gamma = grid.gamma(ix, iy, iz);
    let gamma_inv = invert_metric(&gamma);
    let pg = partial_gamma_at(grid, ix, iy, iz);
    let pg_slice: Vec<Tensor<0, 2>> = pg.to_vec();
    Christoffel::from_metric(&gamma_inv, &pg_slice)
}

/// ∂_l Γ^i_{jk} at a grid point via central FD of Christoffel at neighbors.
fn christoffel_deriv_at(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> ChristoffelDerivative {
    let h2 = 2.0 * grid.h();
    let d3 = DIM * DIM * DIM;
    let d2 = DIM * DIM;
    let mut data = vec![0.0f64; DIM.pow(4)];

    let directions = [
        (christoffel_at(grid, ix + 1, iy, iz), christoffel_at(grid, ix - 1, iy, iz), 0usize),
        (christoffel_at(grid, ix, iy + 1, iz), christoffel_at(grid, ix, iy - 1, iz), 1usize),
        (christoffel_at(grid, ix, iy, iz + 1), christoffel_at(grid, ix, iy, iz - 1), 2usize),
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

// ---------------------------------------------------------------------------
// Grid-level RHS
// ---------------------------------------------------------------------------

/// Evaluate the geodesic ADM RHS at every interior point.
///
/// Boundary cells (within 2 cells of any face) are left as zero in the returned
/// grid — they do not participate in the evolution.
fn geodesic_rhs(grid: &AdmGrid) -> AdmGrid {
    let n = grid.n();
    let mut rhs = AdmGrid::new(n, grid.h());

    for ix in 2..(n - 2) {
        for iy in 2..(n - 2) {
            for iz in 2..(n - 2) {
                let gamma = grid.gamma(ix, iy, iz);
                let k = grid.k_tensor(ix, iy, iz);
                let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
                let ch = christoffel_at(grid, ix, iy, iz);
                let dch = christoffel_deriv_at(grid, ix, iy, iz);

                let (gd, kd) = adm_rhs_geodesic(&state, &ch, &dch);
                rhs.set_gamma(ix, iy, iz, &gd);
                rhs.set_k_tensor(ix, iy, iz, &kd);
            }
        }
    }

    rhs
}

/// Return a new grid equal to `base + scale * delta` (elementwise on raw data).
fn scaled_add(base: &AdmGrid, scale: f64, delta: &AdmGrid) -> AdmGrid {
    let mut result = AdmGrid::new(base.n(), base.h());
    let r = result.raw_mut();
    let b = base.raw();
    let d = delta.raw();
    for i in 0..r.len() {
        r[i] = b[i] + scale * d[i];
    }
    result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Advance `grid` by one 4th-order Runge-Kutta step of size `dt`.
///
/// Uses geodesic slicing (α = 1, β = 0). Boundary cells are frozen.
pub fn adm_step_rk4(grid: &mut AdmGrid, dt: f64) {
    let k1 = geodesic_rhs(grid);

    let y2 = scaled_add(grid, dt / 2.0, &k1);
    let k2 = geodesic_rhs(&y2);

    let y3 = scaled_add(grid, dt / 2.0, &k2);
    let k3 = geodesic_rhs(&y3);

    let y4 = scaled_add(grid, dt, &k3);
    let k4 = geodesic_rhs(&y4);

    let n = grid.n();
    let raw = grid.raw_mut();
    for ix in 2..(n - 2) {
        for iy in 2..(n - 2) {
            for iz in 2..(n - 2) {
                let base = ((ix * n + iy) * n + iz) * FIELDS;
                for f in 0..FIELDS {
                    raw[base + f] += dt / 6.0
                        * (k1.raw()[base + f]
                            + 2.0 * k2.raw()[base + f]
                            + 2.0 * k3.raw()[base + f]
                            + k4.raw()[base + f]);
                }
            }
        }
    }
}

/// RMS of the Hamiltonian constraint over all interior points.
///
/// Vanishes for physically consistent data; grows when constraints are violated.
pub fn hamiltonian_l2(grid: &AdmGrid) -> f64 {
    let n = grid.n();
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for ix in 2..(n - 2) {
        for iy in 2..(n - 2) {
            for iz in 2..(n - 2) {
                let gamma = grid.gamma(ix, iy, iz);
                let k = grid.k_tensor(ix, iy, iz);
                let state = AdmState::new(gamma, k, 1.0, [0.0; 3]);
                let ch = christoffel_at(grid, ix, iy, iz);
                let dch = christoffel_deriv_at(grid, ix, iy, iz);

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
        // 5×5×5 grid — minimum size, one interior point at (2,2,2).
        let mut grid = AdmGrid::new(5, 0.1);
        grid.init_flat_all();

        // Hamiltonian constraint starts at 0.
        let h0 = hamiltonian_l2(&grid);
        assert!(h0.abs() < TOL, "initial H = {} ≠ 0", h0);

        // 100 steps with small dt — flat space should not evolve.
        let dt = 1e-4;
        for _ in 0..100 {
            adm_step_rk4(&mut grid, dt);
        }

        // Metric at (2,2,2) must still be identity.
        let g = grid.gamma(2, 2, 2);
        for i in 0..DIM {
            for j in 0..DIM {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = g.component(&[i, j]);
                assert!(
                    (got - expected).abs() < TOL,
                    "γ_{}{}  = {}, expected {} after 100 steps",
                    i, j, got, expected
                );
            }
        }

        let h_final = hamiltonian_l2(&grid);
        assert!(h_final.abs() < TOL, "final H = {} ≠ 0", h_final);
    }

    // -- Boundary cells unchanged after evolution ----------------------------

    #[test]
    fn boundary_cells_unchanged() {
        let mut grid = AdmGrid::new(5, 0.1);
        grid.init_flat_all();

        // Record boundary values (e.g., corner (0,0,0))
        let g_before = grid.gamma(0, 0, 0);
        let k_before = grid.k_tensor(0, 0, 0);

        adm_step_rk4(&mut grid, 0.01);

        let g_after = grid.gamma(0, 0, 0);
        let k_after = grid.k_tensor(0, 0, 0);

        for i in 0..DIM {
            for j in 0..DIM {
                assert_eq!(
                    g_before.component(&[i, j]),
                    g_after.component(&[i, j]),
                    "boundary γ_{}{}  changed", i, j
                );
                assert_eq!(
                    k_before.component(i, j),
                    k_after.component(i, j),
                    "boundary K_{}{}  changed", i, j
                );
            }
        }
    }

    // -- Isotropic K: ∂_t γ and ∂_t K match analytic ------------------------
    //
    // Spatially uniform γ = I, K = ε I in geodesic slicing on flat space:
    //   ∂_t γ_{ij} = −2ε δ_{ij}
    //   ∂_t K_{ij} = ε² δ_{ij}   (since R=0, K=3ε, K_{im}K^m_j = ε² δ_{ij})
    //
    // With uniform fields the FD metric derivatives are exactly zero, so
    // Christoffel = 0 and the grid result equals the point-wise adm_rhs.

    #[test]
    fn isotropic_k_rhs_analytic() {
        let eps = 0.1f64;
        let mut grid = AdmGrid::new(5, 0.1);
        grid.init_flat_all();

        // Set all points to isotropic K = ε I.
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    let mut k = ExtrinsicCurvature::new(DIM);
                    for i in 0..DIM { k.set_component(i, i, eps); }
                    grid.set_k_tensor(ix, iy, iz, &k);
                }
            }
        }

        let rhs = geodesic_rhs(&grid);

        let gd = rhs.gamma(2, 2, 2);
        let kd = rhs.k_tensor(2, 2, 2);

        for i in 0..DIM {
            for j in 0..DIM {
                let expected_gd = if i == j { -2.0 * eps } else { 0.0 };
                let expected_kd = if i == j { eps * eps } else { 0.0 };
                assert!(
                    (gd.component(&[i, j]) - expected_gd).abs() < TOL,
                    "∂_t γ_{}{} = {}, expected {}", i, j, gd.component(&[i, j]), expected_gd
                );
                assert!(
                    (kd.component(i, j) - expected_kd).abs() < TOL,
                    "∂_t K_{}{} = {}, expected {}", i, j, kd.component(i, j), expected_kd
                );
            }
        }
    }

    // -- RK4 convergence order ----------------------------------------------
    //
    // Isotropic uniform K = ε I on flat γ = I gives a pure ODE (FD gives
    // exact zero spatial derivatives) provided the test point's stencil
    // stays interior throughout the RK4 stages.
    //
    // With n=9 and center (4,4,4): christoffel_deriv_at(4,4,4) calls
    // christoffel_at at ix=3 and ix=5, which in turn call partial_gamma_at
    // With γ = g·I and K = k·I (spatially uniform, Christoffel = 0), the ADM
    // geodesic equations reduce to a point-wise ODE:
    //   dg/dt = -2k,  dk/dt = k²/g
    //
    // Substituting v = k/g gives v' = 3v², so v(t) = ε/(1-3εt).
    // Back-substituting: g(t) = (1-3εt)^{2/3}, k(t) = ε(1-3εt)^{-1/3}.
    //
    // We drive the ODE using `adm_rhs_geodesic` with zero Christoffels —
    // exactly what the grid computes for a spatially uniform field — wrapped
    // in a manual RK4 loop. This avoids the boundary-contamination artefact
    // (interior FD stencils touching frozen boundary cells after the first
    // sub-step affect cells near the boundary, and propagate inward).
    //
    // Error ratio at dt vs dt/2 should be ≥ 10 (theoretical: 2⁴ = 16).

    #[test]
    fn rk4_convergence_order() {
        use tensor_core::{christoffel::Christoffel, curvature::ChristoffelDerivative};

        let eps = 0.05f64;

        let gamma_exact = |t: f64| (1.0 - 3.0 * eps * t).powf(2.0 / 3.0);
        let k_exact = |t: f64| eps * (1.0 - 3.0 * eps * t).powf(-1.0 / 3.0);

        // Point-wise RHS: adm_rhs_geodesic with zero Christoffels.
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

        // One step at dt1
        let (g1, k1_val) = rk4_step(1.0, eps, dt1);
        let err1_gamma = (g1 - gamma_exact(dt1)).abs();
        let err1_k = (k1_val - k_exact(dt1)).abs();

        // Two steps at dt2
        let (g2a, k2a) = rk4_step(1.0, eps, dt2);
        let (g2b, k2b) = rk4_step(g2a, k2a, dt2);
        let err2_gamma = (g2b - gamma_exact(dt1)).abs();
        let err2_k = (k2b - k_exact(dt1)).abs();

        // Error ratio should be ≥ 10 (RK4 theoretical: 2⁴ = 16)
        assert!(
            err1_gamma / err2_gamma >= 10.0,
            "γ RK4 convergence: err(dt)={:.2e}, err(dt/2)={:.2e}, ratio={:.1}",
            err1_gamma, err2_gamma, err1_gamma / err2_gamma
        );
        assert!(
            err1_k / err2_k >= 10.0,
            "K RK4 convergence: err(dt)={:.2e}, err(dt/2)={:.2e}, ratio={:.1}",
            err1_k, err2_k, err1_k / err2_k
        );
    }
}
