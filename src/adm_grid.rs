use crate::adm::{AdmState, ExtrinsicCurvature};
use crate::tensor::Tensor;
use aad::number::Number;

// Per-point field layout (22 f64 values):
//   [ 0.. 9)  — γ_{ij} flat row-major, γ[i*3+j]
//   [ 9..18)  — K_{ij} flat row-major, K[i*3+j]
//   [18]      — α (lapse)
//   [19..22)  — β^i (shift, upper index)
pub(crate) const GAMMA_OFF: usize = 0;
pub(crate) const K_OFF: usize = 9;
pub(crate) const ALPHA_OFF: usize = 18;
pub(crate) const BETA_OFF: usize = 19;
pub(crate) const FIELDS_PER_PT: usize = 22;

/// A uniform 3D Cartesian grid of ADM state variables.
///
/// Grid points are indexed (ix, iy, iz) with ix in 0..nx, iy in 0..ny, iz in 0..nz.
/// Flat storage layout: pt = ix * ny * nz + iy * nz + iz  (z-innermost).
///
/// **Boundary convention**: the outer 2-cell band is never evolved; it holds
/// the initial (or analytically specified) boundary data throughout the run.
/// Interior points `[2..nx-2] × [2..ny-2] × [2..nz-2]` are evolved by RK4.
/// A minimum grid size of 5 in each direction is therefore required.
#[derive(Debug, Clone)]
pub struct AdmGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Physical coordinate of grid point (0,0,0).
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    /// Flat field storage: `fields[pt_flat * FIELDS_PER_PT + field_off]`.
    pub fields: Vec<f64>,
}

impl AdmGrid {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Construct a grid by evaluating a pointwise function.
    ///
    /// `state_fn(x, y, z)` returns the initial `AdmState` at physical coordinates
    /// `(x0 + ix*dx, y0 + iy*dy, z0 + iz*dz)`.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        x0: f64,
        y0: f64,
        z0: f64,
        state_fn: impl Fn(f64, f64, f64) -> AdmState,
    ) -> Self {
        assert!(nx >= 5 && ny >= 5 && nz >= 5, "Grid must be at least 5×5×5");
        let n_pts = nx * ny * nz;
        let mut fields = vec![0.0_f64; n_pts * FIELDS_PER_PT];

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = x0 + ix as f64 * dx;
                    let y = y0 + iy as f64 * dy;
                    let z = z0 + iz as f64 * dz;
                    let state = state_fn(x, y, z);
                    let base = Self::flat_pt_static(ix, iy, iz, ny, nz) * FIELDS_PER_PT;

                    for i in 0..9 {
                        fields[base + GAMMA_OFF + i] = state.gamma.components[i].result;
                        fields[base + K_OFF + i] = state.k.components[i].result;
                    }
                    fields[base + ALPHA_OFF] = state.alpha;
                    for i in 0..3 {
                        fields[base + BETA_OFF + i] = state.beta[i];
                    }
                }
            }
        }

        AdmGrid { nx, ny, nz, dx, dy, dz, x0, y0, z0, fields }
    }

    /// Flat Minkowski grid: γ = δ_{ij}, K = 0, α = 1, β = 0 everywhere.
    pub fn flat(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self::new(nx, ny, nz, dx, dy, dz, 0.0, 0.0, 0.0, |_, _, _| AdmState::flat())
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn flat_pt(&self, ix: usize, iy: usize, iz: usize) -> usize {
        Self::flat_pt_static(ix, iy, iz, self.ny, self.nz)
    }

    #[inline]
    pub(crate) fn flat_pt_static(ix: usize, iy: usize, iz: usize, ny: usize, nz: usize) -> usize {
        ix * ny * nz + iy * nz + iz
    }

    #[inline]
    pub fn n_pts(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    // ── Field accessors ───────────────────────────────────────────────────────

    pub fn gamma_flat(&self, ix: usize, iy: usize, iz: usize) -> [f64; 9] {
        let base = self.flat_pt(ix, iy, iz) * FIELDS_PER_PT + GAMMA_OFF;
        self.fields[base..base + 9].try_into().unwrap()
    }

    pub fn k_flat(&self, ix: usize, iy: usize, iz: usize) -> [f64; 9] {
        let base = self.flat_pt(ix, iy, iz) * FIELDS_PER_PT + K_OFF;
        self.fields[base..base + 9].try_into().unwrap()
    }

    pub fn alpha_at(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.fields[self.flat_pt(ix, iy, iz) * FIELDS_PER_PT + ALPHA_OFF]
    }

    pub fn beta_at(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        let base = self.flat_pt(ix, iy, iz) * FIELDS_PER_PT + BETA_OFF;
        self.fields[base..base + 3].try_into().unwrap()
    }

    /// Reconstruct an `AdmState` at a grid point.
    pub fn state_at(&self, ix: usize, iy: usize, iz: usize) -> AdmState {
        let gamma_f = self.gamma_flat(ix, iy, iz);
        let k_f = self.k_flat(ix, iy, iz);
        AdmState {
            gamma: Tensor::new(3, gamma_f.iter().map(|&v| Number::new(v)).collect()),
            k: ExtrinsicCurvature::new(3, k_f.iter().map(|&v| Number::new(v)).collect()),
            alpha: self.alpha_at(ix, iy, iz),
            beta: self.beta_at(ix, iy, iz),
        }
    }

    // ── Interior check ────────────────────────────────────────────────────────

    /// Returns true if (ix, iy, iz) is in the evolved interior
    /// (at least 2 cells away from every boundary).
    #[inline]
    pub fn is_interior(&self, ix: usize, iy: usize, iz: usize) -> bool {
        ix >= 2
            && ix + 2 < self.nx
            && iy >= 2
            && iy + 2 < self.ny
            && iz >= 2
            && iz + 2 < self.nz
    }

    // ── Apply a flat RHS delta to a copy of the grid ──────────────────────────

    /// Return a new grid with interior fields updated by `base + scale * rhs`.
    ///
    /// `rhs` has the same layout as `self.fields`; boundary entries are ignored.
    pub fn with_rhs(&self, rhs: &[f64], scale: f64) -> Self {
        assert_eq!(rhs.len(), self.fields.len());
        let mut out = self.clone();
        for ix in 2..self.nx.saturating_sub(2) {
            for iy in 2..self.ny.saturating_sub(2) {
                for iz in 2..self.nz.saturating_sub(2) {
                    let base = self.flat_pt(ix, iy, iz) * FIELDS_PER_PT;
                    // Only gamma (0..9) and k (9..18) are evolved
                    for f in 0..18 {
                        out.fields[base + f] += scale * rhs[base + f];
                    }
                }
            }
        }
        out
    }
}
