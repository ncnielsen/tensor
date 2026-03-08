use crate::adm_matter::AdmMatter;
use crate::adm_grid::AdmGrid;
use crate::ops::em_stress_energy::em_stress_energy;
use crate::ops::faraday::faraday;
use crate::ops::partial_deriv::partial_deriv;
use crate::solver::invert_matrix;
use crate::tensor::Tensor;

// ── Single EM source ──────────────────────────────────────────────────────────

/// A single magnetic vortex source in 3D space.
///
/// Produces an axial magnetic flux tube via the 4-potential:
///
///   A₀(x) = −½ B (y − cy) exp(−r²/2σ²)
///   A₁(x) =  ½ B (x − cx) exp(−r²/2σ²)
///   A₂ = A₃ = 0
///
/// where r² = (x−cx)² + (y−cy)², giving an axial field F_{01} peaked at the
/// source centre.  Amplitude B can be set to zero to deactivate the source.
#[derive(Debug, Clone)]
pub struct EmSource {
    /// Centre in 3D Cartesian coordinates (x, y, z).
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    /// Peak magnetic field amplitude B₀.
    pub amplitude: f64,
    /// Gaussian half-width σ.
    pub sigma: f64,
}

impl EmSource {
    /// Evaluate the 4-potential A_μ (dim=4) at spatial position `x = [x, y, z]`.
    pub fn potential_at(&self, x: &[f64]) -> [f64; 4] {
        let dx = x[0] - self.cx;
        let dy = x[1] - self.cy;
        let r2 = dx * dx + dy * dy;
        let gauss = (-r2 / (2.0 * self.sigma * self.sigma)).exp();
        let b = self.amplitude;
        [
            -0.5 * b * dy * gauss, // A₀
             0.5 * b * dx * gauss, // A₁
             0.0,                   // A₂
             0.0,                   // A₃ (time component)
        ]
    }
}

// ── Circular tornado array ────────────────────────────────────────────────────

/// A circular ring of `n` magnetic vortex sources in the xy-plane.
///
/// Sources are equally spaced at azimuthal angles φ_k = 2πk/n, radius R.
/// The activation pattern is a rotating "spotlight": at each time step exactly
/// one source is active, cycling through the ring in sequence.
///
/// This creates a rotating electromagnetic wave pattern that, over many cycles,
/// accumulates angular momentum in the spacetime curvature — the "tornado" seed.
#[derive(Debug, Clone)]
pub struct TornadoArray {
    pub sources: Vec<EmSource>,
    /// Orbital period of one full rotation (one complete cycle through all sources).
    pub period: f64,
}

impl TornadoArray {
    /// Construct a uniform ring of `n` sources centred at `(cx, cy, cz)`.
    ///
    /// # Arguments
    /// - `n`         — number of sources
    /// - `radius`    — ring radius in coordinate units
    /// - `cx, cy, cz`— physical centre of the ring
    /// - `sigma`     — Gaussian width of each source
    /// - `amplitude` — peak B field of each source
    /// - `period`    — time for one full rotation (one cycle through all sources)
    pub fn new(
        n: usize,
        radius: f64,
        cx: f64,
        cy: f64,
        cz: f64,
        sigma: f64,
        amplitude: f64,
        period: f64,
    ) -> Self {
        assert!(n >= 2, "Need at least 2 sources for a meaningful ring");
        let sources = (0..n)
            .map(|k| {
                let phi = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                EmSource {
                    cx: cx + radius * phi.cos(),
                    cy: cy + radius * phi.sin(),
                    cz,
                    amplitude,
                    sigma,
                }
            })
            .collect();
        TornadoArray { sources, period }
    }

    /// Index of the currently active source at time `t`.
    ///
    /// Sources activate in sequence with equal on-time.  The active index
    /// cycles 0 → 1 → … → n−1 → 0 with one full period per complete cycle.
    pub fn active_index(&self, t: f64) -> usize {
        let n = self.sources.len();
        let phase = (t / self.period).fract();         // 0..1
        let phase = if phase < 0.0 { phase + 1.0 } else { phase };
        (phase * n as f64) as usize % n
    }

    /// Sum of 4-potentials from all sources, scaled so only `active_index` is on.
    ///
    /// A smooth "soft" activation could be used later; for now it is a hard on/off.
    pub fn potential_at(&self, x: &[f64], t: f64) -> Vec<f64> {
        let idx = self.active_index(t);
        let pot = self.sources[idx].potential_at(x);
        pot.to_vec()
    }
}

// ── T_{μν} grid from a tornado array ─────────────────────────────────────────

/// Compute the ADM matter decomposition (ρ, J^i, S_{ij}) at every grid point
/// given a tornado source array evaluated at time `t`.
///
/// Uses the existing 4D EM pipeline:
///   A_μ → ∂A → F_{μν} → T_{μν}^{4D} → (ρ, J^i, S_{ij})
///
/// The 4D metric used for F and T is assembled from the ADM state at each
/// point via geodesic slicing: g_{00} = −1, g_{0i} = 0, g_{ij} = γ_{ij}.
///
/// All computation runs inside `aad::no_tape` — no AAD overhead.
///
/// # Arguments
/// - `array`   — tornado source array
/// - `grid`    — current ADM grid (provides the spatial metric)
/// - `t`       — current coordinate time
/// - `mu_0`    — magnetic permeability (use 1.0 for geometric units)
/// - `eps`     — finite-difference step for ∂_μ A_ν
pub fn tornado_matter_grid(
    array: &TornadoArray,
    grid: &AdmGrid,
    t: f64,
    mu_0: f64,
    eps: f64,
) -> Vec<AdmMatter> {
    let n_pts = grid.n_pts();
    let mut matters = Vec::with_capacity(n_pts);

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let matter = aad::no_tape(|| {
                    // Physical coordinates of this grid point.
                    let x = grid.x0 + ix as f64 * grid.dx;
                    let y = grid.y0 + iy as f64 * grid.dy;
                    let z = grid.z0 + iz as f64 * grid.dz;
                    let point = [x, y, z, 0.0_f64]; // 4D spacetime point (t=0 spatial slice)

                    // 3D spatial metric at this point.
                    let gamma_flat = grid.gamma_flat(ix, iy, iz);
                    let gamma_inv_flat = invert_matrix(&gamma_flat, 3)
                        .expect("singular spatial metric in tornado_matter_grid");

                    // Build 4D metric: g_{00} = -1, g_{0i} = 0, g_{ij} = γ_{ij}.
                    let mut g4_flat = [0.0_f64; 16];
                    g4_flat[0] = -1.0;  // g_{00}
                    for i in 0..3 {
                        for j in 0..3 {
                            g4_flat[(i + 1) * 4 + (j + 1)] = gamma_flat[i * 3 + j];
                        }
                    }

                    // 4D inverse metric: g^{00} = -1, g^{0i} = 0, g^{ij} = γ^{ij}.
                    let mut g4_inv_flat = [0.0_f64; 16];
                    g4_inv_flat[0] = -1.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            g4_inv_flat[(i + 1) * 4 + (j + 1)] = gamma_inv_flat[i * 3 + j];
                        }
                    }

                    let g4: Tensor<0, 2> = Tensor::from_f64(4, g4_flat.to_vec());
                    let g4_inv: Tensor<2, 0> = Tensor::from_f64(4, g4_inv_flat.to_vec());
                    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);

                    // Evaluate the 4-potential and compute F and T.
                    let a_fn = |xq: &[f64]| -> Tensor<0, 1> {
                        Tensor::from_f64(4, array.potential_at(xq, t))
                    };
                    let partial_a: Tensor<0, 2> = partial_deriv(&a_fn, &point, eps);
                    let f_tensor = faraday(&partial_a);
                    let t4 = em_stress_energy(&f_tensor, &g4, &g4_inv, mu_0);

                    AdmMatter::from_t4d(&t4, &gamma_inv)
                });

                matters.push(matter);
            }
        }
    }

    matters
}

// ── Flat helper ───────────────────────────────────────────────────────────────

/// Physical coordinates for grid point (ix, iy, iz) — convenience helper.
#[allow(dead_code)]
pub fn grid_coords(grid: &AdmGrid, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
    [
        ix as f64 * grid.dx,
        iy as f64 * grid.dy,
        iz as f64 * grid.dz,
    ]
}
