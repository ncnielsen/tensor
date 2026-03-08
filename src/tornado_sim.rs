use crate::adm_grid::AdmGrid;
use crate::adm_matter::AdmMatter;
use crate::adm_step::{adm_step_rk4_with_source, hamiltonian_l2};
use crate::adm::ExtrinsicCurvature;
use crate::solver::invert_matrix;
use crate::tensor::Tensor;
use crate::tornado::{TornadoArray, tornado_matter_grid};

// ── Configuration ─────────────────────────────────────────────────────────────

/// All parameters for a tornado simulation run.
pub struct TornadoConfig {
    // Source array
    /// Number of EM sources on the ring.
    pub n_sources: usize,
    /// Ring radius in coordinate units.
    pub ring_radius: f64,
    /// Gaussian half-width σ of each source.
    pub source_sigma: f64,
    /// Peak magnetic field amplitude B₀.
    pub amplitude: f64,
    /// Time for one full rotation (all sources activated once).
    pub period: f64,

    // Grid
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,

    // Integration
    /// Time step (should satisfy CFL: dt ≤ 0.5 min(dx,dy,dz)).
    pub dt: f64,
    /// Total number of RK4 time steps.
    pub n_steps: usize,
    /// Record a snapshot every this many steps.
    pub output_every: usize,

    // Physics
    /// Magnetic permeability (1.0 for geometric units G = c = 1).
    pub mu_0: f64,
    /// FD step for computing ∂_μ A_ν (use ≈ 1e-5).
    pub eps_fd: f64,
}

impl TornadoConfig {
    /// Convenience: set up a small perturbative run centred on the grid.
    ///
    /// Uses a 9×9×9 grid with the ring at the grid centre and a CFL-safe dt.
    pub fn perturbative(amplitude: f64, n_sources: usize, n_steps: usize) -> Self {
        let dx = 0.1_f64;
        TornadoConfig {
            n_sources,
            ring_radius: 3.0 * dx,    // ring radius = 3 cells
            source_sigma: 2.0 * dx,   // source width = 2 cells
            amplitude,
            period: 1.0,

            nx: 9, ny: 9, nz: 9,
            dx, dy: dx, dz: dx,

            dt: 0.04 * dx,            // CFL factor 0.4
            n_steps,
            output_every: (n_steps / 10).max(1),

            mu_0: 1.0,
            eps_fd: 1e-5,
        }
    }
}

// ── Per-step diagnostics ──────────────────────────────────────────────────────

/// Diagnostics recorded at each output step.
#[derive(Debug, Clone)]
pub struct TornadoSnapshot {
    /// Step index.
    pub step: usize,
    /// Coordinate time t = step * dt.
    pub t: f64,
    /// L2 norm of Hamiltonian constraint H across interior points (accuracy monitor).
    pub hamiltonian_l2: f64,
    /// RMS of scalar trace K = γ^{ij} K_{ij} over interior points.
    pub k_trace_rms: f64,
    /// RMS of off-diagonal K components {K_{01}, K_{02}, K_{12}} (rotation indicator).
    pub k_offdiag_rms: f64,
    /// RMS deviation of γ_{ij} from δ_{ij} over interior points (curvature growth).
    pub gamma_perturb_rms: f64,
    /// EM angular momentum density in z: J_z = Σ (x J^y − y J^x) dV.
    pub em_angular_momentum_z: f64,
    /// Maximum |ρ| (EM energy density) across all grid points.
    pub max_em_rho: f64,
}

// ── Diagnostics computation ───────────────────────────────────────────────────

fn compute_diagnostics(
    grid: &AdmGrid,
    matters: &[AdmMatter],
    step: usize,
    dt: f64,
) -> TornadoSnapshot {
    let t = step as f64 * dt;

    // Grid centre (physical origin of the ring)
    let cx = grid.x0 + (grid.nx as f64 / 2.0) * grid.dx;
    let cy = grid.y0 + (grid.ny as f64 / 2.0) * grid.dy;

    let mut k_trace_sq = 0.0_f64;
    let mut k_offdiag_sq = 0.0_f64;
    let mut gamma_perturb_sq = 0.0_f64;
    let mut n_interior = 0_usize;

    for ix in 2..grid.nx.saturating_sub(2) {
        for iy in 2..grid.ny.saturating_sub(2) {
            for iz in 2..grid.nz.saturating_sub(2) {
                aad::no_tape(|| {
                    let gamma_flat = grid.gamma_flat(ix, iy, iz);
                    let k_flat = grid.k_flat(ix, iy, iz);
                    let gamma_inv_flat = invert_matrix(&gamma_flat, 3).unwrap();
                    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, gamma_inv_flat);
                    let k = ExtrinsicCurvature::new(
                        3,
                        k_flat.iter().map(|&v| aad::number::Number::new(v)).collect(),
                    );
                    let k_tr = k.trace(&gamma_inv).result;
                    k_trace_sq += k_tr * k_tr;

                    // Off-diagonal K: K_{01}, K_{02}, K_{12}
                    for (i, j) in [(0,1), (0,2), (1,2)] {
                        let v = k_flat[i * 3 + j];
                        k_offdiag_sq += v * v;
                    }

                    // γ deviation from δ
                    for i in 0..3 {
                        for j in 0..3 {
                            let expected = if i == j { 1.0 } else { 0.0 };
                            let d = gamma_flat[i * 3 + j] - expected;
                            gamma_perturb_sq += d * d;
                        }
                    }
                });
                n_interior += 1;
            }
        }
    }

    let n = n_interior.max(1) as f64;
    let k_trace_rms = (k_trace_sq / n).sqrt();
    let k_offdiag_rms = (k_offdiag_sq / (3.0 * n)).sqrt();
    let gamma_perturb_rms = (gamma_perturb_sq / (9.0 * n)).sqrt();

    // EM angular momentum in z: J_z = Σ (x J^y − y J^x) dV
    let dv = grid.dx * grid.dy * grid.dz;
    let em_angular_momentum_z: f64 = matters
        .iter()
        .enumerate()
        .map(|(flat, m)| {
            let ix = flat / (grid.ny * grid.nz);
            let iy = (flat / grid.nz) % grid.ny;
            let x = grid.x0 + ix as f64 * grid.dx - cx;
            let y = grid.y0 + iy as f64 * grid.dy - cy;
            // J^y = j[1], J^x = j[0] (upper index components)
            (x * m.j[1] - y * m.j[0]) * dv
        })
        .sum();

    let max_em_rho = matters.iter().map(|m| m.rho.abs()).fold(0.0_f64, f64::max);

    let hamiltonian = hamiltonian_l2(grid);

    TornadoSnapshot {
        step,
        t,
        hamiltonian_l2: hamiltonian,
        k_trace_rms,
        k_offdiag_rms,
        gamma_perturb_rms,
        em_angular_momentum_z,
        max_em_rho,
    }
}

// ── Simulation result ─────────────────────────────────────────────────────────

pub struct TornadoResult {
    pub snapshots: Vec<TornadoSnapshot>,
    pub final_grid: AdmGrid,
}

impl TornadoResult {
    /// Maximum off-diagonal K RMS across all snapshots — growth indicates rotation.
    pub fn peak_k_offdiag(&self) -> f64 {
        self.snapshots.iter().map(|s| s.k_offdiag_rms).fold(0.0_f64, f64::max)
    }

    /// Maximum Hamiltonian constraint violation — growth indicates loss of accuracy.
    pub fn peak_hamiltonian_violation(&self) -> f64 {
        self.snapshots.iter().map(|s| s.hamiltonian_l2).fold(0.0_f64, f64::max)
    }

    /// Maximum EM angular momentum across all snapshots.
    pub fn peak_em_angular_momentum_z(&self) -> f64 {
        self.snapshots.iter().map(|s| s.em_angular_momentum_z.abs()).fold(0.0_f64, f64::max)
    }

    /// Print a summary table to stdout.
    pub fn print_summary(&self) {
        println!(
            "{:>6}  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            "step", "t", "H_L2", "K_trace", "K_offdiag", "γ_perturb", "J_z^EM"
        );
        for s in &self.snapshots {
            println!(
                "{:>6}  {:>8.4}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>12.3e}",
                s.step, s.t,
                s.hamiltonian_l2, s.k_trace_rms, s.k_offdiag_rms,
                s.gamma_perturb_rms, s.em_angular_momentum_z
            );
        }
    }
}

// ── Result helpers ────────────────────────────────────────────────────────────

impl TornadoResult {
    /// Write all snapshots as CSV to `w`.
    ///
    /// Columns: step, t, hamiltonian_l2, k_trace_rms, k_offdiag_rms,
    ///          gamma_perturb_rms, em_angular_momentum_z, max_em_rho
    pub fn write_csv<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "step,t,hamiltonian_l2,k_trace_rms,k_offdiag_rms,\
             gamma_perturb_rms,em_angular_momentum_z,max_em_rho"
        )?;
        for s in &self.snapshots {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{}",
                s.step,
                s.t,
                s.hamiltonian_l2,
                s.k_trace_rms,
                s.k_offdiag_rms,
                s.gamma_perturb_rms,
                s.em_angular_momentum_z,
                s.max_em_rho,
            )?;
        }
        Ok(())
    }
}

// ── Main runner ───────────────────────────────────────────────────────────────

/// Run a tornado simulation from flat Minkowski initial data.
///
/// The ring is placed at the physical centre of the grid.
/// Flat initial data (γ = δ, K = 0) is used; the EM source drives curvature growth.
///
/// `on_snapshot` is called with each recorded snapshot (including the initial
/// step 0 and the final step), allowing the caller to print progress.
///
/// Returns snapshots at every `config.output_every` steps plus the final grid.
pub fn run_tornado_cb<F: FnMut(&TornadoSnapshot)>(
    config: &TornadoConfig,
    mut on_snapshot: F,
) -> TornadoResult {
    // Physical centre of the grid
    let cx = config.nx as f64 * config.dx / 2.0;
    let cy = config.ny as f64 * config.dy / 2.0;
    let cz = config.nz as f64 * config.dz / 2.0;

    // Set up the source array centred on the grid
    let array = TornadoArray::new(
        config.n_sources,
        config.ring_radius,
        cx, cy, cz,
        config.source_sigma,
        config.amplitude,
        config.period,
    );

    // Flat Minkowski initial data
    let mut grid = AdmGrid::flat(
        config.nx, config.ny, config.nz,
        config.dx, config.dy, config.dz,
    );

    let mut snapshots = Vec::new();

    for step in 0..=config.n_steps {
        let t = step as f64 * config.dt;

        // Evaluate EM source at current time
        let matters = tornado_matter_grid(&array, &grid, t, config.mu_0, config.eps_fd);

        // Record diagnostics at output steps
        if step % config.output_every == 0 || step == config.n_steps {
            let snap = compute_diagnostics(&grid, &matters, step, config.dt);
            on_snapshot(&snap);
            snapshots.push(snap);
        }

        // Advance (don't step past the end)
        if step < config.n_steps {
            grid = adm_step_rk4_with_source(&grid, config.dt, &matters);
        }
    }

    TornadoResult { snapshots, final_grid: grid }
}

/// Run a tornado simulation from flat Minkowski initial data.
///
/// Convenience wrapper around [`run_tornado_cb`] with a no-op callback.
pub fn run_tornado(config: &TornadoConfig) -> TornadoResult {
    run_tornado_cb(config, |_| {})
}
