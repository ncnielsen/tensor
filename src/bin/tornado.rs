#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//! Tornado simulation runner.
//!
//! Runs a spacetime-tornado ADM simulation and writes diagnostics as CSV.
//!
//! Usage:
//!   cargo run --release --bin tornado [amplitude] [n_sources] [n_steps] [grid_n] [output.csv]
//!
//! Defaults: amplitude=0.01, n_sources=6, n_steps=50, grid_n=9, output=stdout.
//!
//! Example:
//!   cargo run --release --bin tornado 0.01 6 200 13 results.csv

use std::fs::File;
use std::io::BufWriter;
use tensor::{run_tornado_cb, TornadoConfig, TornadoSnapshot};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let amplitude: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let n_sources: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let n_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let grid_n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(9);
    let out_path: Option<&str> = args.get(5).map(|s| s.as_str());

    let mut config = TornadoConfig::perturbative(amplitude, n_sources, n_steps);
    // Override grid dimensions while keeping ring geometry relative to cell size
    let dx = config.dx;
    config.nx = grid_n;
    config.ny = grid_n;
    config.nz = grid_n;
    config.dx = dx;
    config.dy = dx;
    config.dz = dx;

    eprintln!(
        "Tornado simulation: {}×{}×{} grid, {} steps, {} sources, amplitude={:.2e}",
        config.nx, config.ny, config.nz, config.n_steps, config.n_sources, config.amplitude
    );
    eprintln!(
        "dt={:.3e}, ring_radius={:.3e}, source_sigma={:.3e}",
        config.dt, config.ring_radius, config.source_sigma
    );
    eprintln!("{:>6}  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
        "step", "t", "H_L2", "K_offdiag", "γ_perturb", "max_ρ_EM");

    let progress = |s: &TornadoSnapshot| {
        eprintln!(
            "{:>6}  {:>8.4}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>12.3e}",
            s.step, s.t, s.hamiltonian_l2, s.k_offdiag_rms, s.gamma_perturb_rms, s.max_em_rho
        );
    };

    let result = run_tornado_cb(&config, progress);

    eprintln!("---");
    eprintln!("Peak K_offdiag:      {:.3e}", result.peak_k_offdiag());
    eprintln!("Peak H_L2 (constraint): {:.3e}", result.peak_hamiltonian_violation());

    // Write CSV
    match out_path {
        Some(path) => {
            let f = File::create(path).unwrap_or_else(|e| {
                eprintln!("Cannot open {path}: {e}");
                std::process::exit(1);
            });
            let mut bw = BufWriter::new(f);
            result.write_csv(&mut bw).expect("CSV write failed");
            eprintln!("CSV written to {path}");
        }
        None => {
            let stdout = std::io::stdout();
            let mut bw = BufWriter::new(stdout.lock());
            result.write_csv(&mut bw).expect("CSV write failed");
        }
    }
}
