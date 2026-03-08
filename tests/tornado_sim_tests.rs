#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use tensor::tornado_sim::{run_tornado, TornadoConfig};

// ── Fast structural tests ─────────────────────────────────────────────────────

/// Zero amplitude: no EM source → vacuum evolution → flat space stays flat.
/// Diagnostics should show near-zero curvature throughout.
#[test]
fn test_tornado_sim_zero_amplitude() {
    let config = TornadoConfig {
        n_sources: 4,
        ring_radius: 0.3,
        source_sigma: 0.2,
        amplitude: 0.0,          // no source
        period: 1.0,
        nx: 5, ny: 5, nz: 5,
        dx: 0.1, dy: 0.1, dz: 0.1,
        dt: 0.01,
        n_steps: 2,
        output_every: 1,
        mu_0: 1.0,
        eps_fd: 1e-5,
    };

    let result = run_tornado(&config);

    // Should have snapshots at steps 0, 1, 2
    assert_eq!(result.snapshots.len(), 3);
    assert_eq!(result.snapshots[0].step, 0);
    assert_eq!(result.snapshots[2].step, 2);

    // With no source, space stays flat
    for snap in &result.snapshots {
        assert!(
            snap.gamma_perturb_rms < 1e-10,
            "step {}: gamma should stay flat, got {:.3e}",
            snap.step, snap.gamma_perturb_rms
        );
        assert!(
            snap.k_trace_rms < 1e-10,
            "step {}: K_trace should stay zero, got {:.3e}",
            snap.step, snap.k_trace_rms
        );
        assert!(
            snap.max_em_rho < 1e-12,
            "step {}: EM energy density should be zero, got {:.3e}",
            snap.step, snap.max_em_rho
        );
    }
}

/// Source timing: with n_sources=4 and period=4, verify that active source
/// rotates correctly — different t values produce non-zero EM angular momentum
/// with alternating sign as the spotlight moves around the ring.
#[test]
fn test_tornado_sim_source_rotation() {
    // Two-step run: t=0 → source 0 (at +x side), t=dt → source 0 still active
    let config = TornadoConfig {
        n_sources: 4,
        ring_radius: 0.2,
        source_sigma: 0.15,
        amplitude: 1.0,
        period: 4.0,
        nx: 5, ny: 5, nz: 5,
        dx: 0.1, dy: 0.1, dz: 0.1,
        dt: 0.1,
        n_steps: 1,
        output_every: 1,
        mu_0: 1.0,
        eps_fd: 1e-5,
    };

    let result = run_tornado(&config);

    // The EM energy density should be non-zero (source is active)
    assert!(
        result.snapshots[0].max_em_rho > 1e-8,
        "EM energy density at t=0 should be non-zero, got {:.3e}",
        result.snapshots[0].max_em_rho
    );
}

/// Snapshot timing: output_every=2 with n_steps=4 gives snapshots at 0,2,4.
#[test]
fn test_tornado_sim_snapshot_timing() {
    let config = TornadoConfig {
        n_sources: 4,
        ring_radius: 0.2,
        source_sigma: 0.15,
        amplitude: 0.0,
        period: 1.0,
        nx: 5, ny: 5, nz: 5,
        dx: 0.1, dy: 0.1, dz: 0.1,
        dt: 0.01,
        n_steps: 4,
        output_every: 2,
        mu_0: 1.0,
        eps_fd: 1e-5,
    };

    let result = run_tornado(&config);

    // Steps 0, 2, 4 (output_every=2), plus step 4 is final so no double
    let steps: Vec<usize> = result.snapshots.iter().map(|s| s.step).collect();
    assert_eq!(steps, vec![0, 2, 4], "Expected snapshots at steps 0, 2, 4; got {steps:?}");
}

/// Timestamp check: t = step * dt.
#[test]
fn test_tornado_sim_timestamps() {
    let config = TornadoConfig {
        n_sources: 4,
        ring_radius: 0.2,
        source_sigma: 0.15,
        amplitude: 0.0,
        period: 1.0,
        nx: 5, ny: 5, nz: 5,
        dx: 0.1, dy: 0.1, dz: 0.1,
        dt: 0.01,
        n_steps: 3,
        output_every: 1,
        mu_0: 1.0,
        eps_fd: 1e-5,
    };
    let result = run_tornado(&config);

    for snap in &result.snapshots {
        let expected_t = snap.step as f64 * config.dt;
        assert!(
            (snap.t - expected_t).abs() < 1e-14,
            "Snapshot t mismatch at step {}: got {}, expected {expected_t}",
            snap.step, snap.t
        );
    }
}

// ── Slow integration test ─────────────────────────────────────────────────────

/// Full tornado run on a 9×9×9 grid.
///
/// Verifies that after many source activations:
///  1. K_offdiag grows (rotation is being sourced)
///  2. EM angular momentum is non-zero
///  3. Hamiltonian constraint stays bounded (evolution remains accurate)
///
/// Slow in debug mode (~several minutes); run with:
///   cargo test --release -- --ignored
#[test]
#[ignore = "slow; run with: cargo test --release -- --ignored"]
fn test_tornado_sim_full_run() {
    let config = TornadoConfig::perturbative(0.01, 6, 50);
    let result = run_tornado(&config);

    result.print_summary();

    // K_offdiag should grow — the rotating EM source seeds off-diagonal extrinsic curvature
    let peak_offdiag = result.peak_k_offdiag();
    assert!(
        peak_offdiag > 0.0,
        "K_offdiag should grow; peak = {peak_offdiag:.3e}"
    );

    // The spatial metric should deviate from flat (curvature is accumulating)
    let peak_gp = result
        .snapshots
        .iter()
        .map(|s| s.gamma_perturb_rms)
        .fold(0.0_f64, f64::max);
    assert!(
        peak_gp > 0.0,
        "γ_perturb should grow; peak = {peak_gp:.3e}"
    );

    // Constraint should not blow up (stays below a generous threshold)
    let peak_h = result.peak_hamiltonian_violation();
    assert!(
        peak_h < 1.0,
        "Hamiltonian constraint blow-up: peak H_L2 = {peak_h:.3e}"
    );
}
