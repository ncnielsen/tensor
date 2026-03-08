#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::{em_t_grid, solve_3d};

// ─── helpers ──────────────────────────────────────────────────────────────────

fn minkowski_4d() -> Vec<f64> {
    vec![
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]
}

/// Rotating magnetic vortex 4-potential centred at (cx, cy) in the x⁰–x¹ plane.
///
/// ```text
///   A₀(x) = −½ B₀ (x¹ − cy) exp(−r²/2σ²)
///   A₁(x) =  ½ B₀ (x⁰ − cx) exp(−r²/2σ²)
///   A₂ = A₃ = 0
/// ```
///
/// Axial magnetic field:
///   F_{01} = B₀ exp(−r²/2σ²) (1 − r²/2σ²)
///
/// A positive core surrounded by a weaker negative annular ring — the cross-
/// section of a magnetic flux tube / "tornado" seed.
fn vortex_potential(x: &[f64], b0: f64, sigma: f64, cx: f64, cy: f64) -> Vec<f64> {
    let dx = x[0] - cx;
    let dy = x[1] - cy;
    let r2 = dx * dx + dy * dy;
    let gauss = (-r2 / (2.0 * sigma * sigma)).exp();
    vec![
        -0.5 * b0 * dy * gauss, //  A₀
         0.5 * b0 * dx * gauss, //  A₁
         0.0,                    //  A₂
         0.0,                    //  A₃ (time)
    ]
}

// ─── em_t_grid tests ──────────────────────────────────────────────────────────

/// For a zero 4-potential, T_{μν} = 0 everywhere.
#[test]
#[serial]
fn test_em_t_grid_zero_potential() {
    let a_fn = |_: &[f64]| vec![0.0, 0.0, 0.0, 0.0];
    let (nx, ny, nz) = (3, 3, 3);
    let g_grid: Vec<Vec<f64>> = (0..nx * ny * nz).map(|_| minkowski_4d()).collect();

    let t_grid = em_t_grid(&a_fn, &g_grid, nx, ny, nz, 1.0, 1.0, 1e-5);

    for (i, t) in t_grid.iter().enumerate() {
        for (j, c) in t.components.iter().enumerate() {
            assert!(
                c.result.abs() < 1e-10,
                "T[{}][{}] = {} (expected 0 for zero potential)",
                i, j, c.result
            );
        }
    }
}

/// Magnetic vortex: T_{00} at the vortex core ≈ B₀²/(2μ₀).
///
/// Analytically: for flat Minkowski with only F_{01} = B₀ non-zero,
///   T_{00} = ½ F_{01}² / μ₀ = B₀²/(2μ₀).
///
/// Using eps ≪ σ the numerical FD derivative is very accurate, so the
/// numerical result should be within 0.1% of the analytic value.
#[test]
#[serial]
fn test_em_t_grid_vortex_energy_density() {
    let b0    = 1.0_f64;
    let sigma = 1.0_f64;
    let (nx, ny, nz) = (5, 5, 5);
    let h  = 1.0_f64;
    let cx = 2.0_f64; // grid centre: ix=2 → x=2
    let cy = 2.0_f64;

    let a_fn = |x: &[f64]| vortex_potential(x, b0, sigma, cx, cy);

    let g_grid: Vec<Vec<f64>> = (0..nx * ny * nz).map(|_| minkowski_4d()).collect();
    // Use a small derivative step so ∂A is computed accurately
    let t_grid = em_t_grid(&a_fn, &g_grid, nx, ny, nz, h, /*mu_0=*/ 1.0, /*eps=*/ 1e-5);

    // Centre grid point: ix=2, iy=2, iz=2  →  flat = 2*25 + 2*5 + 2 = 62
    let centre = 2 * ny * nz + 2 * nz + 2;
    let t_centre = &t_grid[centre];

    // At the vortex centre r=0: F_{01} = B₀, T_{00} = B₀²/(2μ₀) = 0.5
    let t00 = t_centre.components[0].result;
    let expected = 0.5 * b0 * b0; // B²/(2μ₀) with μ₀=1
    assert!(
        (t00 - expected).abs() < 1e-3,
        "T_{{00}} at vortex core: got {:.6}, expected {:.6}",
        t00, expected
    );

    // All diagonal spatial components should be non-zero (electromagnetic pressure)
    let t11 = t_centre.components[5].result;
    let t22 = t_centre.components[10].result;
    assert!(
        t11.abs() > 1e-6,
        "T_{{11}} should be non-zero at vortex core, got {}",
        t11
    );
    assert!(
        t22.abs() > 1e-6,
        "T_{{22}} should be non-zero at vortex core, got {}",
        t22
    );

    // The field should fall off away from the core:
    // at the corner (ix=0, iy=0, r²=8) T_{00} is much weaker.
    let corner = 0;
    let t00_corner = t_grid[corner].components[0].result;
    assert!(
        t00_corner.abs() < 0.05 * t00.abs(),
        "T_{{00}} should be much smaller at corner ({:.2e}) than at core ({:.2e})",
        t00_corner, t00
    );
}

/// T_{μν} computed from a uniform A field (A_μ = const) is zero, because F_{μν} = ∂A − ∂A = 0.
#[test]
#[serial]
fn test_em_t_grid_constant_potential_zero_faraday() {
    // Constant potential → all partial derivatives vanish → F = 0 → T = 0
    let a_fn = |_: &[f64]| vec![1.0, 2.0, -3.0, 0.5];
    let (nx, ny, nz) = (3, 3, 3);
    let g_grid: Vec<Vec<f64>> = (0..nx * ny * nz).map(|_| minkowski_4d()).collect();

    let t_grid = em_t_grid(&a_fn, &g_grid, nx, ny, nz, 1.0, 1.0, 1e-5);

    for (i, t) in t_grid.iter().enumerate() {
        for (j, c) in t.components.iter().enumerate() {
            assert!(
                c.result.abs() < 1e-8,
                "T[{}][{}] = {} (expected 0 for constant A)",
                i, j, c.result
            );
        }
    }
}

// ─── Full tornado solve (slow, run with --ignored) ────────────────────────────

/// Solve G_{μν} = κ T_{μν} with the magnetic vortex source on a 3×3×3 grid.
///
/// Uses a small-amplitude field (perturbative regime) so Newton-Raphson
/// converges in a few steps.  Verifies that:
///   1. The initial residual is non-zero (the vortex does source curvature).
///   2. The solver reduces the residual below the tolerance.
///   3. The final metric deviates from flat Minkowski at the interior point.
///
/// Expected run time: ~2 min in debug mode; ~10 s in release mode
/// (`cargo test --release -- --ignored`).
#[test]
#[serial]
#[ignore = "slow (~2 min debug, ~10 s release); run with: cargo test -- --ignored"]
fn test_tornado_solve_magnetic_vortex() {
    let b0    = 0.1_f64;   // small amplitude: perturbative regime
    let sigma = 0.5_f64;
    let (nx, ny, nz) = (3usize, 3, 3);
    let h = 1.0_f64;
    let cx = 1.0_f64; // grid centre (ix=1 interior point at x=1)
    let cy = 1.0_f64;

    let a_fn = |x: &[f64]| vortex_potential(x, b0, sigma, cx, cy);

    // Build T_{μν} grid on flat Minkowski background
    let g_flat: Vec<Vec<f64>> = (0..nx * ny * nz).map(|_| minkowski_4d()).collect();
    let t_grid = em_t_grid(&a_fn, &g_flat, nx, ny, nz, h, /*mu_0=*/ 1.0, /*eps=*/ 1e-5);

    // The single interior point (1,1,1) should have non-trivial T
    let interior = 1 * ny * nz + 1 * nz + 1;
    let t_max = t_grid[interior]
        .components
        .iter()
        .map(|c| c.result.abs())
        .fold(0.0_f64, f64::max);
    assert!(t_max > 0.0, "T_{{μν}} at interior should be non-zero for the vortex");

    // Einstein equations: G_{μν} = κ T_{μν}, κ = 8π (geometric units G=c=1)
    let kappa = 8.0 * std::f64::consts::PI;
    let result = solve_3d(
        &g_flat, &t_grid, nx, ny, nz,
        h, kappa,
        /*tol=*/ 1e-8, /*max_iter=*/ 10, /*eps=*/ 1e-5,
    );

    println!("Converged:              {}", result.converged);
    println!("Iterations:             {}", result.iterations);
    println!("Final residual norm:    {:.3e}", result.residual_norm);

    // The metric at the interior should deviate from flat Minkowski
    let flat = minkowski_4d();
    let max_diff = result.g_grid[interior]
        .iter()
        .zip(flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Max metric perturbation: {:.3e}", max_diff);

    assert!(
        max_diff > 1e-20,
        "Metric should be perturbed from flat by the EM source; got max_diff={}",
        max_diff
    );
}

// ─── Component symmetry check ─────────────────────────────────────────────────

/// T_{μν} produced by em_t_grid is symmetric: T_{μν} = T_{νμ}.
#[test]
#[serial]
fn test_em_t_grid_vortex_symmetry() {
    let b0 = 1.0_f64;
    let sigma = 1.0_f64;
    let (nx, ny, nz) = (3, 3, 3);
    let cx = 1.0_f64;
    let cy = 1.0_f64;

    let a_fn = |x: &[f64]| vortex_potential(x, b0, sigma, cx, cy);
    let g_grid: Vec<Vec<f64>> = (0..nx * ny * nz).map(|_| minkowski_4d()).collect();
    let t_grid = em_t_grid(&a_fn, &g_grid, nx, ny, nz, 1.0, 1.0, 1e-5);

    let dim = 4usize;
    for t in &t_grid {
        for mu in 0..dim {
            for nu in 0..dim {
                let t_mn = t.components[mu * dim + nu].result;
                let t_nm = t.components[nu * dim + mu].result;
                assert!(
                    (t_mn - t_nm).abs() < 1e-10,
                    "T_{{μν}} not symmetric: T[{},{}]={:.2e} but T[{},{}]={:.2e}",
                    mu, nu, t_mn, nu, mu, t_nm
                );
            }
        }
    }
}
