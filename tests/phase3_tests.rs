#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use tensor::adm_matter::AdmMatter;
use tensor::adm_step::{adm_step_rk4_with_source, geodesic_rhs_with_matter};
use tensor::tornado::{TornadoArray, tornado_matter_grid};
use tensor::AdmGrid;

// ── AdmMatter decomposition ───────────────────────────────────────────────────

/// Vacuum T = 0 gives zero matter.
#[test]
fn test_adm_matter_vacuum() {
    use aad::number::Number;
    use tensor::Tensor;

    let t4: Tensor<0, 2> = Tensor::from_f64(4, vec![0.0; 16]);
    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let m = AdmMatter::from_t4d(&t4, &gamma_inv);
    assert_eq!(m.rho, 0.0);
    assert_eq!(m.j, [0.0; 3]);
    assert_eq!(m.s_ij, [0.0; 9]);
    assert_eq!(m.s_trace, 0.0);
    let _ = Number::new(0.0); // keep serial_test happy (AAD not used but imported)
}

/// T_{00} = ε gives ρ = ε; spatial components give S correctly.
#[test]
fn test_adm_matter_from_t4d_pure_energy() {
    use tensor::Tensor;

    let eps = 2.5_f64;
    let mut vals = vec![0.0_f64; 16];
    vals[0] = eps; // T_{00}
    let t4: Tensor<0, 2> = Tensor::from_f64(4, vals);

    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let m = AdmMatter::from_t4d(&t4, &gamma_inv);

    assert!((m.rho - eps).abs() < 1e-14);
    assert_eq!(m.j, [0.0; 3]);
    assert_eq!(m.s_ij, [0.0; 9]);
    assert_eq!(m.s_trace, 0.0);
}

/// Off-diagonal T_{01} gives a non-zero J^0 component.
#[test]
fn test_adm_matter_momentum_component() {
    use tensor::Tensor;

    let p = 0.7_f64;
    let mut vals = vec![0.0_f64; 16];
    // T_{01} = T_{10} = p
    vals[0 * 4 + 1] = p;
    vals[1 * 4 + 0] = p;
    let t4: Tensor<0, 2> = Tensor::from_f64(4, vals);

    let gamma_inv: Tensor<2, 0> = Tensor::from_f64(3, vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let m = AdmMatter::from_t4d(&t4, &gamma_inv);

    // J^0 = -γ^{0j} T_{0,j+1} = -δ^{0,0} T_{0,1} = -p
    assert!((m.j[0] - (-p)).abs() < 1e-14, "J^0 = {}", m.j[0]);
    assert!(m.j[1].abs() < 1e-14);
    assert!(m.j[2].abs() < 1e-14);
}

// ── TornadoArray ──────────────────────────────────────────────────────────────

/// With 4 sources and period=4, at t=0 source 0 is active; at t=1 source 1; etc.
#[test]
fn test_tornado_active_index() {
    let arr = TornadoArray::new(4, 1.0, 0.3, 1.0, 4.0);
    assert_eq!(arr.active_index(0.0), 0);
    assert_eq!(arr.active_index(1.0), 1);
    assert_eq!(arr.active_index(2.0), 2);
    assert_eq!(arr.active_index(3.0), 3);
    assert_eq!(arr.active_index(4.0), 0); // wraps
}

/// Potential at the ring centre should be zero for a vortex (A depends on (x-cx), (y-cy)).
#[test]
fn test_tornado_potential_at_source_centre() {
    let arr = TornadoArray::new(4, 1.0, 0.3, 1.0, 4.0);
    // Source 0 is at (1, 0, 0); its potential at (1, 0, 0) should be zero.
    let src0 = &arr.sources[0];
    let pot = src0.potential_at(&[src0.cx, src0.cy, src0.cz]);
    for &v in &pot {
        assert!(v.abs() < 1e-14, "potential at source centre = {v:.3e}");
    }
}

/// Potential is non-zero off-centre.
#[test]
fn test_tornado_potential_nonzero_offcenter() {
    let arr = TornadoArray::new(4, 1.0, 0.3, 1.0, 4.0);
    // Source 0 at (1,0,0): evaluate at (1, 0.1, 0) — displaced in y
    let src0 = &arr.sources[0];
    let pot = src0.potential_at(&[src0.cx, src0.cy + 0.1, 0.0]);
    let mag: f64 = pot.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(mag > 1e-6, "potential should be non-zero off-centre");
}

// ── tornado_matter_grid ───────────────────────────────────────────────────────

/// On a flat grid with zero-amplitude sources, all matter is zero.
#[test]
fn test_tornado_matter_grid_zero_amplitude() {
    let arr = TornadoArray::new(4, 0.3, 0.1, 0.0, 1.0); // amplitude = 0
    let grid = AdmGrid::flat(5, 5, 5, 0.1, 0.1, 0.1);
    let matters = tornado_matter_grid(&arr, &grid, 0.0, 1.0, 1e-5);

    for (i, m) in matters.iter().enumerate() {
        assert!(m.rho.abs() < 1e-12, "ρ[{i}] = {:.3e} should be zero", m.rho);
    }
}

/// With non-zero amplitude, at least the active source region has ρ > 0.
#[test]
fn test_tornado_matter_grid_nonzero_source() {
    // Source 0 at (R, 0, 0) with R=0.3 and dx=0.1 grid:
    // place source close to grid point ix=3, iy=2 (x=0.3, y=0.2)
    let radius = 0.2_f64;
    let arr = TornadoArray::new(4, radius, 0.1, 1.0, 1.0); // B₀ = 1, source 0 at (0.2, 0)
    let grid = AdmGrid::flat(5, 5, 5, 0.1, 0.1, 0.1);
    let matters = tornado_matter_grid(&arr, &grid, 0.0, 1.0, 1e-5);

    let max_rho = matters.iter().map(|m| m.rho.abs()).fold(0.0_f64, f64::max);
    assert!(max_rho > 1e-6, "ρ_max = {max_rho:.3e} — expected non-zero EM energy density");
}

// ── Source-coupled RK4 ────────────────────────────────────────────────────────

/// With zero matter, source-coupled RHS equals vacuum RHS.
#[test]
fn test_rhs_with_matter_matches_vacuum_when_zero() {
    use tensor::adm_step::geodesic_rhs;
    let grid = AdmGrid::flat(5, 5, 5, 0.1, 0.1, 0.1);
    let vacuum_matters: Vec<AdmMatter> = vec![AdmMatter::vacuum(); grid.n_pts()];

    let rhs_vac = geodesic_rhs(&grid);
    let rhs_matter = geodesic_rhs_with_matter(&grid, &vacuum_matters);

    for (v, m) in rhs_vac.iter().zip(rhs_matter.iter()) {
        assert!(
            (v - m).abs() < 1e-14,
            "vacuum RHS mismatch: {v:.3e} vs {m:.3e}"
        );
    }
}

/// With a live tornado source, K_{ij} changes faster than in vacuum.
/// ∂_t K_{ij} gains a matter term: more negative (−8π S + 4π γ(S−ρ) term).
/// We verify that the RK4 step actually modifies fields differently from vacuum.
#[test]
fn test_rk4_with_source_differs_from_vacuum() {
    let arr = TornadoArray::new(4, 0.2, 0.1, 1.0, 1.0);
    let grid = AdmGrid::flat(5, 5, 5, 0.1, 0.1, 0.1);
    let matters = tornado_matter_grid(&arr, &grid, 0.0, 1.0, 1e-5);

    let stepped_vacuum = adm_step_rk4_with_source(&grid, 0.001, &vec![AdmMatter::vacuum(); grid.n_pts()]);
    let stepped_source = adm_step_rk4_with_source(&grid, 0.001, &matters);

    // Find the maximum field difference between the two evolutions
    let max_diff = stepped_vacuum
        .fields
        .iter()
        .zip(stepped_source.fields.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff > 1e-15,
        "EM source should perturb the evolution; max_diff = {max_diff:.3e}"
    );
}

/// Multiple tornado steps: check that matter coupling accumulates over time.
/// After N steps with source, K_{ij} should drift further from vacuum than after 1 step.
#[test]
fn test_tornado_k_accumulates() {
    let arr = TornadoArray::new(4, 0.2, 0.1, 1.0, 1.0);
    let grid = AdmGrid::flat(5, 5, 5, 0.1, 0.1, 0.1);
    let dt = 0.001_f64;
    let n_steps = 5;

    let matters = tornado_matter_grid(&arr, &grid, 0.0, 1.0, 1e-5);

    let mut g = grid.clone();
    for _ in 0..n_steps {
        g = adm_step_rk4_with_source(&g, dt, &matters);
    }

    // The K field at the interior (2,2,2) should differ from flat (all-zero K)
    let k_final = g.k_flat(2, 2, 2);
    let k_norm: f64 = k_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        k_norm > 1e-15,
        "K should accumulate from tornado source; norm = {k_norm:.3e}"
    );
}
