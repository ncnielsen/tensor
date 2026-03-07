#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use serial_test::serial;
use tensor::riemann;
use tensor::Christoffel;
use tensor::ChristoffelDerivative;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Zero Christoffel symbols in 2D.
fn zero_gamma_2d() -> Christoffel {
    Christoffel::from_f64(2, vec![0.0; 8])
}

/// Zero Christoffel derivatives in 2D.
fn zero_partial_gamma_2d() -> ChristoffelDerivative {
    ChristoffelDerivative::from_f64(2, vec![0.0; 16])
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Flat Cartesian 2D: all Γ and ∂Γ are zero, so R = 0 trivially.
#[test]
#[serial]
fn test_riemann_flat_cartesian() {
    let r = riemann(&zero_gamma_2d(), &zero_partial_gamma_2d());
    for c in &r.components {
        assert_eq!(c.result, 0.0);
    }
}

/// Flat 2D space in polar coordinates at r = 2.
///
/// Non-zero Christoffel symbols (Γ^ρ_{κμ}, layout [ρ,κ,μ]):
///   Γ^r_{θθ} = Γ^0_{11} = -2         flat 3  (0*4+1*2+1)
///   Γ^θ_{rθ} = Γ^1_{01} = 0.5        flat 5  (1*4+0*2+1)
///   Γ^θ_{θr} = Γ^1_{10} = 0.5        flat 6  (1*4+1*2+0)
///
/// Non-zero Christoffel derivatives (∂_ν Γ^ρ_{κμ}, layout [ρ,κ,μ,ν]):
///   ∂_r Γ^r_{θθ} = -1    → [0,1,1,0]  flat 6  (0*8+1*4+1*2+0)
///   ∂_r Γ^θ_{rθ} = -0.25 → [1,0,1,0]  flat 10 (1*8+0*4+1*2+0)
///   ∂_r Γ^θ_{θr} = -0.25 → [1,1,0,0]  flat 12 (1*8+1*4+0*2+0)
///
/// Despite non-zero Γ and ∂Γ, space is flat so all R^ρ_{σμν} = 0.
///
/// Verification of the one independent component R^r_{θrθ} = R^0_{101}:
///   R^0_{101} = ∂_0 Γ^0_{11} − ∂_1 Γ^0_{01}
///             + Γ^0_{0λ}Γ^λ_{11} − Γ^0_{1λ}Γ^λ_{01}
///           = (−1) − 0 + (0·(−2)+0·0) − (0·0+(−2)·0.5)
///           = −1 + 1 = 0  ✓
#[test]
#[serial]
fn test_riemann_flat_polar_coordinates() {
    // Christoffel symbols at r = 2
    // layout [ρ, κ, μ], flat index = ρ*4 + κ*2 + μ
    let mut gamma_vals = vec![0.0f64; 8];
    gamma_vals[3] = -2.0;  // Γ^0_{11} = -r = -2
    gamma_vals[5] = 0.5;   // Γ^1_{01} = 1/r = 0.5
    gamma_vals[6] = 0.5;   // Γ^1_{10} = 1/r = 0.5
    let gamma = Christoffel::from_f64(2, gamma_vals);

    // Christoffel derivatives at r = 2
    // layout [ρ, κ, μ, ν], flat index = ρ*8 + κ*4 + μ*2 + ν
    let mut pg_vals = vec![0.0f64; 16];
    pg_vals[6]  = -1.0;    // ∂_r Γ^r_{θθ} = -1
    pg_vals[10] = -0.25;   // ∂_r Γ^θ_{rθ} = -1/r² = -0.25
    pg_vals[12] = -0.25;   // ∂_r Γ^θ_{θr} = -1/r² = -0.25
    let partial_gamma = ChristoffelDerivative::from_f64(2, pg_vals);

    let r = riemann(&gamma, &partial_gamma);

    for c in &r.components {
        assert!(
            c.result.abs() < 1e-12,
            "Expected R = 0 for flat space, got {}",
            c.result
        );
    }
}
