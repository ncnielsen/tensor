//! Performance benchmarks — run with: cargo run --release --example bench
//!
//! Zero-dependency timing using std::time::Instant. Measures the hot paths
//! that matter for the tornado simulation goal:
//!   1. Core tensor ops (allocation-heavy: contract, outer, riemann)
//!   2. Christoffel + curvature pipeline (per-point cost)
//!   3. Full grid RK4 step at increasing N (the real workload)
//!
//! The grid numbers are the headline: they tell whether an N³ tornado run
//! is feasible at the current allocation-per-element design.

use std::alloc::{GlobalAlloc, Layout};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use mimalloc::MiMalloc;

/// Counting allocator: delegates to mimalloc (per-thread heaps → low cross-
/// thread contention for the many small allocations the grid makes) and tallies
/// allocation calls. The counter is gated by `COUNTING_ON` so it stays *off*
/// during timing — otherwise the two atomics, hit by every alloc across all
/// rayon threads, bounce cache lines and fake an allocator bottleneck. On only
/// for the dedicated, single-threaded allocation-count measurement.
///
/// The real tornado binary should set `#[global_allocator] static: MiMalloc`
/// directly (without the counting wrapper).
struct Counting;
static MI: MiMalloc = MiMalloc;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static COUNTING_ON: AtomicBool = AtomicBool::new(false);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING_ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        }
        unsafe { MI.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { MI.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

use solver::grid::{adm_step_rk4, AdmGrid};
use tensor_core::{
    christoffel::Christoffel,
    curvature::{riemann, ricci_tensor, ChristoffelDerivative},
    metric::invert_metric,
    ops::{contract, outer},
    tensor::Tensor,
};

/// Time `iters` calls of `f`, return nanoseconds per call.
fn bench<F: FnMut()>(iters: u64, mut f: F) -> f64 {
    // warm up
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_nanos() as f64 / iters as f64
}

/// A mildly curved 4D metric (Schwarzschild-flavoured) + its Christoffel/∂Γ,
/// so the curvature machinery does real work (timing is structure-bound, not
/// value-bound — there are no data-dependent branches).
fn curved_4d() -> (Tensor<2, 0>, Christoffel, ChristoffelDerivative) {
    let r = 10.0_f64;
    let theta = std::f64::consts::FRAC_PI_4;
    let f = 1.0 - 2.0 / r;
    let sin_th = theta.sin();

    let mut g = Tensor::<0, 2>::new(4);
    g.set_component(&[0, 0], -f);
    g.set_component(&[1, 1], 1.0 / f);
    g.set_component(&[2, 2], r * r);
    g.set_component(&[3, 3], r * r * sin_th * sin_th);
    let g_inv = invert_metric(&g);

    let mut partials: Vec<Tensor<0, 2>> = (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();
    partials[1].set_component(&[0, 0], -2.0 / (r * r));
    partials[1].set_component(&[1, 1], -2.0 / (r * r * f * f));
    partials[1].set_component(&[2, 2], 2.0 * r);
    partials[1].set_component(&[3, 3], 2.0 * r * sin_th * sin_th);

    let gamma = Christoffel::from_metric(&g_inv, &partials);

    // Non-zero ∂Γ so riemann's derivative terms are exercised.
    let mut dgamma = ChristoffelDerivative::new(4);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for l in 0..4 {
                    dgamma.set_component(i, j, k, l, 0.001 * (i + j + k + l) as f64);
                }
            }
        }
    }
    (g_inv, gamma, dgamma)
}

fn bench_core() {
    println!("== core tensor ops (4D) ==");

    // contract <1,1> -> <0,0>  (trace; allocs a Vec per output elem + per term)
    let mut t11 = Tensor::<1, 1>::new(4);
    for i in 0..4 {
        t11.set_component(&[i, i], (i + 1) as f64);
    }
    let ns = bench(2_000_000, || {
        let s: Tensor<0, 0> = contract(black_box(&t11), 0, 0);
        black_box(s);
    });
    println!("  contract<1,1>->scalar : {ns:>10.1} ns");

    // outer <1,0> x <0,1> -> <1,1>
    let v = Tensor::<1, 0>::from_vec(4, vec![1.0, 2.0, 3.0, 4.0]);
    let w = Tensor::<0, 1>::from_vec(4, vec![5.0, 6.0, 7.0, 8.0]);
    let ns = bench(2_000_000, || {
        let t: Tensor<1, 1> = outer(black_box(&v), black_box(&w));
        black_box(t);
    });
    println!("  outer  ->  <1,1>      : {ns:>10.1} ns");

    // invert_metric (Gaussian elimination, 4x4)
    let (g_inv0, gamma, dgamma) = curved_4d();
    let mut g = Tensor::<0, 2>::new(4);
    for i in 0..4 {
        g.set_component(&[i, i], if i == 0 { -0.8 } else { (i + 1) as f64 });
    }
    let ns = bench(1_000_000, || {
        let gi = invert_metric(black_box(&g));
        black_box(gi);
    });
    println!("  invert_metric (4x4)   : {ns:>10.1} ns");

    // Christoffel::from_metric (4D)
    let mut partials: Vec<Tensor<0, 2>> = (0..4).map(|_| Tensor::<0, 2>::new(4)).collect();
    partials[1].set_component(&[0, 0], 0.1);
    let ns = bench(500_000, || {
        let c = Christoffel::from_metric(black_box(&g_inv0), black_box(&partials));
        black_box(c);
    });
    println!("  Christoffel::from_metric: {ns:>8.1} ns");

    // riemann (O(dim^5)) + ricci
    let ns = bench(200_000, || {
        let r = riemann(black_box(&gamma), black_box(&dgamma));
        black_box(r);
    });
    println!("  riemann (4D, O(d^5))  : {ns:>10.1} ns");

    let riem = riemann(&gamma, &dgamma);
    let ns = bench(500_000, || {
        let rc = ricci_tensor(black_box(&riem));
        black_box(rc);
    });
    println!("  ricci_tensor          : {ns:>10.1} ns");
}

fn bench_grid() {
    println!("\n== grid: adm_step_rk4 (geodesic, 3D) ==");
    println!("  {:>4}  {:>9}  {:>12}  {:>14}", "N", "interior", "ms / step", "us / pt / step");

    for &n in &[5usize, 9, 13, 17, 25] {
        let mut grid = AdmGrid::new(n, 0.1);
        grid.init_flat_all();
        // Perturb interior K slightly so we're not on a trivial all-zero path.
        let interior = n - 4;
        let npts = interior * interior * interior;

        // pick iteration count so total time stays reasonable
        let iters: u64 = if npts <= 1 {
            2000
        } else if npts <= 200 {
            200
        } else if npts <= 3000 {
            20
        } else {
            5
        };

        let dt = 1e-4;
        // warm
        adm_step_rk4(&mut grid, dt);
        let t0 = Instant::now();
        for _ in 0..iters {
            adm_step_rk4(black_box(&mut grid), dt);
        }
        let per_step_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let per_pt_us = per_step_ms * 1e3 / npts as f64;
        println!("  {n:>4}  {npts:>9}  {per_step_ms:>12.4}  {per_pt_us:>14.3}");
    }

    println!("\n== projection: full tornado run ==");
    let per_pt_us = {
        // re-measure at N=17 for a stable per-point number
        let n = 17usize;
        let mut grid = AdmGrid::new(n, 0.1);
        grid.init_flat_all();
        let npts = (n - 4) * (n - 4) * (n - 4);
        adm_step_rk4(&mut grid, 1e-4);
        let t0 = Instant::now();
        for _ in 0..20 {
            adm_step_rk4(&mut grid, 1e-4);
        }
        t0.elapsed().as_secs_f64() * 1e6 / 20.0 / npts as f64
    };
    for &(n, steps) in &[(50usize, 1000u64), (50, 10_000), (100, 10_000)] {
        let npts = (n - 4) * (n - 4) * (n - 4);
        let secs = per_pt_us * 1e-6 * npts as f64 * steps as f64;
        println!(
            "  N={n:>3}  {steps:>6} steps  ({npts:>8} pts)  ->  {:>10.1} s  ({:.2} h)",
            secs,
            secs / 3600.0
        );
    }
}

fn bench_allocs() {
    println!("\n== heap allocations per RK4 step (N=9, 125 interior pts) ==");
    let mut grid = AdmGrid::new(9, 0.1);
    grid.init_flat_all();
    adm_step_rk4(&mut grid, 1e-4); // warm

    let a0 = ALLOCS.load(Ordering::Relaxed);
    let b0 = BYTES.load(Ordering::Relaxed);
    COUNTING_ON.store(true, Ordering::SeqCst);
    adm_step_rk4(&mut grid, 1e-4);
    COUNTING_ON.store(false, Ordering::SeqCst);
    let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
    let bytes = BYTES.load(Ordering::Relaxed) - b0;

    let npts = 125u64;
    let per_pt = allocs as f64 / npts as f64 / 4.0; // 4 RK4 stages
    println!("  allocations / step      : {allocs}");
    println!("  bytes / step            : {bytes}");
    println!("  allocations / pt / stage: {per_pt:.1}");
}

fn main() {
    println!("tensor performance benchmarks (release)\n");
    bench_core();
    bench_grid();
    bench_allocs();
}
