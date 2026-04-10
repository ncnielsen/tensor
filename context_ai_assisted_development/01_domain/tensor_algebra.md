# Tensor Algebra

Foundational index notation and operations for tensor computation in physics.

## Index Notation and Einstein Summation

A tensor of rank (M, N) has M **contravariant** (upper) indices and N **covariant**
(lower) indices:

```
T^{i j}_{k l}    — rank (2, 1) would be T^{ij}_k, but this example is (2, 2)
```

**Einstein summation convention:** when an index appears once upper and once lower
in a product, summation over that index is implied:

```
A^i B_i  =  Σ_i  A^i B_i       (scalar result)
T^{ij} v_j  =  Σ_j  T^{ij} v_j   (vector result, free index i)
```

Only contract between one upper and one lower index — never two upper or two lower.

## Dimension

In 4D spacetime: indices run 0..3 (time = 0, space = 1,2,3).
In 3D spatial slices (ADM formalism): indices run 0..2.

A rank (M, N) tensor in dimension d has d^(M+N) components.

## The Metric Tensor g_μν

The metric is a symmetric rank (0, 2) tensor that defines distances and angles:

```
ds² = g_{μν} dx^μ dx^ν
g_{μν} = g_{νμ}     (symmetric: d(d+1)/2 independent components)
```

The inverse metric g^{μν} satisfies:

```
g^{μα} g_{αν} = δ^μ_ν    (Kronecker delta)
```

### Raising and Lowering Indices

The metric converts between contravariant and covariant:

```
v^μ = g^{μν} v_ν       (raise with inverse metric)
v_μ = g_{μν} v^ν       (lower with metric)
T^μ_{ν} = g^{μα} T_{αν}   (raise first index of a (0,2) tensor)
```

## Outer Product

The outer product of a rank (M1, N1) tensor and a rank (M2, N2) tensor produces
a rank (M1+M2, N1+N2) tensor:

```
(A ⊗ B)^{i1...iM1, j1...jM2}_{k1...kN1, l1...lN2}
    = A^{i1...iM1}_{k1...kN1} · B^{j1...jM2}_{l1...lN2}
```

No summation — just multiply every pair of components.

## Contraction

Contraction pairs one upper index with one lower index and sums, reducing rank
by (1, 1):

```
contract(T^{ij}_{kl}, upper=1, lower=0)  =  Σ_a  T^{ia}_{al}  →  rank (1, 1)
```

The **trace** of a (1, 1) tensor is contraction of its only upper with its only
lower index:

```
tr(T) = T^i_i = Σ_i T^i_i    (scalar)
```

## Symmetries

A tensor is **symmetric** in two indices if swapping them leaves it unchanged:

```
S_{μν} = S_{νμ}    (symmetric in lower indices)
```

A tensor is **antisymmetric** if swapping introduces a sign flip:

```
A_{μν} = -A_{νμ}   (antisymmetric / skew-symmetric)
```

Key symmetric tensors in GR: g_{μν}, R_{μν} (Ricci), G_{μν} (Einstein), T_{μν}
(stress-energy), K_{ij} (extrinsic curvature).

Key antisymmetric tensor: F_{μν} (Faraday/electromagnetic field tensor).

## Kronecker Delta and Levi-Civita

```
δ^μ_ν = 1 if μ = ν, 0 otherwise     (identity as a (1,1) tensor)

ε_{μνρσ} = totally antisymmetric symbol (±1 or 0)
    ε_{0123} = +1, odd permutation → -1, repeated index → 0
```

## Storage Layout Convention

For a rank (M, N) tensor in dimension d, components are stored in a flat array
of length d^(M+N), in **row-major** order with **upper indices first**:

```
flat_index = i1 * d^(M+N-1) + i2 * d^(M+N-2) + ... + jN
```

Index order: (upper_1, upper_2, ..., upper_M, lower_1, lower_2, ..., lower_N).
