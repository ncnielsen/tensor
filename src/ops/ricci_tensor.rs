use crate::ops::contract::contract;
use crate::tensor::Tensor;

/// Ricci curvature tensor R_{σν}.
///
/// Defined as the contraction of the Riemann tensor over its upper index and
/// second lower index:
///
///   R_{σν} = Σ_ρ R^ρ_{σρν}
///
/// Takes a Riemann tensor of type Tensor<1,3> (layout [ρ, σ, μ, ν]) and
/// returns a Tensor<0,2> (layout [σ, ν]).
///
/// The Ricci tensor encodes how volumes are distorted by curvature and appears
/// directly in the Einstein field equations via the Einstein tensor.
pub fn ricci_tensor(r: &Tensor<1, 3>) -> Tensor<0, 2> {
    contract(r, 0, 1)
}
