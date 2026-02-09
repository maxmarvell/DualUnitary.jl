using Revise
using DualUnitary
using TensorKit
using Serialization
using LinearAlgebra

# --- Folded perturbation (Eq. 30-31) ---
println("="^60)
println("Folded perturbation test (Eq. 30 vs 31)")
println("="^60)

gate = soliton_dual_unitary_U1_qubit(0.5, 0.5)
P = random_perturbation(gate)
ϵ = 10
D = 3
t = 10

# Unfolded path: U_η = U · exp(iηP), then fold
pgate = peturb(gate, ϵ; P=P)
fg_unfolded = fold(pgate)

# Folded path: W_η = (exp(-iηP) ⊗ exp(iηP*)) · W
fg_folded = peturb_folded(fold(gate), P, ϵ)

println("\nConsistency check:")
println("  W_unfolded ≈ W_folded: ", fg_unfolded.W ≈ fg_folded.W)
println("  Max difference: ", maximum(abs.(convert(Array, fg_unfolded.W - fg_folded.W))))

# Correlation test using folded-perturbed gate
dir = "tmp/folded_peturbed"

V = gate.V
d = dim(V)
Z = TensorMap(Dict(U1Irrep(0) => fill(1,1,1),
              U1Irrep(1) => fill(-1,1,1)), V ← V) / sqrt(d)

get_tiles(fg_folded, D, dir)

println("\nComparing tile-based vs direct correlation for t=$t:")
println("="^60)

let
    tile_sum = zero(ComplexF64)
    direct_sum = zero(ComplexF64)

    for x in -t+1:1:t
        tile_result = two_point_correlation(dir, x, t, Z, Z, D)
        direct_result = two_point_correlation(fg_folded, x, t, Z, Z)

        tile_sum += tile_result
        direct_sum += direct_result

        diff = abs(tile_result - direct_result)
        status = diff < 1e-10 ? "✓" : "✗ MISMATCH"
        println("x=$x: tile=$(real(tile_result)), direct=$(real(direct_result)), diff=$(diff) $status")
    end

    println("="^60)
    println("Charge conservation check (sum over all x):")
    println("  Tile sum:   $tile_sum")
    println("  Direct sum: $direct_sum")
    println("  Difference: $(abs(tile_sum - direct_sum))")
end