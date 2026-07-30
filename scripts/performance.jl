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
D = 4
max_time = 75

# Folded path: W_η = (exp(-iηP) ⊗ exp(iηP*)) · W
fg_folded = peturb_folded(fold(gate), P, ϵ)

# Correlation test using folded-perturbed gate
dir = "tmp/folded_peturbed"

V = gate.V
d = dim(V)
Z = TensorMap(Dict(U1Irrep(0) => fill(1,1,1),
              U1Irrep(1) => fill(-1,1,1)), V ← V) / sqrt(d)

get_tiles(fg_folded, D, dir)

println("="^60)

let
    
    for t in 1:max_time
        tile_sum = zero(ComplexF64)
        
        println("\nComparing tile-based vs direct correlation for t=$t:")
        for x in -t+1:1:t

            tile_result = two_point_correlation(dir, x, t, Z, Z, D)
            tile_sum += tile_result
            println("x=$x: tile=$(real(tile_result))")
        end

        println("="^60)
        println("Charge conservation check (sum over all x):")
        println("  Tile sum:   $tile_sum")
    end
end