
using Revise
using DualUnitary
using TensorKit
using Serialization

dir = "tmp/peturbed"

gate = soliton_dual_unitary_U1_qubit(0.5, 0.5)
pgate = peturb(gate, 0.1)
fg = fold(pgate)

V = U1Space(0 => 1, 1 => 1)
d = dim(gate.V)
Z = TensorMap(Dict(U1Irrep(0) => fill(1,1,1),
              U1Irrep(1) => fill(-1,1,1)), V ← V) / sqrt(d)

D = 2

# Regenerate tiles to pick up any fixes
get_tiles(fg, D, dir)

# Compare tile-based vs direct methods
t = 2
println("\nComparing tile-based vs direct correlation for t=$t:")
println("="^60)



# Direct test: compare single tile application vs direct transfer matrices
println("\n" * "="^60)
println("Testing single tile vs direct transfer matrices:")
println("="^60)

let
    tile_sum = zero(ComplexF64)
    direct_sum = zero(ComplexF64)

    for x in -t+1:1:t
        tile_result = two_point_correlation(dir, x, t, Z, Z, D)
        direct_result = two_point_correlation(fg, x, t, Z, Z)

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