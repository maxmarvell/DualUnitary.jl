
using Revise
using DualUnitary
using TensorKit

# gate = soliton_dual_unitary_U1_qubit(0.5, 0.5)
# peurbed = peturb(gate)

# fg = fold(peurbed)

# # get_tiles(fg, 3, "tmp/peturbed")

V = U1Space(0 => 1, 1 => 1)
Z = TensorMap(Dict(U1Irrep(0) => fill(1,1,1),
              U1Irrep(1) => fill(-1,1,1)), V ← V) / sqrt(2)

@show two_point_correlation("tmp/peturbed", 5, 5, Z, Z, 3)