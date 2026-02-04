module DualUnitary

using LinearAlgebra
using TensorKit
using Serialization

include("gates.jl")

export Gate,
       FoldedGate,
       soliton_dual_unitary_U1_qubit,
       is_dual_unitary,
       fold,
       is_unital,
       is_dual_unital,
       dual_fold,
       has_soliton,
       pertubation,
       peturb

include("tiles.jl")

export hd_tile,
       hx_tile,
       vd_tile,
       vx_tile,
       get_tiles,
       load_tiles,
       TileSet

include("paths.jl")

export path_set,
       all_compositions,
       valid_path_pairs

include("correlation.jl")

export operator_string,
       TileConfig,
       build_tile_config,
       skeleton,
       two_point_correlation,
       generate_paths

end # module DualUnitary
