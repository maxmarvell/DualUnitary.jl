
function operator_string(V::ElementarySpace, n::Int; op::Union{TensorMap, Nothing}=nothing, op_first::Bool=true)::TensorMap
    d = dim(V)
    I_vec = permute(id(V)/sqrt(d), ((1, 2), ()))
    if op === nothing
        return reduce(⊗, fill(I_vec, n))
    end
    op_vec = permute(op, ((1, 2), ()))
    if op_first
        return reduce(⊗, vcat([op_vec], fill(I_vec, n-1)))  # O ⊗ I ⊗ I ⊗ ...
    else
        return reduce(⊗, vcat(fill(I_vec, n-1), [op_vec]))  # I ⊗ I ⊗ ... ⊗ O
    end
end

struct TileConfig
    main::NamedTuple{(:hd,:hx,:vd,:vx), NTuple{4,TensorMap}}
    low::Union{Nothing, NamedTuple{(:hd,:hx,:vd,:vx), NTuple{4,TensorMap}}}
    left::Union{Nothing, NamedTuple{(:hd,:hx,:vd,:vx), NTuple{4,TensorMap}}}
    corner::Union{Nothing, NamedTuple{(:hd,:hx), NTuple{2,TensorMap}}}
    V::ElementarySpace
end

function build_tile_config(tile_dir::String, d::Int, x_h::Int, x_v::Int;
                           main=nothing)::TileConfig
    main_tiles = main === nothing ? load_tiles(tile_dir, d, d) : main
    V = domain(main_tiles.hd, 1)

    x_v_rem = x_v % d
    x_h_rem = x_h % d

    low = x_v_rem != 0 ? load_tiles(tile_dir, d, x_v_rem) : nothing
    left = x_h_rem != 0 ? load_tiles(tile_dir, x_h_rem, d) : nothing

    corner = nothing
    if x_v_rem != 0 && x_h_rem != 0
        cn = load_tiles(tile_dir, x_h_rem, x_v_rem)
        corner = (hd=cn.hd, hx=cn.hx)
    end

    return TileConfig(main_tiles, low, left, corner, V)
end

function _get_tiles(config::TileConfig, status::Symbol)
    status === :low && return config.low
    status === :left && return config.left
    return config.main
end

function _h_status(i::Int, x_h_rem::Int, x_v_rem::Int)::Symbol
    if i == 1
        x_v_rem != 0 && return :low
        x_h_rem != 0 && return :left
    end
    return :main
end

function _v_status(i::Int, x_h_rem::Int, h1::Int)::Symbol
    (i == 1 && x_h_rem != 0 && h1 == 1) && return :left
    return :main
end

function apply_tiles(a::TensorMap, config::TileConfig, l::Int;
                     horizontal::Bool, use_defect::Bool,
                     status::Symbol, x_h_rem::Int)::TensorMap
    tiles = _get_tiles(config, status)
    if horizontal
        direct = tiles.hd
        defect = tiles.hx
    else
        direct = tiles.vd
        defect = tiles.vx
    end

    n_direct = use_defect ? l - 1 : l

    for i in 1:n_direct
        if horizontal && status === :low && i == 1 && x_h_rem != 0
            a = config.corner.hd * a
        elseif horizontal && status === :left && i > 1
            a = config.main.hd * a
        else
            a = direct * a
        end
    end

    if use_defect
        if horizontal && status === :low && l == 1 && x_h_rem != 0
            a = config.corner.hx * a
        elseif horizontal && status === :left && l > 1
            a = config.main.hx * a
        else
            a = defect * a
        end
    end

    return a
end

function skeleton(h::Vector{Int}, v::Vector{Int}, a::TensorMap,
                  config::TileConfig, parity::Bool,
                  x_h_rem::Int, x_v_rem::Int, O::TensorMap)::ComplexF64

    if length(v) == length(h)
        for i in 1:(length(v) - 1)
            hs = _h_status(i, x_h_rem, x_v_rem)
            a = apply_tiles(a, config, h[i];
                            horizontal=true, use_defect=true,
                            status=hs, x_h_rem=x_h_rem)

            vs = _v_status(i, x_h_rem, h[1])
            a = apply_tiles(a, config, v[i];
                            horizontal=false, use_defect=true,
                            status=vs, x_h_rem=x_h_rem)
        end

        hs = _h_status(length(v), x_h_rem, x_v_rem)
        a = apply_tiles(a, config, h[end];
                        horizontal=true, use_defect=true,
                        status=hs, x_h_rem=x_h_rem)

        vs = _v_status(length(v), x_h_rem, h[1])
        a = apply_tiles(a, config, v[end];
                        horizontal=false, use_defect=!parity,
                        status=vs, x_h_rem=x_h_rem)
    else
        for i in 1:length(v)
            hs = _h_status(i, x_h_rem, x_v_rem)
            a = apply_tiles(a, config, h[i];
                            horizontal=true, use_defect=true,
                            status=hs, x_h_rem=x_h_rem)

            vs = _v_status(i, x_h_rem, h[1])
            a = apply_tiles(a, config, v[i];
                            horizontal=false, use_defect=true,
                            status=vs, x_h_rem=x_h_rem)
        end

        hs = _h_status(length(v) + 1, x_h_rem, x_v_rem)
        a = apply_tiles(a, config, h[end];
                        horizontal=true, use_defect=parity,
                        status=hs, x_h_rem=x_h_rem)
    end

    n_legs = div(numout(a), 2)
    b = operator_string(config.V, n_legs; op=O, op_first=false)
    return dot(b, a)
end

function two_point_correlation(config::TileConfig, x::Int, t::Int,
                               O₁::TensorMap, O₂::TensorMap,
                               k::Int=typemax(Int))::ComplexF64
    V = config.V

    if O₁.space != (V ← V) || O₂.space != (V ← V)
        throw(ArgumentError("Supplied operator(s) does not act on correct space!"))
    end

    d = div(numout(config.main.hd), 2)

    yₕ = cld(t + x, 2)  # ceiling division
    yᵥ = fld(t + 2 - x, 2)  # floor division
    # k = min(yₕ, yᵥ, k)
    parity = isodd(t + x) 

    if yᵥ % d != 0
        a = operator_string(V, yᵥ % d; op=O₁)
    else
        a = operator_string(V, d; op=O₁)
    end

    if yₕ == 0 && yᵥ == 0
        b = operator_string(V, div(numout(a), 2), op=O₂)
        return dot(b, a)
    end

    paths = generate_paths(
        yₕ % d != 0 ? fld(yₕ, d) + 1 : fld(yₕ, d),
        yᵥ % d != 0 ? fld(yᵥ, d) : fld(yᵥ, d) - 1,
        k
    )

    result = zero(ComplexF64)
    for (h, v) in paths
        result += skeleton(h, v, copy(a), config, parity, yₕ % d, yᵥ % d, O₂)
    end

    return result
end

function two_point_correlation(tile_dir::String, x::Int, t::Int,
                               O₁::TensorMap, O₂::TensorMap,
                               d::Int, k::Int=typemax(Int))::ComplexF64
    yₕ = cld(t + x, 2)  # ceiling division
    yᵥ = fld(t + 2 - x, 2)  # floor division
    config = build_tile_config(tile_dir, d, yₕ, yᵥ)
    return two_point_correlation(config, x, t, O₁, O₂, k)
end

function two_point_correlation(gate::FoldedGate, x::Int, t::Int,
                               O₁::TensorMap, O₂::TensorMap;
                               d::Int, k::Int=typemax(Int))::ComplexF64
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "tmp/run_$timestamp/"
    get_tiles(gate, d, dir)
    return two_point_correlation(dir, x, t, O₁, O₂, d, k)
end

function apply_transfer_matrix(gate::FoldedGate, a::TensorMap, l::Int, dir::Symbol; pivot::Bool=true)::TensorMap

    if dir == :h
        @tensor W[i, i'; k, k'] := gate.W[a, a, i, i'; k, k', b, b]
    elseif dir == :v
        @tensor W[i, i'; k, k'] := gate.W[i, i', a, a; b, b, k, k']
    end

    for _ in 1:l-1
        a = W * a
    end

    if pivot && dir == :h 
        @tensor W[i, i'; k, k'] = gate.W[i, i', a, a; k, k', b, b]
    elseif pivot && dir == :v
        @tensor W[i, i'; k, k'] = gate.W[a, a, i, i'; b, b, k, k']
    end

    return W * a
end

function skeleton(gate::FoldedGate, h::Vector{Int}, v::Vector{Int}, a::TensorMap, b::TensorMap, parity::Bool)::ComplexF64

    if length(v) == 0
        if length(h) == 1
            a = apply_transfer_matrix(gate, a, h[1], :h, pivot=parity)
        end
        return dot(b, a) 
    end

    for i in 1:length(v)-1
        a = apply_transfer_matrix(gate, a, h[i], :h)
        a = apply_transfer_matrix(gate, a, v[i], :v)
    end

    if length(h) == length(v) # terminate with vertical
        a = apply_transfer_matrix(gate, a, h[end], :h)
        a = apply_transfer_matrix(gate, a, v[end], :v, pivot=!parity)

    elseif length(h) == length(v) + 1 # terminate with horizontal
        a = apply_transfer_matrix(gate, a, h[end-1], :h)
        a = apply_transfer_matrix(gate, a, v[end], :v)
        a = apply_transfer_matrix(gate, a, h[end], :h, pivot=parity)

    else
        throw(ArgumentError("Unexpected path form"))
    end

    return dot(b, a)
end

function two_point_correlation(gate::FoldedGate, x::Int, t::Int, O₁::TensorMap, O₂::TensorMap, k::Int=typemax(Int))::Complex

    # check domain and codomain of O
    V = gate.V
    if O₁.space != (V ← V) || O₂.space != (V ← V)
        throw(ArgumentError("Supplied operator(s) does not act on correct space!"))
    end

    # vectorize operators
    a = permute(O₁, ((1, 2), ()))
    b = permute(O₂, ((1, 2), ()))

    # get circuit block dimensions and parity
    yₕ = cld(t + x, 2)  # ceiling division
    yᵥ = fld(t + 2 - x, 2)  # floor division
    parity = isodd(t + x) 

    # k = min(yₕ, yᵥ, k)

    result = zero(ComplexF64)
    paths = generate_paths(yₕ, yᵥ - 1, k) # truncate vertical height by convention
    for (h, v) in paths
        result += skeleton(gate, h, v, copy(a), copy(b), parity)
    end

    return result
end