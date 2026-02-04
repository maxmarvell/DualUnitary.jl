# Traced gate variants.
# W has indices [i,i',j,j'; k,k',l,l'] with
#   top=(i,i')  right=(j,j')  left=(k,k')  bottom=(l,l')
# Tracing a bond = contracting ket+bra via repeated index label.

_trace_top(W::TensorMap) =
    @tensor T[j, j'; k, k', l, l'] := W[a, a, j, j'; k, k', l, l']

_trace_right(W::TensorMap) =
    @tensor T[i, i'; k, k', l, l'] := W[i, i', a, a; k, k', l, l']

_trace_bottom(W::TensorMap) =
    @tensor T[i, i', j, j'; l, l'] := W[i, i', j, j'; l, l', a, a]

_trace_left(W::TensorMap) =
    @tensor T[i, i', j, j'; k, k'] := W[i, i', j, j'; a, a, k, k']

_trace_top_left(W::TensorMap) =
    @tensor T[j, j'; k, k'] := W[a, a, j, j'; b, b, k, k']

_trace_top_bottom(W::TensorMap) =
    @tensor T[j, j'; l, l'] := W[a, a, j, j'; l, l', b, b]

_trace_right_left(W::TensorMap) =
    @tensor T[i, i'; k, k'] := W[i, i', a, a; b, b, k, k']

_trace_right_bottom(W::TensorMap) =
    @tensor T[i, i'; l, l'] := W[i, i', a, a; l, l', b, b]

# Shorthand: permute with codomain/domain split
_perm(t, cod, dom) = permute(t, (cod, dom))

# yields column like (rights) ← (lefts) ordered bottom to top
function _hd_column(W::TensorMap, dᵥ::Int)

    if dᵥ == 1
        return _trace_top_bottom(W)  # traces top+bottom, keeps right+left
    end

    tensors = ntuple(_ -> W, dᵥ)
    indices = (
        [2, 3, -1, -2, -(2*dᵥ+1), -(2*dᵥ+2), 1, 1],
        ntuple(i -> [2*i+2, 2*i+3, -(2*i+1), -(2*i+2), -(2*dᵥ+2*i+1), -(2*dᵥ+2*i+2), 2*i, 2*i+1,], dᵥ - 2)...,
        [2*dᵥ, 2*dᵥ, -(2*dᵥ-1), -(2*dᵥ), -(4*dᵥ-1), -(4*dᵥ), 2*dᵥ-2, 2*dᵥ-1]
    )
    col = ncon(tensors, indices)
    return permute(col, (ntuple(identity, 2*dᵥ), ntuple(i -> i + 2*dᵥ, 2*dᵥ)))
end

# yields column like (rights, top) ← (lefts) ordered bottom to top
function _hx_column(W::TensorMap, dᵥ::Int)
    if dᵥ == 1
        Wᵦ = _trace_bottom(W)
        return permute(Wᵦ, ((1, 2, 3, 4), (5, 6)))  # (top,right) ← (left)
    end

    tensors = ntuple(_ -> W, dᵥ)
    indices = (
        [2, 3, -1, -2, -(2*dᵥ+1), -(2*dᵥ+2), 1, 1],
        ntuple(i -> [2*i+2, 2*i+3, -(2*i+1), -(2*i+2), -(2*dᵥ+2*i+1), -(2*dᵥ+2*i+2), 2*i, 2*i+1,], dᵥ - 2)...,
        [-(4*dᵥ+1), -(4*dᵥ+2), -(2*dᵥ-1), -(2*dᵥ), -(4*dᵥ-1), -(4*dᵥ), 2*dᵥ-2, 2*dᵥ-1]
    )
    col = ncon(tensors, indices)
    return permute(col, ((ntuple(identity, 2*dᵥ)..., 4*dᵥ+1, 4*dᵥ+2), ntuple(i -> i + 2*dᵥ, 2*dᵥ)))
end

# yields column like (top) ← (lefts) ordered bottom to top — traces top+right on rightmost column
function _hx_column_defect(W::TensorMap, dᵥ::Int)
    if dᵥ == 1
        return _perm(_trace_right_bottom(W), (1, 2), (3, 4))  # (top) ← (left)
    end

    tensors = ntuple(_ -> W, dᵥ)
    indices = (
        [dᵥ+2, dᵥ+3, 2, 2, -3, -4, 1, 1],
        ntuple(i -> [dᵥ+2*i+2, dᵥ+2*i+3, i+2, i+2, -(2*i+3), -(2*i+4), dᵥ+2*i, dᵥ+2*i+1], dᵥ - 2)...,
        [-1, -2, dᵥ+1, dᵥ+1, -(2*dᵥ+1), -(2*dᵥ+2), 3*dᵥ-2, 3*dᵥ-1]
    )
    col = ncon(tensors, indices)
    return permute(col, ((1, 2), tuple(3:2*dᵥ...)))
end

# yields row like (tops) ← (left) ordered left to right — traces bottom+right
# W: [top(1,2), right(3,4); left(5,6), bottom(7,8)]
function _hx_row_defect(W::TensorMap, dₕ::Int)
    if dₕ == 1
        return _perm(_trace_right_bottom(W), (1, 2), (3, 4))  # (top) ← (left)
    end

    tensors = ntuple(_ -> W, dₕ)
    indices = (
        [-3, -4, dₕ+2, dₕ+3, -1, -2, 2, 2],
        ntuple(i -> [-(2*i+3), -(2*i+4), dₕ+2*i+2, dₕ+2*i+3, dₕ+2*i, dₕ+2*i+1, i+2, i+2], dₕ - 2)...,
        [-(2*dₕ+1), -(2*dₕ+2), 1, 1, 3*dₕ-2, 3*dₕ-1, dₕ+1, dₕ+1]
    )
    row = ncon(tensors, indices)
    return permute(row, (tuple(3:2*dₕ+2...), (1, 2)))
end

# yields row like (tops) ← (bottoms) ordered left to right
function _vd_row(W::TensorMap, dₕ::Int)
    if dₕ == 1
        return _trace_right_left(W)  # traces right+left, keeps top+bottom
    end

    tensors = ntuple(_ -> W, dₕ)
    indices = (
        [-1, -2, 2, 3, 1, 1, -(2*dₕ+1), -(2*dₕ+2)],
        ntuple(i -> [-(2*i+1), -(2*i+2), 2*i+2, 2*i+3, 2*i, 2*i+1, -(2*dₕ+2*i+1), -(2*dₕ+2*i+2)], dₕ-2)...,
        [-(2*dₕ-1), -(2*dₕ), 2*dₕ, 2*dₕ, 2*dₕ-2, 2*dₕ-1, -(4*dₕ-1), -(4*dₕ)]
    )
    row = ncon(tensors, indices)
    return permute(row, (ntuple(identity, 2*dₕ), ntuple(i -> i + 2*dₕ, 2*dₕ)))
end

# yields row like (tops, right) ← (bottoms) ordered left to right
function _vx_row(W::TensorMap, dₕ::Int)
    if dₕ == 1
        Wₗ = _trace_left(W)
        return permute(Wₗ, ((1, 2, 3, 4), (5, 6)))  # (top,right) ← (bottom)
    end

    tensors = ntuple(_ -> W, dₕ)
    indices = (
        [-1, -2, 2, 3, 1, 1, -(2*dₕ+1), -(2*dₕ+2)],
        ntuple(i -> [-(2*i+1), -(2*i+2), 2*i+2, 2*i+3, 2*i, 2*i+1, -(2*dₕ+2*i+1), -(2*dₕ+2*i+2)], dₕ-2)...,
        [-(2*dₕ-1), -(2*dₕ), -(4*dₕ+1), -(4*dₕ+2), 2*dₕ-2, 2*dₕ-1, -(4*dₕ-1), -(4*dₕ)]
    )
    row = ncon(tensors, indices)
    return permute(row, ((ntuple(identity, 2*dₕ)..., 4*dₕ+1, 4*dₕ+2), ntuple(i -> i + 2*dₕ, 2*dₕ)))
end

# yields column like (rights) ← (bottom) ordered bottom to top — traces top+left
# W: [top(1,2), right(3,4); left(5,6), bottom(7,8)]
function _vx_column_defect(W::TensorMap, dᵥ::Int)
    if dᵥ == 1
        return _perm(_trace_top_left(W), (1, 2), (3, 4))  # (right) ← (bottom)
    end

    tensors = ntuple(_ -> W, dᵥ)
    # Gate 1 (bottom): top→connect, right→external, left→trace, bottom→external (input)
    # Middle gates: top→connect, right→external, left→trace, bottom→connect
    # Gate dᵥ (top): top→trace, right→external, left→trace, bottom→connect
    indices = (
        [dᵥ+2, dᵥ+3, -1, -2, 1, 1, -(2*dᵥ+1), -(2*dᵥ+2)],
        ntuple(i -> [dᵥ+2*i+2, dᵥ+2*i+3, -(2*i+1), -(2*i+2), i+1, i+1, dᵥ+2*i, dᵥ+2*i+1], dᵥ - 2)...,
        [dᵥ, dᵥ, -(2*dᵥ-1), -(2*dᵥ), dᵥ+1, dᵥ+1, 2*dᵥ-2, 2*dᵥ-1]
    )
    col = ncon(tensors, indices)
    return permute(col, (ntuple(identity, 2*dᵥ), (2*dᵥ+1, 2*dᵥ+2)))
end

# yields row like (right) ← (bottoms) ordered left to right
function _vx_row_defect(W::TensorMap, dₕ::Int)
    tensors = ntuple(_ -> W, dₕ)
    indices = (
        [2, 2, dₕ+2, dₕ+3, 1, 1, -3, -4],
        ntuple(i -> [i+2, i+2, dₕ+2*i+2, dₕ+2*i+3, dₕ+2*i, dₕ+2*i+1, -(2*i+3), -(2*i+4)], dₕ - 2)...,
        [dₕ+1, dₕ+1, -1, -2, 3*dₕ-2, 3*dₕ-1, -(2*dₕ+1), -(2*dₕ+2)]
    )
    row = ncon(tensors, indices)
    return permute(row, ((1, 2), tuple(3:2*dₕ...)))
end

function hd_tile(W::TensorMap, d_h::Int, d_v::Int)::TensorMap
    if d_v == 1
        T = _trace_top_bottom(W)  # traces top+bottom, keeps right+left
        tile = _perm(T, (1, 2), (3, 4))  # (right) ← (left)
        for _ in 1:(d_h - 1)
            tile = tile * _perm(T, (1, 2), (3, 4))
        end
        return tile
    end
    col = _hd_column(W, d_v)
    tile = col
    for _ in 1:(d_h - 1)
        tile = tile * col
    end
    return tile
end

function hx_tile(W::TensorMap, dₕ::Int, dᵥ::Int)::TensorMap

    if dₕ == 1 && dᵥ == 1
        return _perm(_trace_right_bottom(W), (1, 2), (3, 4))  # (top) ← (left)
    end

    if dₕ == 1
        return _hx_column_defect(W, dᵥ)  # (tops) ← (lefts)
    end

    if dᵥ == 1
        return _hx_row_defect(W, dₕ)  # (tops) ← (left)
    end

    col = _hx_column(W, dᵥ)
    acc = col

    for _ in 1:(dₕ - 2)
        acc = permute(acc, (tuple(1:2*dᵥ...), (tuple(
            numout(acc)+1:numin(acc)+numout(acc)-2*dᵥ..., # top indices of accumulant
            2*dᵥ+1:numout(acc)..., # new top indices
            numin(acc)+numout(acc)-2*dᵥ+1:numin(acc)+numout(acc)... # left indices
        ))))
        acc = col * acc # (rights, topₙ₊₁) ← (top₁,...,topₙ,lefts)
    end

    # do final permutation before contracting with final column
    acc = permute(acc, (tuple(1:2*dᵥ...), (tuple(
        numout(acc)+1:numin(acc)+numout(acc)-2*dᵥ..., # top indices of accumulant
        2*dᵥ+1:numout(acc)..., # new top indices
        numin(acc)+numout(acc)-2*dᵥ+1:numin(acc)+numout(acc)... # left indices
    ))))

    col = _hx_column_defect(W, dᵥ)
    tile = col * acc # (topₙ₊₁) ← (top₁,...,topₙ,lefts)
    return permute(tile, (tuple(3:2*dₕ..., 1, 2), tuple(2*dₕ+1:2*dₕ+2*dᵥ...)))
end

function vd_tile(W::TensorMap, d_h::Int, d_v::Int)::TensorMap
    if d_h == 1
        T = _trace_right_left(W)  # traces right+left, keeps top+bottom
        tile = _perm(T, (1, 2), (3, 4))  # (top) ← (bottom)
        for _ in 1:(d_v - 1)
            tile = tile * _perm(T, (1, 2), (3, 4))
        end
        return tile
    end
    row = _vd_row(W, d_h)
    tile = row
    for _ in 1:(d_v - 1)
        tile = tile * row
    end
    return tile
end

function vx_tile(W::TensorMap, dᵥ::Int, dₕ::Int)::TensorMap

    if dₕ == 1 && dᵥ == 1
        return _perm(_trace_top_left(W), (1, 2), (3, 4))  # (right) ← (bottom)
    end

    if dᵥ == 1
        return _vx_row_defect(W, dₕ)  # row mapping (rights) ← (bottoms)
    end

    if dₕ == 1
        return _vx_column_defect(W, dᵥ)  # (rights) ← (bottom)
    end

    row = _vx_row(W, dₕ)
    acc = row

    for _ in 1:(dᵥ - 2)
        acc = permute(acc, (tuple(1:2*dₕ...), (tuple(
            numout(acc)+1:numin(acc)+numout(acc)-2*dₕ..., # right indices of accumulant
            2*dₕ+1:numout(acc)..., # new right indices
            numin(acc)+numout(acc)-2*dₕ+1:numin(acc)+numout(acc)... # bottom indices
        ))))
        acc = row * acc # (tops, rightₙ₊₁) ← (right₁,...,rightₙ,bottoms)
    end

    # do final permutation before contracting with final row
    acc = permute(acc, (tuple(1:2*dₕ...), (tuple(
        numout(acc)+1:numin(acc)+numout(acc)-2*dₕ..., # right indices of accumulant
        2*dₕ+1:numout(acc)..., # new right indices
        numin(acc)+numout(acc)-2*dₕ+1:numin(acc)+numout(acc)... # bottom indices
    ))))

    row = _vx_row_defect(W, dₕ)
    tile = row * acc # (rightₙ₊₁) ← (right₁,...,rightₙ,bottoms)
    return permute(tile, (tuple(3:2*dᵥ..., 1, 2), tuple(2*dᵥ+1:2*dᵥ+2*dₕ...)))
end

struct TileSet
    hd::TensorMap
    hx::TensorMap
    vd::TensorMap
    vx::TensorMap
end

function load_tiles(dir::String, d_h::Int, d_v::Int)
    load(name) = deserialize(joinpath(dir, "$(name)_$(d_h)x$(d_v).jls"))
    (hd=load("hd"), hx=load("hx"),
     vd=load("vd"), vx=load("vx"))
end

get_tiles(gate::Gate, d::Int, output_dir::String) = get_tiles(fold(gate), d, output_dir)

function get_tiles(fg::FoldedGate, d::Int, output_dir::String)
    W = fg.W
    mkpath(output_dir)
    for d_v in 1:d, d_h in 1:d
        for (name, f) in [("hd", hd_tile), ("hx", hx_tile),
                          ("vd", vd_tile), ("vx", vx_tile)]
            path = joinpath(output_dir, "$(name)_$(d_h)x$(d_v).jls")
            serialize(path, f(W, d_h, d_v))
            println("  saved $path")
        end
    end
    println("done — $(4*d^2) tiles written to $output_dir")
end
