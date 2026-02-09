
"""
    _hd_column(gate, dᵥ)

Construct column of gates yielding (rights) ← (lefts) ordered bottom to top.
Traces top on topmost gate and bottom on bottommost gate.

Gate legs after apply_caps:
- W₁ = apply_caps(gate, (1)): traces top, gives [right; left, bottom] = [j,j'; k,k',l,l']
- W₄ = apply_caps(gate, (4)): traces bottom, gives [top,right; left] = [i,i',j,j'; k,k']
- W = gate.W (no caps): [top,right; left,bottom] = [i,i',j,j'; k,k',l,l']
"""
function _hd_column(gate::FoldedGate, dᵥ::Int)::TensorMap
    if dᵥ == 1
        return apply_caps(gate, (1, 4))  # traces top+bottom, gives [right; left] = [j,j'; k,k']
    end

    W₁ = apply_caps(gate, (1,))  # top gate: [right; left, bottom]
    
    C = permute(W₁, ((1, 2, 3, 4), (5, 6)))  # [out₂ₙ,in₁ₙ;in₂ₙ]
    W̃ = permute(gate.W, ((1, 2), (3, 4, 5, 6, 7, 8))) # [out₁;out₂,in₁,in₂]
    
    for i in 2:(dᵥ-1)
        C = C * W̃  # [out₂ₙ,in₁ₙ,...,out₂₍ₙ₋ᵢ₊₂₎,in₁₍ₙ₋ᵢ₊₂₎,] ← [out₂₍ₙ₋ᵢ₊₁₎,in₁₍ₙ₋ᵢ₊₁₎,in₂]
        n = 4*i
        C = permute(C, (tuple(1:n...), (n+1, n+2))) # [out₂ₙ,in₁ₙ,...,out₂₍ₙ₋ᵢ₊₁₎,in₁₍ₙ₋ᵢ₊₁₎,] ← [in₂]
    end
    
    W₄ = apply_caps(gate, (4,))  # bottom gate: [top,right; left]
    W̃₄ = permute(W₄, ((1, 2), (3, 4, 5, 6))) # [out₁;out₂,in₁]
    C = C * W̃₄ # [out₂ₙ,in₁ₙ,...,out₂₍₂₎,in₁₍₂₎,] ← [out₂₍₁₎,in₁₍₁₎]

    # [out₂₍₁₎,...,out₂₍ₙ₎] ← [in₁₍₁₎...in₁₍ₙ₎]
    return permute(C, (
        tuple((x for i in dᵥ-1:-1:0 for x in (4i+1, 4i+2))...), 
        tuple((x for i in dᵥ-1:-1:0 for x in (4i+3, 4i+4))...)
    ))
end

"""
    _hx_column(gate::FoldedGate, dᵥ)

Construct column yielding (rights, top) ← (lefts) ordered bottom to top.
Traces bottom on bottommost gate only; top gate keeps top leg exposed.
"""
function _hx_column(gate::FoldedGate, dᵥ::Int)::TensorMap
    if dᵥ == 1
        W₄ = apply_caps(gate, (4,))  # traces bottom: [out₁,out₂; in₁]
        return permute(W₄, ((1, 2, 3, 4), (5, 6)))  # [out₁,out₂; in₁]
    end

    # Top gate (no traces): [out₁,out₂; in₁,in₂]
    C = permute(gate.W, ((1, 2, 3, 4, 5, 6), (7, 8)))  # [out₁ₙ,out₂ₙ,in₁ₙ; in₂ₙ]

    # Middle gates: [out₁; out₂,in₁,in₂]
    W̃ = permute(gate.W, ((1, 2), (3, 4, 5, 6, 7, 8)))

    for i in 2:(dᵥ-1)
        C = C * W̃  # [out₁ₙ,out₂ₙ,in₁ₙ,...,out₂₍ₙ₋ᵢ₊₂₎,in₁₍ₙ₋ᵢ₊₂₎] ← [out₂₍ₙ₋ᵢ₊₁₎,in₁₍ₙ₋ᵢ₊₁₎,in₂]
        n = 2 + 4*i
        C = permute(C, (tuple(1:n...), tuple(n+1:n+2...)))  # [out₁ₙ,out₂ₙ,in₁ₙ,...; in₂]
    end

    # Bottom gate: traces bottom [out₁; out₂,in₁]
    W₄ = apply_caps(gate, (4,))
    W̃₄ = permute(W₄, ((1, 2), (3, 4, 5, 6)))
    C = C * W̃₄  # [out₁ₙ,out₂ₙ,in₁ₙ,...,out₂₍₁₎,in₁₍₁₎]

    # [out₂₍₁₎,...,out₂₍ₙ₎,out₁ₙ] ← [in₁₍₁₎,...,in₁₍ₙ₎]
    rights = tuple((x for i in dᵥ-1:-1:0 for x in (4i+3, 4i+4))...)
    top = (1, 2)
    lefts = tuple((x for i in dᵥ-1:-1:0 for x in (4i+5, 4i+6))...)

    return permute(C, ((rights..., top...), lefts))
end

"""
    _hx_column_defect(gate::FoldedGate, dᵥ)

Construct column yielding (out₁) ← (in₁₍₁₎...in₁₍ₙ₎) ordered bottom to top.
Traces right on all gates, bottom+right on bottom gate.
"""
function _hx_column_defect(gate::FoldedGate, dᵥ::Int)::TensorMap
    if dᵥ == 1
        return permute(apply_caps(gate, (2, 4)), ((1, 2), (3, 4)))  # [out₁] ← [in₁]
    end

    W₁ = apply_caps(gate, (2,))
    C = permute(W₁, ((1, 2, 3, 4), (5, 6)))  # [out₁,in₁] ← [in₂]

    W̃ = permute(W₁, ((1, 2), (3, 4, 5, 6))) # [out₁] ← [in₁,in₂]

    for i in 2:(dᵥ-1)
        C = C * W̃  # [out₁,in₁ₙ,...,in₁₍ₙ₋ᵢ₊₂₎] ← [in₁₍ₙ₋ᵢ₊₁₎,in₂]
        n = 2*i + 2 
        C = permute(C, (tuple(1:n...), tuple(n+1:n+2...))) # [out₁,in₁ₙ,...,in₁₍ₙ₋ᵢ₊₁₎] ← [in₂]
    end

    W₄ = apply_caps(gate, (2, 4)) # [out₁] ← [in₁]
    C = C * W₄  # [out₁,in₁₍ₙ₎,...,in₁₍₂₎] ← [in₁₍₁₎]

    # [out₁ₙ] ← [in₁₍₁₎,...,in₁₍ₙ₎]
    return permute(C, (
        (1,2), 
        tuple((x for i in dᵥ-1:-1:0 for x in (2i+3, 2i+4))...)
    ))
end

"""
    _hx_row_defect(gate::FoldedGate, dₕ)

Construct row yielding (out₁₍₁₎...out₁₍ₙ₎) ← (in₁) ordered left to right.
Traces bottom on all gates, bottom+right on rightmost gate.
"""
function _hx_row_defect(gate::FoldedGate, dₕ::Int)::TensorMap
    if dₕ == 1
        return permute(apply_caps(gate, (2, 4)), ((1, 2), (3, 4)))  # [out₁] ← [in₁]
    end

    # build from left
    W₄ = apply_caps(gate, (4,)) # [out₁,out₂] ← [in₁]
    C = permute(W₄, ((3, 4), (1, 2, 5, 6))) # [out₂] ← [out₁,in₁]

    for i in 2:(dₕ-1)
        C = W₄ * C # [out₁₍ᵢ₎,out₂₍ᵢ₎] ← [out₁₍ᵢ₋₁₎,...,out₁₍₁₎,in₁]
        n = 2*i + 2
        C = permute(C, ((3, 4), tuple(1, 2, 5:n..., n+1, n+2))) # [out₂₍ᵢ₎] ← [out₁₍ᵢ₎,...,out₁₍₁₎,in₁]
    end

    W₂₄ = apply_caps(gate, (2, 4)) # [out₁] ← [in₁]
    C = W₂₄ * C  # [out₁₍ₙ₎] ← [out₁₍ₙ₋₁₎,...,out₁₍₁₎,in₁]

    # [out₁₍₁₎,...,out₁₍ₙ₎] ← [in₁]
    return permute(C, (
        tuple((x for i in dₕ-1:-1:0 for x in (2i+1, 2i+2))...),
        (2*dₕ+1,2*dₕ+2), 
    ))
end

"""
    _vd_row(gate::FoldedGate, dₕ)

Construct row yielding (tops) ← (bottoms) ordered left to right.
Traces left on leftmost gate, right on rightmost gate.
"""
function _vd_row(gate::FoldedGate, dₕ::Int)::TensorMap
    if dₕ == 1
        return apply_caps(gate, (2, 3))
    end

    W₃ = apply_caps(gate, (3,)) # [out₁,out₂] ← [in₂]
    C = permute(W₃, ((1, 2, 5, 6), (3, 4))) # [out₁,in₂] ← [out₂]

    W̃ = permute(gate.W, ((5, 6), (1, 2, 7, 8, 3, 4))) # [in₁] ← [out₁,in₂,out₂]
    for i in 2:(dₕ-1)
        C = C * W̃  # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ᵢ₋₁₎,in₂₍ᵢ₋₁₎] ← [out₁₍ᵢ₎,in₂₍ᵢ₎,out₂]
        n = 4*i
        C = permute(C, (tuple(1:n...), tuple(n+1:n+2...))) # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ᵢ₎,in₂₍ᵢ₎] ← [out₂]
    end

    # Right gate: traces right 
    W₂ = apply_caps(gate, (2,)) # [out₁] ← [in₁,in₂]
    W̃₂ = permute(W₂, ((3, 4), (1, 2, 5, 6)))  # [in₁] ← [out₁,in₂]
    C = C * W̃₂  # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ₙ₋₁₎,in₂₍ₙ₋₁₎] ← [out₁₍ₙ₎,in₂₍ₙ₎]

    # [out₁₍₁₎,...,out₁₍ₙ₎] ← [in₂₍₁₎,...,in₂₍ₙ₎]
    return permute(C, (
        tuple((x for i in 0:dₕ-1 for x in (4i+1, 4i+2))...),
        tuple((x for i in 0:dₕ-1 for x in (4i+3, 4i+4))...)
    ))
end

"""
    _vx_row(gate::FoldedGate, dₕ)

Construct row yielding (tops, right) ← (bottoms) ordered left to right.
Traces left on leftmost gate only; rightmost gate keeps right leg exposed.
"""
function _vx_row(gate::FoldedGate, dₕ::Int)::TensorMap
    if dₕ == 1
        return apply_caps(gate, (3,))  # [out₁,out₂] ← [in₂]
    end

    W₃ = apply_caps(gate, (3,)) # [out₁,out₂] ← [in₂]
    C = permute(W₃, ((1, 2, 5, 6), (3, 4))) # [out₁,in₂] ← [out₂]

    W̃ = permute(gate.W, ((5, 6), (1, 2, 7, 8, 3, 4))) # [in₁] ← [out₁,in₂,out₂]
    for i in 2:(dₕ-1)
        C = C * W̃  # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ᵢ₋₁₎,in₂₍ᵢ₋₁₎] ← [out₁₍ᵢ₎,in₂₍ᵢ₎,out₂]
        n = 4*i
        C = permute(C, (tuple(1:n...), tuple(n+1:n+2...))) # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ᵢ₎,in₂₍ᵢ₎] ← [out₂]
    end

    C = C * W̃  # [out₁₍₁₎,in₂₍₁₎,...,out₁₍ₙ₋₁₎,in₂₍ₙ₋₁₎] ← [out₁₍ₙ₎,in₂₍ₙ₎,out₂]

    # [out₁₍₁₎,...,out₁₍ₙ₎,out₂] ← [in₂₍₁₎,...,in₂₍ₙ₎]
    return permute(C, (
        tuple((x for i in 0:dₕ-1 for x in (4i+1, 4i+2))..., 4*dₕ+1, 4*dₕ+2),
        tuple((x for i in 0:dₕ-1 for x in (4i+3, 4i+4))...)
    ))
end

"""
    _vx_row_defect(gate::FoldedGate, dₕ)

Construct row yielding [out₂] ← [in₂₍₁₎...in₂₍ₙ₎] ordered left to right.
Traces top on all gates, top+left on leftmost gate.
"""
function _vx_row_defect(gate::FoldedGate, dₕ::Int)::TensorMap
    if dₕ == 1
        return apply_caps(gate, (1, 3))  # [out₂] ← [in₂]
    end

    W₁₃ = apply_caps(gate, (1, 3)) # [out₂] ← [in₂]
    C = permute(W₁₃, ((3, 4), (1, 2))) # [in₂] ← [out₂]

    W₁ = apply_caps(gate, (1,)) # [out₂] ← [in₁,in₂]
    W̃ = permute(W₁, ((3, 4), ( 5, 6, 1, 2)))  # [in₁] ← [in₂,out₂]

    for i in 2:(dₕ-1)
        C = C * W̃  # [in₂₍₁₎...in₂₍ᵢ₋₁₎] ← [in₂₍ᵢ₎,out₂]
        n = 2*i
        C = permute(C, (tuple(1:n...), tuple(n+1:n+2...))) # [in₂₍₁₎...in₂₍ᵢ₎] ← [out₂]
    end

    C = C * W̃  # [in₂₍₁₎...in₂₍ₙ₎] ← [out₂]

    # [out₂] ← [in₂₍₁₎...in₂₍ₙ₎]
    return permute(C, (
        (2*dₕ+1, 2*dₕ+2),
        tuple((x for i in 0:dₕ-1 for x in (2i+1, 2i+2))...),
    ))
end

"""
    _vx_column_defect(gate::FoldedGate, dᵥ)

Construct column yielding (rights) ← (bottom) ordered bottom to top.
Traces left on all gates, top+left on top gate.
"""
function _vx_column_defect(gate::FoldedGate, dᵥ::Int)::TensorMap
    if dᵥ == 1
        return apply_caps(gate, (1, 3)) # [out₂] ← [in₂]
    end

    W₁₃ = apply_caps(gate, (1, 3)) # [out₂] ← [in₂]
    C = permute(W₁₃, ((3, 4), (1, 2))) # [in₂] ← [out₂]

    W₃ = apply_caps(gate, (3,)) # [out₁,out₂] ← [in₂]
    W₃ = permute(W₃, ((5, 6, 3, 4), (1, 2))) # [in₂,out₂] ← [out₁]

    for i in 2:(dᵥ-1)
        C = W₃ * C # [in₂,out₂₍ₙ₋ᵢ₊₁₎] ← [out₂₍ₙ₋ᵢ₊₂₎,...,out₂₍ₙ₎]
        n = 2*i
        C = permute(C, ((1, 2), tuple(3:n+2...))) # [in₂] ← [out₂₍ₙ₋ᵢ₊₁₎,...,out₂₍ₙ₎]
    end
    C = W₃ * C  # [in₂,out₂₍₁₎] ← [out₂₍₂₎,...,out₂₍ₙ₎]

    # [out₂₍₁₎,...,out₂₍ₙ₎] ← [in₂] 
    return permute(C, (
        tuple(3:2*dᵥ+2...),
        (1, 2)
    ))
end

function hd_tile(gate::FoldedGate, dₕ::Int, dᵥ::Int)::TensorMap
    col = _hd_column(gate, dᵥ)
    tile = col
    for _ in 1:(dₕ - 1)
        tile = tile * col
    end
    return tile
end

function hx_tile(gate::FoldedGate, dₕ::Int, dᵥ::Int)::TensorMap
    if dₕ == 1
        return _hx_column_defect(gate, dᵥ)
    elseif dᵥ == 1
        return _hx_row_defect(gate, dₕ)
    end

    col = _hx_column(gate, dᵥ)
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

    col = _hx_column_defect(gate, dᵥ)
    tile = col * acc # (topₙ₊₁) ← (top₁,...,topₙ,lefts)
    return permute(tile, (tuple(3:2*dₕ..., 1, 2), tuple(2*dₕ+1:2*dₕ+2*dᵥ...)))
end

function vd_tile(gate::FoldedGate, dₕ::Int, dᵥ::Int)::TensorMap
    row = _vd_row(gate, dₕ)
    tile = row
    for _ in 1:(dᵥ - 1)
        tile = tile * row
    end
    return tile
end

function vx_tile(gate::FoldedGate, dₕ::Int, dᵥ::Int)::TensorMap

    if dᵥ == 1
        return _vx_row_defect(gate, dₕ)
    elseif dₕ == 1
        return _vx_column_defect(gate, dᵥ)
    end

    row = _vx_row(gate, dₕ) # (tops, right₁) ← (bottoms)
    acc = row

    for _ in 1:(dᵥ - 2)
        acc = permute(acc, (tuple(1:2*dₕ...), (tuple(
            numout(acc)+1:numin(acc)+numout(acc)-2*dₕ..., # right indices of accumulant
            2*dₕ+1:numout(acc)..., # new right indices
            numin(acc)+numout(acc)-2*dₕ+1:numin(acc)+numout(acc)... # bottom indices
        )))) # (tops) ← (right₁,...,rightₙ,bottoms)
        acc = row * acc # (tops, rightₙ₊₁) ← (right₁,...,rightₙ,bottoms)
    end

    # do final permutation before contracting with final row
    acc = permute(acc, (tuple(1:2*dₕ...), (tuple(
        numout(acc)+1:numin(acc)+numout(acc)-2*dₕ..., # right indices of accumulant
        2*dₕ+1:numout(acc)..., # new right indices
        numin(acc)+numout(acc)-2*dₕ+1:numin(acc)+numout(acc)... # bottom indices
    )))) # (tops) ← (right₁,...,rightₙ,bottoms)

    row = _vx_row_defect(gate, dₕ)
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

function get_tiles(gate::FoldedGate, d::Int, output_dir::String)
    mkpath(output_dir)
    for dᵥ in 1:d, dₕ in 1:d
        for (name, f) in [("hd", hd_tile), ("hx", hx_tile),
                          ("vd", vd_tile), ("vx", vx_tile)]
            path = joinpath(output_dir, "$(name)_$(dₕ)x$(dᵥ).jls")
            serialize(path, f(gate, dₕ, dᵥ))
            println("  saved $path")
        end
    end
    println("done — $(4*d^2) tiles written to $output_dir")
end
