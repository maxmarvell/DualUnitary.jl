abstract type AbstractGate{S<:ElementarySpace} end

struct Gate{S<:ElementarySpace}
    U::TensorMap
    V::S

    function Gate(U::TensorMap, V::S) where {S<:ElementarySpace}
        W = V ⊗ V
        space(U) == (W ← W) ||
            throw(ArgumentError("TensorMap must act on V⊗V ← V⊗V"))
        new{S}(U, V)
    end
end

function dual(U::TensorMap)::TensorMap
    @tensor Ũ[i, k; j, l] := U[i, j; k, l]
end

function soliton_dual_unitary_U1_qubit(ϕ::Float64, J::Float64)::Gate

    V = U1Space(0 => 1, 1 => 1)
    
    u₊ = randisometry(V, V)
    u₋ = randisometry(V, V)

    v₊ = randisometry(V, V)
    v₋ = randisometry(V, V)

    blocks = Dict(
        U1Irrep(0) => fill(exp(-im * J), 1, 1),
        U1Irrep(1) => [0.0+0im        -im*exp(im*J);
                    -im*exp(im*J)  0.0+0im],
        U1Irrep(2) => fill(exp(-im * J), 1, 1)
    )
    U = TensorMap(blocks, V ⊗ V ← V ⊗ V)

    return Gate(exp(im * ϕ) * (u₊ ⊗ u₋) * U * (v₋ ⊗ v₊), V)
end

struct FoldedGate{S<:ElementarySpace}
    W::TensorMap
    V::S

    function FoldedGate(W::TensorMap, V::S) where {S<:ElementarySpace}
        if space(W) != (V ⊗ V' ⊗ V ⊗ V' ← V ⊗ V' ⊗ V ⊗ V')
            throw(ArgumentError(""))
        end
        new{S}(W, V)
    end
end


"""
    apply_caps(gate, which)

Cap (trace out) one interleaved (V, V′) pair of a folded tensor `W`.

# Arguments
- `W`: A folded TensorMap with interleaved legs `(V, V′, V, V′)`
- `which`: Int's specifying the gate pair to cap.\n
  1 - Left output\n
  2 - Right output\n
  3 - Left input\n
  4 - Right input


# Returns
A new TensorMap with the selected pair traced out.
"""
function apply_caps(gate::FoldedGate, which::Tuple{Vararg{Int}})
    W = gate.W
    d = dim(gate.V)
    s = Set(which)

    if s == Set([1, 2, 3, 4])
        @tensor T[] := W[a,a,b,b; c,c,d,d]
        return T / d^2
    elseif s == Set([1, 2, 3])
        @tensor T[l,l'] := W[a,a,b,b; c,c,l,l']
        return T / sqrt(d)^3
    elseif s == Set([1, 2, 4])
        @tensor T[k,k'] := W[a,a,b,b; k,k',d,d]
        return T / sqrt(d)^3
    elseif s == Set([1, 3, 4])
        @tensor T[j,j'] := W[a,a,j,j'; c,c,d,d]
        return T / sqrt(d)^3
    elseif s == Set([2, 3, 4])
        @tensor T[i,i'] := W[i,i',b,b; c,c,d,d]
        return T / sqrt(d)^3
    elseif s == Set([1, 2])
        @tensor T[k,k',l,l'] := W[a,a,b,b; k,k',l,l']
        return T / d
    elseif s == Set([1, 3])
        @tensor T[j,j';l,l'] := W[a,a,j,j'; c,c,l,l']
        return T / d
    elseif s == Set([1, 4])
        @tensor T[j,j';k,k'] := W[a,a,j,j'; k,k',d,d]
        return T / d
    elseif s == Set([2, 3])
        @tensor T[i,i';l,l'] := W[i,i',b,b; c,c,l,l']
        return T / d
    elseif s == Set([2, 4])
        @tensor T[i,i';k,k'] := W[i,i',b,b; k,k',d,d]
        return T / d
    elseif s == Set([3, 4])
        @tensor T[i,i',j,j'] := W[i,i',j,j'; c,c,d,d]
        return T / d
    elseif s == Set([1])
        @tensor T[j,j';k,k',l,l'] := W[a,a,j,j'; k,k',l,l']
        return T / sqrt(d)
    elseif s == Set([2])
        @tensor T[i,i';k,k',l,l'] := W[i,i',b,b; k,k',l,l']
        return T / sqrt(d)
    elseif s == Set([3])
        @tensor T[i,i',j,j';l,l'] := W[i,i',j,j'; c,c,l,l']
        return T / sqrt(d)
    elseif s == Set([4])
        @tensor T[i,i',j,j';k,k'] := W[i,i',j,j'; k,k',d,d]
        return T / sqrt(d)
    else  # No traces (empty set)
        return W
    end
end

function fold(gate::Gate)::FoldedGate
    @tensor W[i, i', j, j'; k, k', l, l'] := gate.U[i, j; k, l] * conj(gate.U[i', j'; k', l'])
    FoldedGate(W, gate.V)
end

function dual_fold(W::TensorMap)::TensorMap
    @tensor W̃[i, i', k, k'; j, j', l, l'] := W[i, i', j, j'; k, k', l, l']
end

function _is_unitary(U::TensorMap)::Bool
    U' * U ≈ id(domain(U)) && U * U' ≈ id(codomain(U))
end

function is_dual_unitary(U::TensorMap)::Bool
    @tensor Ũ[i, k; j, l] := U[i, j; k, l] # construct dual
    _is_unitary(Ũ)
end

function is_unital(W::TensorMap)::Bool
    @tensor I₁[i, j; i', j'] := W[i, i', j, j'; k, k, l, l]
    @tensor I₂[k', l'; k, l] := W[i, i, j, j; k, k', l, l']
    I₁ ≈ id(codomain(W, 1)) ⊗ id(codomain(W,3)) && I₂ ≈ id(domain(W, 1)) ⊗ id(domain(W, 3))
end

function is_dual_unital(W::TensorMap)::Bool
    W̃ = dual_fold(W)
    is_unital(W̃)
end

function has_soliton(W::TensorMap)::Bool

    V = domain(W, 1)
    blocks = Dict(
        U1Irrep(0) => fill(1, 1, 1),
        U1Irrep(1) => fill(-1, 1, 1)
    )
    Z = TensorMap(blocks, V ← V) / sqrt(2)
    Z = permute(Z, ((1, 2), ()))

    I = permute(id(V), ((1, 2), ()))

    I ⊗ Z ≈ W * (Z ⊗ I) && Z ⊗ I ≈ W * (I ⊗ Z)
end


function pertubation(gate::Gate; ϵ::Float64=0.01)::Gate
    δ = randn(ComplexF64, gate.U.space)
    A = (δ - δ') / 2
    return Gate(exp(ϵ * A), gate.V)
end

function to_complex_space(gate::Gate)::Gate
    d = dim(gate.V)
    V = ℂ^d
    M = convert(Array, gate.U)
    return Gate(TensorMap(M, V ⊗ V ← V ⊗ V), V)
end

function soliton_dual_unitary_complex(d::Int, ϕ::Float64, J::Float64)::Gate
    V = ℂ^d

    u₊ = randisometry(V, V)
    u₋ = randisometry(V, V)
    v₊ = randisometry(V, V)
    v₋ = randisometry(V, V)

    W = V ⊗ V
    M = zeros(ComplexF64, d^2, d^2)
    for i in 1:d, j in 1:d
        r = (i - 1) * d + j
        if i == j
            M[r, r] = exp(-im * J)
        else
            c = (j - 1) * d + i
            M[r, c] = -im * exp(im * J)
        end
    end
    U = TensorMap(M, W ← W)

    return Gate(exp(im * ϕ) * (u₊ ⊗ u₋) * U * (v₋ ⊗ v₊), V)
end

function random_perturbation(gate::Gate)::TensorMap
    V = gate.V
    δ = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    return (δ + δ') / 2
end

function peturb_folded(gate::FoldedGate, P::TensorMap, ϵ::Real=0.01)::FoldedGate
    V = gate.V
    G = exp(-im * ϵ * P)

    @tensor W_new[i,i',j,j'; k,k',l,l'] :=
        G[i,j; a,b] * conj(G[i',j'; a',b']) * gate.W[a,a',b,b'; k,k',l,l']

    return FoldedGate(W_new, V)
end

function peturb(gate::Gate, ϵ::Real=0.01; P::Union{TensorMap, Nothing}=nothing)::Gate
    V = gate.V
    if P === nothing
        δ = randn(ComplexF64, V ⊗ V ← V ⊗ V)
        P = (δ + δ') / 2
    end
    return Gate(gate.U * exp(im * ϵ * P), V)
end