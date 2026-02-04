
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

function fold(gate::Gate)::FoldedGate
    d = dim(gate.V)
    @tensor W[i, i', j, j'; k, k', l, l'] := gate.U[i, j; k, l] * conj(gate.U[i', j'; k', l'])
    FoldedGate(W / d, gate.V)
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

function peturb(gate::Gate, ϵ::Real=0.01)::Gate
    V = gate.V
    δ = randn(ComplexF64, V ⊗ V ← V ⊗ V)
    A = (δ - δ') / 2
    return Gate(exp(ϵ * A) * gate.U, V)
end