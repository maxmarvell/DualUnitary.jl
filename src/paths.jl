function path_set(x::Int, k::Int=typemax(Int))
    res = Dict{Int, Vector{Vector{Int}}}()
    buf = Int[]

    function list_of_lists(remaining::Int)
        if remaining == 0
            push!(get!(res, length(buf), Vector{Int}[]), copy(buf))
            return
        end
        length(buf) >= k && return

        for i in 1:remaining
            push!(buf, i)
            list_of_lists(remaining - i)
            pop!(buf)
        end
    end

    list_of_lists(x)
    return res
end

function all_compositions(x::Int, k::Int=typemax(Int))::Vector{Vector{Int}}
    result = Vector{Int}[]
    buf = Int[]

    function generate(remaining::Int)
        if remaining == 0
            push!(result, copy(buf))
            return
        end
        length(buf) >= k && return
        for i in 1:remaining
            push!(buf, i)
            generate(remaining - i)
            pop!(buf)
        end
    end

    generate(x)
    return result
end

function valid_path_pairs(x_h::Int, x_v::Int, k::Int)::Vector{Tuple{Vector{Int}, Vector{Int}}}
    h_decomps = all_compositions(x_h, k)
    v_decomps = all_compositions(x_v, k)

    result = Tuple{Vector{Int}, Vector{Int}}[]

    for h in h_decomps
        for v in v_decomps
            len_h, len_v = length(h), length(v)
            if len_h == len_v || len_h == len_v + 1
                push!(result, (h, v))
            end
        end
    end

    return result
end

generate_paths(yₕ::Int, yᵥ::Int, k::Int=typemax(Int)) = valid_path_pairs(yₕ, yᵥ, k)