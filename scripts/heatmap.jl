using Revise
using DualUnitary
using TensorKit
using Plots
using LaTeXStrings
using LinearAlgebra

# --- Setup ---
gate = soliton_dual_unitary_U1_qubit(0.5, 0.5)
P = random_perturbation(gate)
ϵ = 8
D = 3
T_max = 15

fg = peturb_folded(fold(gate), P, ϵ)

V = gate.V
d = dim(V)
Z = TensorMap(Dict(U1Irrep(0) => fill(1,1,1),
              U1Irrep(1) => fill(-1,1,1)), V ← V) / sqrt(d)

# Pre-compute tiles
tile_dir = "tmp/heatmap_tiles"
get_tiles(fg, D, tile_dir)

# --- Compute correlations ---
x_range = -T_max:T_max
t_range = 0:T_max

tile_data = zeros(Float64, length(t_range), length(x_range))
direct_data = zeros(Float64, length(t_range), length(x_range))

for (ti, t) in enumerate(t_range)
    println("Computing t=$t / $T_max")
    for (xi, x) in enumerate(x_range)
        if x < -t+1 || x > t
            continue  # outside causal light cone
        end
        tile_data[ti, xi] = real(two_point_correlation(tile_dir, x, t, Z, Z, D))
        direct_data[ti, xi] = real(two_point_correlation(fg, x, t, Z, Z))
    end
end

# --- Log₁₀ of absolute value (floor at -8 for zero entries) ---
log_floor = -8.0
tile_log = map(v -> v == 0.0 ? log_floor : log10(abs(v)), tile_data)
direct_log = map(v -> v == 0.0 ? log_floor : log10(abs(v)), direct_data)

# --- Plot heatmaps ---
xs = collect(x_range)
ts = collect(t_range)

common = (xlabel=L"x", ylabel=L"t", color=:inferno, clims=(log_floor, 0),
          colorbar_title=L"\log_{10}\langle \mathcal{U}^\dagger Z_0 \mathcal{U}_t, Z_x \rangle",
          aspect_ratio=:auto, size=(600, 500))

p1 = heatmap(xs, ts, tile_log; title="Tile sum (D=$D)", common...)
p2 = heatmap(xs, ts, direct_log; title="Direct (transfer matrix)", common...)

p = plot(p1, p2, layout=(1, 2), size=(1200, 500), margin=5Plots.mm)
savefig(p, "scripts/heatmap_correlations.png")
println("\nSaved to scripts/heatmap_correlations.png")
