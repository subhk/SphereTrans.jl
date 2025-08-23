#!/usr/bin/env julia --startup-file=no --compile=no

# Debug derivative computation for different l values  
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Testing derivative computation for different l values...")

cfg = create_gauss_config(4, 4)
θ = π/4  # 45 degrees
lat_idx = 3  # some middle latitude point

println("θ = $θ ($(θ*180/π) degrees)")
println()

for (l, m) in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 2)]
    # Find coefficient index
    coeff_idx = 0
    for (idx, (ll, mm)) in enumerate(cfg.lm_indices)
        if ll == l && mm == m
            coeff_idx = idx
            break
        end
    end
    
    if coeff_idx > 0
        derivative = SHTnsKit._compute_plm_theta_derivative(cfg, l, m, θ, coeff_idx, lat_idx)
        plm_val = cfg.plm_cache[lat_idx, coeff_idx]
        
        println("(l=$l, m=$m): P_l^m = $plm_val, dP_l^m/dθ = $derivative")
    end
end