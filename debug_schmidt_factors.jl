#!/usr/bin/env julia --startup-file=no --compile=no

# Debug Schmidt normalization factors vs C code
include("src/SHTnsKit.jl")
using .SHTnsKit

println("=== Schmidt Normalization Factor Analysis ===")

cfg = create_gauss_config(4, 4)
println("Config normalization: $(cfg.norm)")

# Check what our plm_cache contains for small cases
println("\nChecking P_1^0 and P_1^1 values:")
for i in 1:min(3, cfg.nlat)
    theta = cfg.theta_grid[i]
    println("θ=$theta:")
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l == 1 && m <= 1
            plm_val = cfg.plm_cache[i, idx]
            println("  P_$(l)^$(m) = $plm_val")
        end
    end
end

# Test single coefficient roundtrip for l=1,m=0
println("\n=== Single Coefficient Test: l=1,m=0 ===")
sph_in = zeros(Float64, cfg.nlm)
tor_in = zeros(Float64, cfg.nlm)

# Find index for l=1,m=0
l1m0_idx = 0
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        global l1m0_idx = idx
        break
    end
end

if l1m0_idx > 0
    sph_in[l1m0_idx] = 1.0
else
    println("Could not find l=1,m=0 mode")
    destroy_config(cfg)
    exit(1)
end
println("Input sph_coeff[l=1,m=0] = $(sph_in[l1m0_idx])")

uθ, uφ = synthesize_vector(cfg, sph_in, tor_in)
println("Synthesized field max: $(maximum(abs.(uθ)))")

sph_out, tor_out = analyze_vector(cfg, uθ, uφ)
println("Output sph_coeff[l=1,m=0] = $(sph_out[l1m0_idx])")
println("Roundtrip ratio: $(sph_out[l1m0_idx] / sph_in[l1m0_idx])")

destroy_config(cfg)