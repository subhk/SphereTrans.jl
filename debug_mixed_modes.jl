#!/usr/bin/env julia --startup-file=no --compile=no

# Debug mixed mode behavior
include("src/SHTnsKit.jl")
using .SHTnsKit
using Random

println("=== Mixed Mode Vector Transform Analysis ===")

cfg = create_gauss_config(4, 4)
nlm = get_nlm(cfg)

# Test a small mixed case
println("\nTesting small mixed coefficient case:")

rng = MersenneTwister(42)
sph = zeros(Float64, nlm)
tor = zeros(Float64, nlm)

# Add just two modes to see interaction
# (1,0) and (2,0)
idx_10 = 0
idx_20 = 0
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        global idx_10 = idx
    elseif l == 2 && m == 0
        global idx_20 = idx
    end
end

if idx_10 > 0 && idx_20 > 0
    sph[idx_10] = 1.0
    sph[idx_20] = 0.5
    
    println("Input coefficients:")
    println("  (1,0) sph = $(sph[idx_10]), tor = $(tor[idx_10])")
    println("  (2,0) sph = $(sph[idx_20]), tor = $(tor[idx_20])")
    
    u_theta, u_phi = synthesize_vector(cfg, sph, tor)
    sph2, tor2 = analyze_vector(cfg, u_theta, u_phi)
    
    println("Reconstructed coefficients:")
    println("  (1,0) sph = $(sph2[idx_10]), tor = $(tor2[idx_10])")
    println("  (2,0) sph = $(sph2[idx_20]), tor = $(tor2[idx_20])")
    
    println("Errors:")
    println("  (1,0) sph error = $(abs(sph2[idx_10] - sph[idx_10]))")
    println("  (2,0) sph error = $(abs(sph2[idx_20] - sph[idx_20]))")
    
    # Total relative error
    sph_error = maximum(abs.(sph2[1:nlm] .- sph[1:nlm]))
    sph_norm = maximum(abs.(sph[1:nlm])) + eps()
    
    println("Overall relative error: $(sph_error / sph_norm)")
end

destroy_config(cfg)