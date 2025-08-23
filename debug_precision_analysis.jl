#!/usr/bin/env julia --startup-file=no --compile=no

# Detailed precision analysis of vector transforms
include("src/SHTnsKit.jl")
using .SHTnsKit

println("=== Detailed Vector Transform Precision Analysis ===")

cfg = create_gauss_config(4, 4)

# Test each individual mode to identify patterns
println("\nTesting individual modes:")
println("l\tm\tSph_error\tTor_error\tSph_ratio\tTor_ratio")

for (l, m) in [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0), (4, 2)]
    # Find index for this mode
    idx_target = 0
    for (idx, (ll, mm)) in enumerate(cfg.lm_indices)
        if ll == l && mm == m
            idx_target = idx
            break
        end
    end
    
    if idx_target > 0
        # Test spheroidal
        sph_coeffs = zeros(Float64, get_nlm(cfg))
        tor_coeffs = zeros(Float64, get_nlm(cfg))
        sph_coeffs[idx_target] = 1.0
        
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        sph_error = abs(sph_reconstructed[idx_target] - 1.0)
        sph_ratio = sph_reconstructed[idx_target]
        
        # Test toroidal  
        sph_coeffs[idx_target] = 0.0
        tor_coeffs[idx_target] = 1.0
        
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        tor_error = abs(tor_reconstructed[idx_target] - 1.0)
        tor_ratio = tor_reconstructed[idx_target]
        
        println("$l\t$m\t$(round(sph_error, digits=6))\t$(round(tor_error, digits=6))\t$(round(sph_ratio, digits=6))\t$(round(tor_ratio, digits=6))")
    end
end

# Analyze if there's a pattern with l or empirical correction factor
println("\nEmpirical correction factors needed:")
println("l\tm\tSph_factor\tTor_factor")

for (l, m) in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0)]
    idx_target = 0
    for (idx, (ll, mm)) in enumerate(cfg.lm_indices)
        if ll == l && mm == m
            idx_target = idx
            break
        end
    end
    
    if idx_target > 0
        # Test spheroidal
        sph_coeffs = zeros(Float64, get_nlm(cfg))
        tor_coeffs = zeros(Float64, get_nlm(cfg))
        sph_coeffs[idx_target] = 1.0
        
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        sph_factor = 1.0 / sph_reconstructed[idx_target]
        
        # Test toroidal
        sph_coeffs[idx_target] = 0.0 
        tor_coeffs[idx_target] = 1.0
        
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        tor_factor = 1.0 / tor_reconstructed[idx_target]
        
        println("$l\t$m\t$(round(sph_factor, digits=6))\t$(round(tor_factor, digits=6))")
    end
end

destroy_config(cfg)