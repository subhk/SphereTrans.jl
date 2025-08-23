#!/usr/bin/env julia --startup-file=no --compile=no

# Debug multiple vector modes
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Debug multiple vector modes...")

# Create config
cfg = create_gauss_config(4, 4)

# Test several individual modes
test_modes = [(1, 0), (2, 0), (2, 1), (3, 0), (4, 2)]

for (l_test, m_test) in test_modes
    println("\n--- Testing mode (l=$l_test, m=$m_test) ---")
    
    # Create single mode test
    sph_coeffs = zeros(Float64, get_nlm(cfg))
    tor_coeffs = zeros(Float64, get_nlm(cfg))
    
    # Find index for (l_test, m_test)
    idx_target = 0
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l == l_test && m == m_test
            idx_target = idx
            break
        end
    end
    
    if idx_target > 0
        # Test spheroidal coefficient
        sph_coeffs[idx_target] = 1.0
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        sph_error = abs(sph_reconstructed[idx_target] - 1.0)
        println("  Spheroidal: reconstructed = $(sph_reconstructed[idx_target]), error = $sph_error")
        
        # Reset and test toroidal coefficient
        sph_coeffs[idx_target] = 0.0
        tor_coeffs[idx_target] = 1.0
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
        
        tor_error = abs(tor_reconstructed[idx_target] - 1.0)
        println("  Toroidal: reconstructed = $(tor_reconstructed[idx_target]), error = $tor_error")
        
        # Reset
        tor_coeffs[idx_target] = 0.0
    else
        println("  Could not find mode (l=$l_test, m=$m_test)")
    end
end