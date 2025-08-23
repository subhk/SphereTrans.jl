#!/usr/bin/env julia --startup-file=no --compile=no

# Debug single vector mode
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Debug single vector mode...")

# Create config
cfg = create_gauss_config(4, 4)

# Create single mode test - spheroidal l=1, m=0
sph_coeffs = zeros(Float64, get_nlm(cfg))
tor_coeffs = zeros(Float64, get_nlm(cfg))

# Find index for (l=1, m=0)
idx_10 = 0
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        global idx_10 = idx
        break
    end
end

if idx_10 > 0
    # Set spheroidal coefficient for (1,0) mode
    sph_coeffs[idx_10] = 1.0
    
    println("Original spheroidal coefficient for (l=1,m=0): ", sph_coeffs[idx_10])
    
    # Synthesize
    u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
    
    println("Synthesized field ranges:")
    println("  u_theta: [$(minimum(u_theta)), $(maximum(u_theta))]")
    println("  u_phi: [$(minimum(u_phi)), $(maximum(u_phi))]")
    
    # Analyze back
    sph_reconstructed, tor_reconstructed = analyze_vector(cfg, u_theta, u_phi)
    
    println("Reconstructed spheroidal coefficient for (l=1,m=0): ", sph_reconstructed[idx_10])
    println("Reconstructed toroidal coefficient for (l=1,m=0): ", tor_reconstructed[idx_10])
    
    println("Spheroidal ratio (reconstructed/original): ", sph_reconstructed[idx_10] / sph_coeffs[idx_10])
    
    # Check error
    sph_error = abs(sph_reconstructed[idx_10] - sph_coeffs[idx_10])
    tor_error = abs(tor_reconstructed[idx_10] - tor_coeffs[idx_10])
    
    println("Spheroidal error: $sph_error")
    println("Toroidal error: $tor_error")
else
    println("Could not find (l=1, m=0) mode")
end