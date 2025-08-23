#!/usr/bin/env julia --startup-file=no --compile=no

# Test orthonormal normalization specifically
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Testing Schmidt normalization...")

# Create config with explicit orthonormal normalization 
cfg = create_config(4, 4, 1; norm=SHTnsKit.SHT_SCHMIDT)
set_grid!(cfg, 20, 21)  # Set up grid manually

println("Config normalization: $(cfg.norm)")

# Create a simple spatial field (just Y_1^0)
nlat, nphi = get_nlat(cfg), get_nphi(cfg)
spatial = zeros(Float64, nlat, nphi)

# Set Y_1^0 = 1 in spatial domain
for i in 1:nlat
    theta = get_theta(cfg, i)
    for j in 1:nphi
        spatial[i, j] = sqrt(3/(4π)) * cos(theta)  # Y_1^0 in Schmidt normalization
    end
end

println("Created Y_1^0 field")

# Analyze 
coeffs = spat_to_sh(cfg, spatial)

println("Coefficients (first 10): $(coeffs[1:10])")

# Find the Y_1^0 coefficient
idx_10 = 0
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        global idx_10 = idx
        break
    end
end

if idx_10 > 0
    println("Y_1^0 coefficient: $(coeffs[idx_10]) (should be ~1.0)")
    
    # Synthesize back
    reconstructed = sh_to_spat(cfg, coeffs)
    
    # Check error
    error = maximum(abs.(spatial .- reconstructed))
    println("Roundtrip error: $error")
else
    println("Could not find Y_1^0 mode")
end