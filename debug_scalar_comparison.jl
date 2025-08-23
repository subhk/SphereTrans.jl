#!/usr/bin/env julia --startup-file=no --compile=no

# Compare scalar vs vector transform normalization
include("src/SHTnsKit.jl")
using .SHTnsKit

println("=== Scalar vs Vector Transform Normalization ===")

cfg = create_gauss_config(4, 4)
nlm = get_nlm(cfg)

# Test scalar transform first
println("\n--- Scalar Transform Test ---")
scalar_coeffs = zeros(Float64, nlm)
# Find l=1,m=0 for scalar
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        scalar_coeffs[idx] = 1.0
        println("Set scalar coeff[l=1,m=0] = 1.0 at index $idx")
        break
    end
end

# Scalar roundtrip
scalar_field = synthesize(cfg, scalar_coeffs)
scalar_back = analyze(cfg, scalar_field)

for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        println("Scalar roundtrip: $(scalar_back[idx]) (should be 1.0)")
        println("Scalar error: $(abs(scalar_back[idx] - 1.0))")
        break
    end
end

# Test vector transform 
println("\n--- Vector Transform Test ---")
sph_coeffs = zeros(Float64, nlm)
tor_coeffs = zeros(Float64, nlm)

# Find l=1,m=0 for vector
for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        sph_coeffs[idx] = 1.0
        println("Set vector sph_coeff[l=1,m=0] = 1.0 at index $idx")
        break
    end
end

# Vector roundtrip
uθ, uφ = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
sph_back, tor_back = analyze_vector(cfg, uθ, uφ)

for (idx, (l, m)) in enumerate(cfg.lm_indices)
    if l == 1 && m == 0
        println("Vector roundtrip: $(sph_back[idx]) (should be 1.0)")
        println("Vector error: $(abs(sph_back[idx] - 1.0))")
        println("Vector ratio to scalar: $(sph_back[idx] / scalar_back[idx])")
        break
    end
end

destroy_config(cfg)