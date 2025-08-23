#!/usr/bin/env julia --startup-file=no --compile=no

# Test 3D vector transform approach
include("src/SHTnsKit.jl")
using .SHTnsKit
using Random

println("=== 3D Vector Transform Approach ===")

cfg = create_gauss_config(4, 4)
nlm = get_nlm(cfg)

# Test same random coefficients as the failing test
rng = MersenneTwister(7)
sph = zeros(Float64, nlm)
tor = zeros(Float64, nlm)
for (idx, (l, m)) in enumerate(SHTnsKit.lm_from_index.(Ref(cfg), 1:nlm))
    if l >= 1
        sph[idx] = randn(rng)
        tor[idx] = randn(rng)
    end
end

println("Testing 3D vector transforms (no radial component)...")

# Create 3D coefficient arrays: Q=0 (no radial), S=sph, T=tor
q_coeffs = zeros(Float64, nlm)  # No radial component
s_coeffs = copy(sph)            # Spheroidal = original sph
t_coeffs = copy(tor)            # Toroidal = original tor

# 3D synthesis: Q,S,T -> Vr,Vθ,Vφ
vr, vθ, vφ = synthesize_3d_vector(cfg, q_coeffs, s_coeffs, t_coeffs)

println("Synthesized 3D field:")
println("  Max |Vr|: $(maximum(abs.(vr))) (should be ~0 since Q=0)")
println("  Max |Vθ|: $(maximum(abs.(vθ)))")
println("  Max |Vφ|: $(maximum(abs.(vφ)))")

# 3D analysis: Vr,Vθ,Vφ -> Q,S,T
q2, s2, t2 = analyze_3d_vector(cfg, vr, vθ, vφ)

# Check roundtrip accuracy (ignore l=0)
mask = [l >= 1 for (l, m) in (SHTnsKit.lm_from_index(cfg, i) for i in 1:nlm)]

# Radial component should be zero
q_error = maximum(abs.(q2[mask]))
println("\nRadial component error: $q_error (should be ~0)")

# Tangential components should match
num_s = maximum(abs.(s2[mask] .- s_coeffs[mask]))
den_s = maximum(abs.(s_coeffs[mask])) + eps()
num_t = maximum(abs.(t2[mask] .- t_coeffs[mask]))
den_t = maximum(abs.(t_coeffs[mask])) + eps()

println("3D Spheroidal relative error: $(num_s / den_s)")
println("3D Toroidal relative error: $(num_t / den_t)")
println("3D Test passes: $(num_s / den_s < 1e-3 && num_t / den_t < 1e-3)")

# Compare with 2D approach
println("\n--- Comparison with 2D approach ---")
uθ_2d, uφ_2d = synthesize_vector(cfg, sph, tor)
sph2_2d, tor2_2d = analyze_vector(cfg, uθ_2d, uφ_2d)

num_s_2d = maximum(abs.(sph2_2d[mask] .- sph[mask]))
den_s_2d = maximum(abs.(sph[mask])) + eps()
num_t_2d = maximum(abs.(tor2_2d[mask] .- tor[mask]))
den_t_2d = maximum(abs.(tor[mask])) + eps()

println("2D Spheroidal relative error: $(num_s_2d / den_s_2d)")
println("2D Toroidal relative error: $(num_t_2d / den_t_2d)")

println("\nImprovement ratio:")
println("  Spheroidal: $(num_s_2d / den_s_2d) / $(num_s / den_s) = $((num_s_2d / den_s_2d) / (num_s / den_s))")
println("  Toroidal: $(num_t_2d / den_t_2d) / $(num_t / den_t) = $((num_t_2d / den_t_2d) / (num_t / den_t))")

destroy_config(cfg)