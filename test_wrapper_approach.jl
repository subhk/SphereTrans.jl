#!/usr/bin/env julia --startup-file=no --compile=no

# Test wrapper approach from vector_transforms_final.jl
include("src/SHTnsKit.jl")
using .SHTnsKit
using Random

# Include the wrapper functions
include("src/transforms/vector_transforms_final.jl")

println("=== Testing Wrapper Approach ===")

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

println("Testing wrapper vector transforms...")
uθ, uϕ = sphtor_to_spat!(cfg, copy(sph), copy(tor), 
                         Matrix{Float64}(undef, cfg.nlat, cfg.nphi), 
                         Matrix{Float64}(undef, cfg.nlat, cfg.nphi))

sph2, tor2 = spat_to_sphtor!(cfg, uθ, uϕ, 
                            Vector{Float64}(undef, nlm), 
                            Vector{Float64}(undef, nlm))

# Ignore l=0 modes
mask = [l >= 1 for (l, m) in (SHTnsKit.lm_from_index(cfg, i) for i in 1:nlm)]

num_s = maximum(abs.(sph2[mask] .- sph[mask]))
den_s = maximum(abs.(sph[mask])) + eps()
num_t = maximum(abs.(tor2[mask] .- tor[mask]))
den_t = maximum(abs.(tor[mask])) + eps()

println("Wrapper spheroidal relative error: $(num_s / den_s)")
println("Wrapper toroidal relative error: $(num_t / den_t)")
println("Test passes: $(num_s / den_s < 1e-3 && num_t / den_t < 1e-3)")

destroy_config(cfg)