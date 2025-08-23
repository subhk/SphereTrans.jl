#!/usr/bin/env julia --startup-file=no --compile=no

# Debug exact precision to meet 0.1% test requirement
include("src/SHTnsKit.jl")
using .SHTnsKit
using Random

println("=== Exact Precision Analysis for 0.1% Test ===")

for (lmax, mmax) in ((4,4), (6,4))
    println("\nTesting lmax=$lmax, mmax=$mmax")
    cfg = create_gauss_config(lmax, mmax)
    nlm = get_nlm(cfg)
    
    # Use same random seed as test
    rng = MersenneTwister(7)
    sph = zeros(Float64, nlm)
    tor = zeros(Float64, nlm)
    for (idx, (l, m)) in enumerate(SHTnsKit.lm_from_index.(Ref(cfg), 1:nlm))
        if l >= 1
            sph[idx] = randn(rng)
            tor[idx] = randn(rng)
        end
    end
    
    uθ, uϕ = synthesize_vector(cfg, sph, tor)
    sph2, tor2 = analyze_vector(cfg, uθ, uϕ)

    # Ignore l=0 modes (should be zero anyway)
    mask = [l >= 1 for (l, m) in (SHTnsKit.lm_from_index(cfg, i) for i in 1:nlm)]

    num_s = maximum(abs.(sph2[mask] .- sph[mask]))
    den_s = maximum(abs.(sph[mask])) + eps()
    num_t = maximum(abs.(tor2[mask] .- tor[mask]))
    den_t = maximum(abs.(tor[mask])) + eps()
    
    println("Spheroidal relative error: $(num_s / den_s)")
    println("Toroidal relative error: $(num_t / den_t)")
    
    println("Spheroidal correction needed: $(1.0 / (num_s / den_s))")
    println("Toroidal correction needed: $(1.0 / (num_t / den_t))")
    
    destroy_config(cfg)
end