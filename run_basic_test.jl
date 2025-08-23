#!/usr/bin/env julia --startup-file=no --compile=no

using Test

# Load SHTnsKit directly 
println("Loading SHTnsKit...")
include("src/SHTnsKit.jl")
using .SHTnsKit
println("SHTnsKit loaded successfully!")

@testset "Basic functionality tests" begin
    println("Creating config...")
    cfg = create_gauss_config(4, 4)
    @test get_lmax(cfg) == 4
    @test get_mmax(cfg) == 4
    println("✓ Configuration created")
    
    println("Testing complex transforms...")
    # Create some test complex data  
    spat = rand(Complex{Float64}, get_nlat(cfg), get_nphi(cfg))
    
    # Transform: spatial -> spectral -> spatial
    sh = SHTnsKit.cplx_spat_to_sh(cfg, spat)
    reconstructed = SHTnsKit.cplx_sh_to_spat(cfg, sh)
    
    # Check roundtrip error
    max_error = maximum(abs.(spat .- reconstructed))
    @test max_error < 1e-12
    println("✓ Complex roundtrip error: $max_error")
    
    # Test real transforms
    println("Testing real transforms...")
    real_spat = rand(Float64, get_nlat(cfg), get_nphi(cfg))
    real_coeffs = SHTnsKit.analyze_real(cfg, real_spat)
    real_reconstructed = SHTnsKit.synthesize_real(cfg, real_coeffs)
    real_error = maximum(abs.(real_spat .- real_reconstructed))
    @test real_error < 1e-10
    println("✓ Real roundtrip error: $real_error")
    
    destroy_config(cfg)
    println("✓ All basic tests passed!")
end

println("\n🎉 Tests completed successfully!")
