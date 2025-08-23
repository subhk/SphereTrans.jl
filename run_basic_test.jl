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
    # Create random band-limited complex spectral coefficients
    cplx_nlm = SHTnsKit._cplx_nlm(cfg)
    original_cplx_coeffs = randn(ComplexF64, cplx_nlm)
    
    # Transform: spectral -> spatial -> spectral
    spat = SHTnsKit.cplx_sh_to_spat(cfg, original_cplx_coeffs)
    reconstructed_coeffs = SHTnsKit.cplx_spat_to_sh(cfg, spat)
    
    # Check roundtrip error
    max_error = maximum(abs.(original_cplx_coeffs .- reconstructed_coeffs))
    @test max_error < 1e-12
    println("✓ Complex roundtrip error: $max_error")
    
    # Test real transforms
    println("Testing real transforms...")
    # Create random band-limited real spectral coefficients
    original_real_coeffs = randn(Float64, get_nlm(cfg))
    real_spat = sh_to_spat(cfg, original_real_coeffs)
    reconstructed_real_coeffs = spat_to_sh(cfg, real_spat)
    real_error = maximum(abs.(original_real_coeffs .- reconstructed_real_coeffs))
    @test real_error < 1e-12
    println("✓ Real roundtrip error: $real_error")
    
    destroy_config(cfg)
    println("✓ All basic tests passed!")
end

println("\n🎉 Tests completed successfully!")
