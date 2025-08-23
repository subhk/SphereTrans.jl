#!/usr/bin/env julia --startup-file=no --compile=no

# Proper roundtrip test using band-limited data
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Testing proper roundtrip...")

# Create config
cfg = create_gauss_config(4, 4)
nlm = get_nlm(cfg)

# Test real transforms
println("Creating random band-limited spectral data...")
original_coeffs = randn(Float64, nlm)

# Synthesize to spatial domain
spatial = sh_to_spat(cfg, original_coeffs)
println("Synthesized to spatial domain")

# Analyze back to spectral domain
reconstructed_coeffs = spat_to_sh(cfg, spatial)
println("Analyzed back to spectral domain")

# Check the coefficient error
coeff_error = maximum(abs.(original_coeffs .- reconstructed_coeffs))
println("Real coefficient roundtrip error: $coeff_error")

# Also check spatial roundtrip
reconstructed_spatial = sh_to_spat(cfg, reconstructed_coeffs)
spatial_error = maximum(abs.(spatial .- reconstructed_spatial))
println("Real spatial roundtrip error: $spatial_error")

# Test complex transforms
println("\nTesting complex transforms...")
cplx_nlm = SHTnsKit._cplx_nlm(cfg)
original_cplx_coeffs = randn(ComplexF64, cplx_nlm)

# Synthesize to spatial domain
cplx_spatial = SHTnsKit.cplx_sh_to_spat(cfg, original_cplx_coeffs)
println("Complex synthesized to spatial domain")

# Analyze back to spectral domain
reconstructed_cplx_coeffs = SHTnsKit.cplx_spat_to_sh(cfg, cplx_spatial)
println("Complex analyzed back to spectral domain")

# Check the coefficient error
cplx_coeff_error = maximum(abs.(original_cplx_coeffs .- reconstructed_cplx_coeffs))
println("Complex coefficient roundtrip error: $cplx_coeff_error")

# Also check spatial roundtrip
reconstructed_cplx_spatial = SHTnsKit.cplx_sh_to_spat(cfg, reconstructed_cplx_coeffs)
cplx_spatial_error = maximum(abs.(cplx_spatial .- reconstructed_cplx_spatial))
println("Complex spatial roundtrip error: $cplx_spatial_error")

if coeff_error < 1e-12 && spatial_error < 1e-12 && cplx_coeff_error < 1e-12 && cplx_spatial_error < 1e-12
    println("\n✓ All tests passed!")
else
    println("\n✗ Some tests failed!")
    println("  Real coefficient error: $coeff_error")
    println("  Real spatial error: $spatial_error")
    println("  Complex coefficient error: $cplx_coeff_error") 
    println("  Complex spatial error: $cplx_spatial_error")
end