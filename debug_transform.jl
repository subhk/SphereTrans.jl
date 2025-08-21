using LinearAlgebra, FFTW, Statistics
include("src/SHTnsKit.jl")
using .SHTnsKit

function debug_transform()
    cfg = create_gauss_config(4, 4)
    
    println("=== DEBUGGING THE NORMALIZATION ISSUE ===")
    
    # The issue: FFT of constant field gives nphi * constant
    # But we need the coefficient for spherical harmonic analysis
    
    constant_field = ones(get_nlat(cfg), get_nphi(cfg))
    fourier_coeffs = SHTnsKit.compute_fourier_coefficients_spatial(constant_field, cfg)
    
    # Extract m=0 mode
    mode_data = Vector{ComplexF64}(undef, get_nlat(cfg))
    SHTnsKit.extract_fourier_mode!(fourier_coeffs, 0, mode_data, get_nlat(cfg))
    
    println("nphi: $(get_nphi(cfg))")
    println("m=0 mode data[1]: $(mode_data[1])")
    
    # Manual integration for Y_0^0 coefficient
    total = 0.0
    for i in 1:get_nlat(cfg)
        plm_val = cfg.plm_cache[i, 1]  # Y_0^0 PLM value  
        weight = cfg.gauss_weights[i]
        contrib = real(mode_data[i]) * plm_val * weight
        total += contrib
        if i <= 3
            println("i=$i: mode_data=$(real(mode_data[i])), plm=$plm_val, weight=$weight, contrib=$contrib")
        end
    end
    
    println("\nTotal integral: $total")
    expected = sqrt(4π)
    println("Expected Y_0^0 coefficient: $expected")
    println("Ratio (actual/expected): $(total / expected)")
    
    # The ratio shows we're off by a factor. Let's think about this:
    # - FFT gives us integral over φ (factor of 2π for constant)
    # - But FFT of constant over nphi points gives nphi, not 2π
    # - So we need to multiply by 2π/nphi to get the proper φ integration
    
    # The φ integration should give 2π for a constant field
    corrected_coeff = total * (2π / get_nphi(cfg))
    println("Corrected coefficient (proper φ integration): $corrected_coeff") 
    println("Corrected ratio: $(corrected_coeff / expected)")
    
    # Alternative: let's check what the exact ratio should be
    println("\nAnalysis:")
    println("nphi = $(get_nphi(cfg))")
    println("2π/nphi = $(2π / get_nphi(cfg))")
    println("Ratio should be 1.0 for perfect agreement")
    
    destroy_config(cfg)
    return corrected_coeff, expected
end

debug_transform()