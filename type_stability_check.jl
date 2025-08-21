#!/usr/bin/env julia

using LinearAlgebra, FFTW, InteractiveUtils
include("src/SHTnsKit.jl")
using .SHTnsKit

function check_type_stability()
    println("=== Type Stability Analysis ===")
    
    # Test configuration creation
    println("\n1. Checking create_gauss_config:")
    cfg = create_gauss_config(4, 4)
    @code_warntype create_gauss_config(4, 4)
    
    # Test allocation functions
    println("\n2. Checking allocate_spectral:")
    @code_warntype allocate_spectral(cfg)
    
    println("\n3. Checking allocate_spatial:")
    @code_warntype allocate_spatial(cfg)
    
    # Test core transforms
    println("\n4. Checking sh_to_spat:")
    coeffs = allocate_spectral(cfg)
    coeffs[1] = 1.0
    @code_warntype sh_to_spat(cfg, coeffs)
    
    println("\n5. Checking spat_to_sh:")
    spatial = ones(get_nlat(cfg), get_nphi(cfg))
    @code_warntype spat_to_sh(cfg, spatial)
    
    # Test FFT utilities
    println("\n6. Checking compute_fourier_coefficients_spatial:")
    @code_warntype SHTnsKit.compute_fourier_coefficients_spatial(spatial, cfg)
    
    # Test utility functions
    println("\n7. Checking power_spectrum:")
    @code_warntype power_spectrum(cfg, coeffs)
    
    println("\n8. Checking lmidx:")
    @code_warntype lmidx(cfg, 1, 0)
    
    # Test Gauss-Legendre computations
    println("\n9. Checking compute_gauss_legendre_nodes_weights:")
    @code_warntype SHTnsKit.compute_gauss_legendre_nodes_weights(10, Float64)
    
    println("\n10. Checking compute_associated_legendre:")
    @code_warntype SHTnsKit.compute_associated_legendre(4, 0.5, SHTnsKit.SHT_ORTHONORMAL)
    
    destroy_config(cfg)
end

check_type_stability()