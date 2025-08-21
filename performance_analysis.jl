#!/usr/bin/env julia

using LinearAlgebra, FFTW, InteractiveUtils
include("src/SHTnsKit.jl")
using .SHTnsKit

function analyze_performance_issues()
    println("=== Performance Analysis ===")
    
    cfg = create_gauss_config(8, 8)
    coeffs = allocate_spectral(cfg)
    coeffs[1] = 1.0
    spatial = ones(get_nlat(cfg), get_nphi(cfg))
    
    println("\n1. Memory allocation analysis:")
    
    # Test in-place transforms (should allocate minimal memory)
    spatial_out = allocate_spatial(cfg)
    coeffs_out = allocate_spectral(cfg)
    
    println("sh_to_spat! allocations:")
    @allocated sh_to_spat!(cfg, coeffs, spatial_out)
    allocs = @allocated sh_to_spat!(cfg, coeffs, spatial_out)
    println("  Allocated: $allocs bytes")
    
    println("spat_to_sh! allocations:")
    allocs = @allocated spat_to_sh!(cfg, spatial, coeffs_out)
    println("  Allocated: $allocs bytes")
    
    # Check FFT utilities
    println("compute_fourier_coefficients_spatial allocations:")
    allocs = @allocated SHTnsKit.compute_fourier_coefficients_spatial(spatial, cfg)
    println("  Allocated: $allocs bytes")
    
    println("compute_spatial_from_fourier allocations:")
    fourier_coeffs = SHTnsKit.compute_fourier_coefficients_spatial(spatial, cfg)
    allocs = @allocated SHTnsKit.compute_spatial_from_fourier(fourier_coeffs, cfg)
    println("  Allocated: $allocs bytes")
    
    println("\n2. Checking for type instabilities in hot paths:")
    
    # Look for Any types and type instabilities in critical functions
    spatial_temp = ones(cfg.nlat, cfg.nphi)
    
    # Check the internal implementation functions
    println("Checking _sh_to_spat_impl!:")
    test_func1() = SHTnsKit._sh_to_spat_impl!(cfg, coeffs, spatial_temp)
    @code_warntype test_func1()
    
    println("\nChecking _spat_to_sh_impl!:")  
    test_func2() = SHTnsKit._spat_to_sh_impl!(cfg, spatial_temp, coeffs_out)
    @code_warntype test_func2()
    
    destroy_config(cfg)
    
    println("\n3. Checking for concrete types in struct fields:")
    cfg2 = create_gauss_config(4, 4)
    println("SHTnsConfig field types:")
    for field in fieldnames(SHTnsConfig)
        field_type = typeof(getfield(cfg2, field))
        println("  $field: $field_type")
    end
    
    destroy_config(cfg2)
end

analyze_performance_issues()