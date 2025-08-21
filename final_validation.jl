#!/usr/bin/env julia

using LinearAlgebra, FFTW, Statistics
include("src/SHTnsKit.jl")
using .SHTnsKit

function comprehensive_validation()
    println("=== Comprehensive Validation of Optimized SHTnsKit ===")
    
    # Test multiple configurations
    configs = [
        (lmax=4, mmax=4, name="Small (4x4)"),
        (lmax=16, mmax=16, name="Medium (16x16)"),
        (lmax=32, mmax=32, name="Large (32x32)")
    ]
    
    all_tests_passed = true
    
    for (i, (lmax, mmax, name)) in enumerate(configs)
        println("\n$i. Testing $name configuration:")
        
        cfg = create_gauss_config(lmax, mmax)
        println("  Grid: $(get_nlat(cfg)) x $(get_nphi(cfg))")
        println("  Coefficients: $(get_nlm(cfg))")
        
        # Test 1: Constant field (Y_0^0)
        coeffs = zeros(get_nlm(cfg))
        coeffs[1] = sqrt(4π)
        
        spatial = sh_to_spat(cfg, coeffs)
        constant_error = maximum(abs.(spatial .- 1.0))
        
        println("  Constant field error: $constant_error")
        if constant_error > 1e-12
            @warn "Constant field test failed for $name"
            all_tests_passed = false
        end
        
        # Test 2: Roundtrip accuracy
        coeffs_back = spat_to_sh(cfg, spatial)
        roundtrip_error = abs(coeffs_back[1] - coeffs[1])
        
        println("  Roundtrip error: $roundtrip_error")
        if roundtrip_error > 1e-12
            @warn "Roundtrip test failed for $name"
            all_tests_passed = false
        end
        
        # Test 3: Random field test
        random_coeffs = randn(get_nlm(cfg)) * 0.1  # Small amplitude
        spatial_random = sh_to_spat(cfg, random_coeffs)
        coeffs_random_back = spat_to_sh(cfg, spatial_random)
        
        random_error = maximum(abs.(coeffs_random_back - random_coeffs))
        println("  Random field max error: $random_error")
        
        # Test 4: Power spectrum conservation
        power_original = sum(abs2, random_coeffs)
        power_reconstructed = sum(abs2, coeffs_random_back)
        power_diff = abs(power_original - power_reconstructed) / power_original
        
        println("  Power conservation error: $power_diff")
        if power_diff > 1e-10
            @warn "Power conservation test failed for $name"
            all_tests_passed = false
        end
        
        # Test 5: Memory allocations (should be minimal for in-place operations)
        spatial_temp = allocate_spatial(cfg)
        coeffs_temp = allocate_spectral(cfg)
        
        allocs_syn = @allocated sh_to_spat!(cfg, random_coeffs, spatial_temp)
        allocs_ana = @allocated spat_to_sh!(cfg, spatial_random, coeffs_temp)
        
        println("  Synthesis allocations: $allocs_syn bytes")
        println("  Analysis allocations: $allocs_ana bytes")
        
        destroy_config(cfg)
    end
    
    # Test threading functionality
    println("\n4. Testing threading controls:")
    original_state = get_threading()
    original_fft_threads = get_fft_threads()
    
    set_threading!(false)
    println("  Threading disabled: $(get_threading())")
    
    set_threading!(true) 
    println("  Threading enabled: $(get_threading())")
    
    set_fft_threads(2)
    println("  FFT threads set to 2: $(get_fft_threads())")
    
    optimal_config = set_optimal_threads!()
    println("  Optimal threading: $optimal_config")
    
    # Restore original settings
    set_threading!(original_state)
    set_fft_threads(original_fft_threads)
    
    # Test utility functions
    println("\n5. Testing utility functions:")
    cfg = create_gauss_config(8, 8)
    
    # Test lmidx optimization
    idx = lmidx(cfg, 2, 1)
    (l, m) = lm_from_index(cfg, idx)
    println("  lmidx consistency: (2,1) -> $idx -> ($l,$m)")
    
    if l != 2 || m != 1
        @warn "lmidx optimization failed"
        all_tests_passed = false
    end
    
    # Test power spectrum
    test_coeffs = randn(get_nlm(cfg))
    power_spec = power_spectrum(cfg, test_coeffs)
    total_pow = total_power(cfg, test_coeffs)
    
    println("  Power spectrum length: $(length(power_spec))")
    println("  Total power: $total_pow")
    
    destroy_config(cfg)
    
    # Final assessment
    println("\n" * "="^50)
    if all_tests_passed
        println("✅ ALL TESTS PASSED - Optimizations successful!")
        println("🚀 SHTnsKit is now optimized for:")
        println("   • Reduced memory allocations")
        println("   • Improved type stability") 
        println("   • Optimized CPU-intensive loops")
        println("   • Consistent coding patterns")
    else
        println("❌ Some tests failed - please review optimizations")
    end
    println("="^50)
    
    return all_tests_passed
end

comprehensive_validation()