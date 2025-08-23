#!/usr/bin/env julia

using SHTnsKit

function test_normalization_fix()
    println("Testing normalization fixes based on C code...")
    
    # Test with different normalizations
    normalizations = [SHT_ORTHONORMAL, SHT_FOURPI, SHT_SCHMIDT]
    
    for norm in normalizations
        println("\n=== Testing $(norm) normalization ===")
        
        # Create configuration
        lmax, mmax = 4, 3
        cfg = create_config(Float64, lmax, mmax; norm=norm)
        nlat, nphi = 16, 32
        set_grid!(cfg, nlat, nphi)
        
        # Test round-trip transform for a simple function
        # Create a test spherical harmonic: Y_2^1
        sh_coeffs = zeros(Float64, cfg.nlm)
        test_l, test_m = 2, 1
        
        # Find the coefficient index for (l=2, m=1)
        coeff_idx = 0
        for (idx, (l, m)) in enumerate(cfg.lm_indices)
            if l == test_l && m == test_m
                coeff_idx = idx
                break
            end
        end
        
        if coeff_idx > 0
            sh_coeffs[coeff_idx] = 1.0  # Unit coefficient
            
            # Forward transform: SH -> spatial
            spatial_data = sh_to_spat(cfg, sh_coeffs)
            
            # Backward transform: spatial -> SH  
            recovered_coeffs = spat_to_sh(cfg, spatial_data)
            
            # Check round-trip error
            error = abs(recovered_coeffs[coeff_idx] - 1.0)
            println("Round-trip error for Y_$(test_l)^$(test_m): $(error)")
            
            # Print some normalization values for comparison with C code
            println("Synthesis normalization for (2,1): $(SHTnsKit._get_synthesis_normalization(cfg, test_l, test_m))")
            println("Analysis normalization for (2,1): $(SHTnsKit._get_analysis_normalization(cfg, test_l, test_m))")
            
            if error < 1e-12
                println("✓ Round-trip test PASSED")
            else
                println("✗ Round-trip test FAILED (error = $(error))")
            end
        else
            println("Could not find coefficient index for (2,1)")
        end
    end
    
    println("\nNormalization fix test completed.")
end

# Run the test
test_normalization_fix()