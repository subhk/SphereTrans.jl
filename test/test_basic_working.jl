using Test
using SHTnsKit

@testset "Complex transforms: roundtrip" begin
    for (lmax, mmax) in [(2, 2), (4, 4)]
        cfg = create_gauss_config(lmax, mmax)
        
        # Create some test complex data
        spat = rand(Complex{Float64}, get_nlat(cfg), get_nphi(cfg))
        
        # Transform: spatial -> spectral -> spatial
        sh = cplx_spat_to_sh(cfg, spat)
        reconstructed = cplx_sh_to_spat(cfg, sh)
        
        # Check roundtrip error
        max_error = maximum(abs.(spat .- reconstructed))
        @test max_error < 1e-12
        
        destroy_config(cfg)
    end
end

@testset "Complex transforms: single Y_l^m" begin
    for (lmax, mmax) in [(4, 4), (8, 4)]
        cfg = create_gauss_config(lmax, mmax)
        
        # Test individual spherical harmonics
        for l in 0:min(2, lmax)
            for m in -min(l, mmax):min(l, mmax)
                # Create field with single Y_l^m
                field = create_complex_test_field(cfg, l, m)
                
                # Transform to spectral
                sh = cplx_spat_to_sh(cfg, field)
                
                # The coefficient for Y_l^m should be non-zero, others should be small
                expected_idx = findfirst(lm -> lm == (l, m), cfg.lm_indices)
                if expected_idx !== nothing
                    @test abs(sh[expected_idx]) > 0.1  # Significant coefficient
                    
                    # Other coefficients should be small
                    for (idx, coeff) in enumerate(sh)
                        if idx != expected_idx
                            @test abs(coeff) < 1e-10
                        end
                    end
                end
            end
        end
        
        destroy_config(cfg)
    end
end
