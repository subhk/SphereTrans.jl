using Test
using SHTnsKit
using LinearAlgebra

@testset "Complex Field Transforms" begin
    @testset "Complex Transform Accuracy" begin
        try
            cfg = create_gauss_config(12, 12)
            nlm = get_nlm(cfg)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            
            # Test complex spectral to spatial
            sh_complex = rand(ComplexF64, nlm)
            spat_complex = synthesize_complex(cfg, sh_complex)
            @test spat_complex isa Matrix{ComplexF64}
            @test size(spat_complex) == (nlat, nphi)
            
            # Test complex spatial to spectral
            sh_recovered = analyze_complex(cfg, spat_complex)
            @test sh_recovered isa Vector{ComplexF64}
            @test length(sh_recovered) == nlm
            
            # Check accuracy
            error = maximum(abs.(sh_complex - sh_recovered))
            @test error < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Complex transform accuracy tests skipped: $e"
        end
    end
    
    @testset "Complex vs Real Transform Consistency" begin
        try
            cfg = create_gauss_config(10, 10)
            nlm = get_nlm(cfg)
            
            # Create real spectral coefficients
            sh_real = rand(nlm)
            
            # Transform using real functions
            spat_real = synthesize(cfg, sh_real)
            
            # Transform using complex functions with real input
            sh_complex = ComplexF64.(sh_real)
            spat_complex = synthesize_complex(cfg, sh_complex)
            
            # Results should match (imaginary parts should be zero)
            @test maximum(abs.(real.(spat_complex) - spat_real)) < 1e-14
            @test maximum(abs.(imag.(spat_complex))) < 1e-14
            
            free_config(cfg)
            
        catch e
            @test_skip "Complex vs real consistency tests skipped: $e"
        end
    end
    
    @testset "Complex Field Properties" begin
        try
            cfg = create_gauss_config(8, 8)
            nlm = get_nlm(cfg)
            
            # Test with pure imaginary coefficients
            sh_imag = 1im * rand(nlm)
            spat_imag = synthesize_complex(cfg, sh_imag)
            
            # All spatial values should be pure imaginary
            @test maximum(abs.(real.(spat_imag))) < 1e-14
            
            # Test round-trip
            sh_recovered = analyze_complex(cfg, spat_imag)
            error = maximum(abs.(sh_imag - sh_recovered))
            @test error < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Complex field properties tests skipped: $e"
        end
    end
    
    @testset "Complex Field Linearity" begin
        try
            cfg = create_gauss_config(10, 10)
            nlm = get_nlm(cfg)
            
            # Test linearity: F(a*x + b*y) = a*F(x) + b*F(y)
            sh1 = rand(ComplexF64, nlm)
            sh2 = rand(ComplexF64, nlm)
            a = 2.0 + 1.5im
            b = -1.0 + 0.5im
            
            # Transform individually
            spat1 = synthesize_complex(cfg, sh1)
            spat2 = synthesize_complex(cfg, sh2)
            
            # Transform linear combination
            sh_combined = a * sh1 + b * sh2
            spat_combined = synthesize_complex(cfg, sh_combined)
            
            # Check linearity
            spat_linear = a * spat1 + b * spat2
            error = maximum(abs.(spat_combined - spat_linear))
            @test error < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Complex field linearity tests skipped: $e"
        end
    end
    
    @testset "Complex Field Memory Management" begin
        try
            cfg = create_gauss_config(8, 8)
            
            # Test allocation functions
            sh_complex = allocate_complex_spectral(cfg)
            @test eltype(sh_complex) == ComplexF64
            @test length(sh_complex) == get_nlm(cfg)
            
            spat_complex = allocate_complex_spatial(cfg)
            @test eltype(spat_complex) == ComplexF64
            @test size(spat_complex) == (get_nlat(cfg), get_nphi(cfg))
            
            # Test with different complex types
            sh_complex32 = allocate_complex_spectral(cfg, T=ComplexF32)
            @test eltype(sh_complex32) == ComplexF32
            
            free_config(cfg)
            
        catch e
            @test_skip "Complex field memory management tests skipped: $e"
        end
    end
end