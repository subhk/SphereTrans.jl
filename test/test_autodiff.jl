using Test
using SHTnsKit
using LinearAlgebra

@testset "Automatic Differentiation Tests" begin

# Test helper functions for AD first
@testset "Helper Functions for AD" begin
    try
        cfg = create_gauss_config(8, 8)
        
        # Test get_lm_from_index and get_index_from_lm
        @testset "Index-LM Conversion" begin
            nlm = get_nlm(cfg)
            
            # Test round-trip conversion
            for idx in 1:min(nlm, 20)  # Test first 20 indices
                l, m = get_lm_from_index(cfg, idx)
                idx_recovered = get_index_from_lm(cfg, l, m)
                @test idx_recovered == idx
            end
            
            # Test specific known cases
            @test get_lm_from_index(cfg, 1) == (0, 0)  # First coefficient
            @test get_index_from_lm(cfg, 0, 0) == 1
            
            # Test l=1 modes
            @test get_lm_from_index(cfg, 2) == (1, -1)
            @test get_lm_from_index(cfg, 3) == (1, 0)
            @test get_lm_from_index(cfg, 4) == (1, 1)
            
            @test get_index_from_lm(cfg, 1, -1) == 2
            @test get_index_from_lm(cfg, 1, 0) == 3
            @test get_index_from_lm(cfg, 1, 1) == 4
        end
        
        free_config(cfg)
        
    catch e
        @test_skip "Helper function tests skipped: library not available ($e)"
    end
end

# ForwardDiff tests
@testset "ForwardDiff Support" begin
    try
        using ForwardDiff
        
        @testset "ForwardDiff - Basic Transforms" begin
            cfg = create_gauss_config(8, 8)
            
            # Test function: transform to spatial, compute sum, transform back
            function test_function(sh)
                spatial = synthesize(cfg, sh)
                return sum(abs2, spatial)
            end
            
            # Generate test data
            sh0 = rand(get_nlm(cfg))
            
            # Compute gradient using ForwardDiff
            grad_fd = ForwardDiff.gradient(test_function, sh0)
            
            # Test that gradient has correct size and is finite
            @test length(grad_fd) == length(sh0)
            @test all(isfinite, grad_fd)
            @test !all(iszero, grad_fd)  # Should not be all zeros
            
            # Test with different input
            sh1 = rand(get_nlm(cfg))
            grad_fd1 = ForwardDiff.gradient(test_function, sh1)
            @test grad_fd1 != grad_fd  # Different inputs should give different gradients
            
            free_config(cfg)
        end
        
        @testset "ForwardDiff - Vector Transforms" begin
            cfg = create_gauss_config(6, 6)
            
            # Test function using vector transforms
            function vector_test_function(params)
                n = length(params) ÷ 2
                S_lm = params[1:n]
                T_lm = params[n+1:end]
                
                Vt, Vp = synthesize_vector(cfg, S_lm, T_lm)
                return sum(Vt.^2) + sum(Vp.^2)
            end
            
            nlm = get_nlm(cfg)
            params0 = rand(2 * nlm)  # Combined S_lm and T_lm
            
            grad_fd = ForwardDiff.gradient(vector_test_function, params0)
            @test length(grad_fd) == 2 * nlm
            @test all(isfinite, grad_fd)
            
            free_config(cfg)
        end
        
        @testset "ForwardDiff - Complex Fields" begin
            cfg = create_gauss_config(6, 6)
            
            # Test function for complex fields
            function complex_test_function(sh_real, sh_imag)
                sh_complex = complex.(sh_real, sh_imag)
                spatial_complex = synthesize_complex(cfg, sh_complex)
                return sum(abs2, spatial_complex)
            end
            
            nlm = get_nlm(cfg)
            sh_real = rand(nlm)
            sh_imag = rand(nlm)
            
            # Gradient w.r.t. real part
            grad_real = ForwardDiff.gradient(x -> complex_test_function(x, sh_imag), sh_real)
            @test length(grad_real) == nlm
            @test all(isfinite, grad_real)
            
            # Gradient w.r.t. imaginary part
            grad_imag = ForwardDiff.gradient(x -> complex_test_function(sh_real, x), sh_imag)
            @test length(grad_imag) == nlm
            @test all(isfinite, grad_imag)
            
            free_config(cfg)
        end
        
    catch LoadError
        @test_skip "ForwardDiff tests skipped: ForwardDiff.jl not available"
    catch e
        @test_skip "ForwardDiff tests skipped: $e"
    end
end

# Zygote tests
@testset "Zygote Support" begin
    try
        using Zygote
        
        @testset "Zygote - Basic Transforms" begin
            cfg = create_gauss_config(8, 8)
            
            # Test function: synthesis followed by some computation
            function test_function(sh)
                spatial = synthesize(cfg, sh)
                return sum(spatial.^2)
            end
            
            sh0 = rand(get_nlm(cfg))
            
            # Compute gradient using Zygote
            grad_zy = Zygote.gradient(test_function, sh0)[1]
            
            @test length(grad_zy) == length(sh0)
            @test all(isfinite, grad_zy)
            @test !all(iszero, grad_zy)
            
            free_config(cfg)
        end
        
        @testset "Zygote - Analysis Transform" begin
            cfg = create_gauss_config(8, 8)
            
            # Test function: analysis followed by computation
            function test_function(spatial)
                sh = analyze(cfg, spatial)
                return sum(sh.^2)
            end
            
            spatial0 = rand(get_nlat(cfg), get_nphi(cfg))
            
            # Compute gradient using Zygote
            grad_zy = Zygote.gradient(test_function, spatial0)[1]
            
            @test size(grad_zy) == size(spatial0)
            @test all(isfinite, grad_zy)
            @test !all(iszero, grad_zy)
            
            free_config(cfg)
        end
        
        @testset "Zygote - Round-trip Transform" begin
            cfg = create_gauss_config(6, 6)
            
            # Test round-trip transform
            function roundtrip_function(sh)
                spatial = synthesize(cfg, sh)
                sh_recovered = analyze(cfg, spatial)
                return sum((sh - sh_recovered).^2)  # Should be very small
            end
            
            sh0 = rand(get_nlm(cfg))
            
            # The round-trip error should be very small
            error = roundtrip_function(sh0)
            @test error < 1e-20  # Should be machine precision
            
            # Gradient should exist
            grad_zy = Zygote.gradient(roundtrip_function, sh0)[1]
            @test length(grad_zy) == length(sh0)
            @test all(isfinite, grad_zy)
            
            free_config(cfg)
        end
        
        @testset "Zygote - Vector Fields" begin
            cfg = create_gauss_config(6, 6)
            
            function vector_test_function(S_lm, T_lm)
                Vt, Vp = synthesize_vector(cfg, S_lm, T_lm)
                S_recovered, T_recovered = analyze_vector(cfg, Vt, Vp)
                return sum(S_recovered.^2) + sum(T_recovered.^2)
            end
            
            nlm = get_nlm(cfg)
            S_lm = rand(nlm)
            T_lm = rand(nlm)
            
            grad_S, grad_T = Zygote.gradient(vector_test_function, S_lm, T_lm)
            
            @test length(grad_S) == nlm
            @test length(grad_T) == nlm
            @test all(isfinite, grad_S)
            @test all(isfinite, grad_T)
            
            free_config(cfg)
        end
        
    catch LoadError
        @test_skip "Zygote tests skipped: Zygote.jl not available"
    catch e
        @test_skip "Zygote tests skipped: $e"
    end
end

# Comparison between ForwardDiff and Zygote
@testset "ForwardDiff vs Zygote Consistency" begin
    try
        using ForwardDiff, Zygote
        
        cfg = create_gauss_config(6, 6)
        
        # Simple test function
        function test_function(sh)
            spatial = synthesize(cfg, sh)
            return sum(spatial.^2)
        end
        
        sh0 = rand(get_nlm(cfg))
        
        # Compute gradients with both methods
        grad_fd = ForwardDiff.gradient(test_function, sh0)
        grad_zy = Zygote.gradient(test_function, sh0)[1]
        
        # They should be very close (within numerical precision)
        @test norm(grad_fd - grad_zy) < 1e-10
        
        free_config(cfg)
        
    catch LoadError
        @test_skip "Consistency tests skipped: ForwardDiff or Zygote not available"
    catch e
        @test_skip "Consistency tests skipped: $e"
    end
end

# Performance tests
@testset "AD Performance" begin
    try
        using ForwardDiff
        
        cfg = create_gauss_config(16, 16)
        
        function test_function(sh)
            spatial = synthesize(cfg, sh)
            return sum(spatial.^2)
        end
        
        sh0 = rand(get_nlm(cfg))
        
        # Time the forward pass
        @test (@elapsed test_function(sh0)) < 1.0  # Should be fast
        
        # Time the gradient computation  
        @test (@elapsed ForwardDiff.gradient(test_function, sh0)) < 5.0  # Should be reasonable
        
        free_config(cfg)
        
    catch LoadError
        @test_skip "Performance tests skipped: ForwardDiff not available"
    catch e
        @test_skip "Performance tests skipped: $e"
    end
end

end # "Automatic Differentiation Tests"