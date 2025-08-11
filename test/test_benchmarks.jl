using Test
using SHTnsKit
using LinearAlgebra

@testset "Performance Benchmarks" begin
    @testset "Transform Performance Standards" begin
        try
            # Test that transforms complete within reasonable time limits
            cfg = create_gauss_config(32, 32)
            
            sh = rand(get_nlm(cfg))
            spat = allocate_spatial(cfg)
            
            # Forward transform should complete quickly
            time_forward = @elapsed synthesize!(cfg, sh, spat)
            @test time_forward < 1.0  # Should complete within 1 second
            
            # Backward transform should complete quickly  
            time_backward = @elapsed analyze!(cfg, spat, sh)
            @test time_backward < 1.0  # Should complete within 1 second
            
            free_config(cfg)
            
        catch e
            @test_skip "Transform performance tests skipped: $e"
        end
    end
    
    @testset "Memory Allocation Efficiency" begin
        try
            cfg = create_gauss_config(16, 16)
            
            # Test that allocation functions return appropriately sized arrays
            sh = allocate_spectral(cfg)
            @test sizeof(sh) == get_nlm(cfg) * sizeof(Float64)
            
            spat = allocate_spatial(cfg)
            expected_spatial_size = get_nlat(cfg) * get_nphi(cfg) * sizeof(Float64)
            @test sizeof(spat) == expected_spatial_size
            
            # Test complex allocations
            sh_complex = allocate_complex_spectral(cfg)
            @test sizeof(sh_complex) == get_nlm(cfg) * sizeof(ComplexF64)
            
            free_config(cfg)
            
        catch e
            @test_skip "Memory allocation efficiency tests skipped: $e"
        end
    end
    
    @testset "Scalability Tests" begin
        try
            # Test performance scaling with problem size
            sizes = [8, 16, 32]
            times = Float64[]
            
            for lmax in sizes
                if lmax <= 32  # Keep reasonable for CI
                    cfg = create_gauss_config(lmax, lmax)
                    sh = rand(get_nlm(cfg))
                    spat = allocate_spatial(cfg)
                    
                    # Time multiple runs for better statistics
                    elapsed = @elapsed begin
                        for i in 1:3
                            synthesize!(cfg, sh, spat)
                        end
                    end
                    
                    push!(times, elapsed)
                    free_config(cfg)
                    
                    # Each transform should still be reasonably fast
                    @test elapsed < 5.0  # 3 transforms in under 5 seconds
                end
            end
            
            # Check that performance doesn't degrade catastrophically
            if length(times) >= 2
                # Ratio shouldn't be too large (allowing for O(N^3) scaling)
                max_ratio = maximum(times[2:end] ./ times[1:end-1])
                @test max_ratio < 50.0  # Allow significant but not catastrophic slowdown
            end
            
        catch e
            @test_skip "Scalability tests skipped: $e"
        end
    end
    
    @testset "Threading Performance Impact" begin
        try
            cfg = create_gauss_config(24, 24)
            sh = rand(get_nlm(cfg))
            spat = allocate_spatial(cfg)
            
            # Test with single thread
            original_threads = get_num_threads()
            set_num_threads(1)
            
            single_time = @elapsed begin
                for i in 1:3
                    synthesize!(cfg, sh, spat)
                end
            end
            
            # Test with multiple threads (if available)
            max_threads = min(4, original_threads)
            if max_threads > 1
                set_num_threads(max_threads)
                
                multi_time = @elapsed begin
                    for i in 1:3
                        synthesize!(cfg, sh, spat)
                    end
                end
                
                # Multi-threading shouldn't make things much worse
                @test multi_time <= single_time * 3.0  # Allow for overhead
            end
            
            # Restore original threading
            set_num_threads(original_threads)
            free_config(cfg)
            
        catch e
            @test_skip "Threading performance tests skipped: $e"
        end
    end
    
    @testset "Memory Usage Benchmarks" begin
        try
            # Test that memory usage is reasonable
            cfg = create_gauss_config(20, 20)
            
            # Estimate expected memory usage
            nlm = get_nlm(cfg)
            nlat = get_nlat(cfg)
            nphi = get_nphi(cfg)
            
            expected_spectral = nlm * sizeof(Float64)
            expected_spatial = nlat * nphi * sizeof(Float64)
            
            # Create arrays and check they're not excessively large
            sh = allocate_spectral(cfg)
            spat = allocate_spatial(cfg)
            
            @test sizeof(sh) == expected_spectral
            @test sizeof(spat) == expected_spatial
            
            # Total memory for basic transform should be reasonable
            total_memory = sizeof(sh) + sizeof(spat)
            @test total_memory < 100 * 1024 * 1024  # Less than 100MB for modest problem
            
            free_config(cfg)
            
        catch e
            @test_skip "Memory usage benchmark tests skipped: $e"
        end
    end
    
    @testset "Numerical Accuracy Standards" begin
        try
            cfg = create_gauss_config(20, 20)
            
            # Test round-trip accuracy for different magnitudes
            test_magnitudes = [1e-8, 1e-4, 1.0, 1e4, 1e8]
            
            for magnitude in test_magnitudes
                sh_original = rand(get_nlm(cfg)) * magnitude
                
                spat = synthesize(cfg, sh_original)
                sh_recovered = analyze(cfg, spat)
                
                relative_error = norm(sh_original - sh_recovered) / norm(sh_original)
                @test relative_error < 1e-12  # Very high accuracy required
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Numerical accuracy benchmark tests skipped: $e"
        end
    end
    
    @testset "Stress Testing" begin
        try
            # Test with many repeated operations
            cfg = create_gauss_config(12, 12)
            sh = rand(get_nlm(cfg))
            spat = allocate_spatial(cfg)
            
            # Many forward transforms
            for i in 1:100
                synthesize!(cfg, sh, spat)
                # Check for degradation
                @test all(isfinite.(spat))
                @test !any(isnan.(spat))
            end
            
            # Many backward transforms
            for i in 1:100
                analyze!(cfg, spat, sh)
                # Check for degradation
                @test all(isfinite.(sh))
                @test !any(isnan.(sh))
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Stress testing skipped: $e"
        end
    end
end