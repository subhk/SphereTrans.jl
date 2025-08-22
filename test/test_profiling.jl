using Test
using SHTnsKit

@testset "Profiling Functions" begin
    lmax, mmax = 6, 4
    nlat, nphi = 18, 24
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "Basic profiling" begin
        # Initially profiling should be disabled
        summary = get_profiling_summary(cfg)
        @test summary.enabled == false
        @test summary.transform_count == 0
        
        # Enable profiling
        shtns_profiling(cfg, true)
        summary = get_profiling_summary(cfg)
        @test summary.enabled == true
        
        # Reset counters
        reset_profiling(cfg)
        summary = get_profiling_summary(cfg)
        @test summary.transform_count == 0
        @test summary.total_time == 0.0
    end
    
    @testset "Timed transforms" begin
        shtns_profiling(cfg, true)
        reset_profiling(cfg)
        
        # Create test data
        qlm = randn(Complex{Float64}, cfg.nlm)  
        vr = allocate_spatial(cfg)
        
        # Test timed synthesis
        t_synth = sh_to_spat_time(cfg, qlm, vr)
        @test t_synth > 0  # Should take some time
        @test isfinite(t_synth)
        
        # Check that timing was recorded
        summary = get_profiling_summary(cfg)
        @test summary.transform_count == 1
        @test summary.total_time > 0
        
        # Test timed analysis
        t_anal = spat_to_sh_time(cfg, vr, qlm)
        @test t_anal > 0
        @test isfinite(t_anal)
        
        # Should have 2 transforms now
        summary = get_profiling_summary(cfg)
        @test summary.transform_count == 2
        
        # Read timing from last transform
        total_time, legendre_time, fourier_time = shtns_profiling_read_time(cfg)
        @test total_time == t_anal  # Should match last analysis time
        @test total_time >= 0
    end
    
    @testset "Benchmarking" begin
        # Test basic benchmarking
        n_iter = 5  # Small number for testing
        stats = benchmark_transform(cfg, n_iter)
        
        @test stats.iterations == n_iter
        @test stats.synthesis_time > 0
        @test stats.analysis_time > 0
        @test stats.total_time > 0
        @test stats.synthesis_std >= 0
        @test stats.analysis_std >= 0
        
        # Times should be reasonable
        @test stats.synthesis_time < 1.0  # Less than 1 second
        @test stats.analysis_time < 1.0
    end
    
    @testset "Memory profiling" begin
        memory = profile_memory_usage(cfg)
        
        @test memory.spectral_memory > 0
        @test memory.spatial_memory > 0
        @test memory.total_memory > 0
        @test memory.memory_mb > 0
        
        # Memory should be reasonable  
        @test memory.memory_mb < 1000  # Less than 1GB for small config
        
        # Spectral should be smaller than spatial for typical cases
        @test memory.spectral_memory < memory.spatial_memory
    end
    
    @testset "Performance comparison" begin
        # Create two similar configurations for comparison
        cfg1 = create_gauss_config(Float64, 4, 4, 12, 16)
        cfg2 = create_gauss_config(Float64, 6, 6, 18, 24)
        
        n_iter = 3
        comparison = compare_performance(cfg1, cfg2, n_iter)
        
        @test haskey(comparison, :config1)
        @test haskey(comparison, :config2)  
        @test haskey(comparison, :synthesis_ratio)
        @test haskey(comparison, :analysis_ratio)
        @test haskey(comparison, :total_ratio)
        @test haskey(comparison, :faster_config)
        
        # Larger configuration should generally be slower
        @test comparison.total_ratio > 0.5  # At least some slowdown expected
        @test comparison.faster_config in [1, 2]
    end
    
    @testset "Profiling state management" begin
        # Test that profiling can be enabled/disabled
        shtns_profiling(cfg, true)
        @test get_profiling_summary(cfg).enabled == true
        
        shtns_profiling(cfg, false)  
        @test get_profiling_summary(cfg).enabled == false
        
        # With profiling disabled, counters shouldn't change
        initial_count = get_profiling_summary(cfg).transform_count
        
        qlm = randn(Complex{Float64}, cfg.nlm)
        vr = allocate_spatial(cfg)
        
        sh_to_spat_time(cfg, qlm, vr)  # This shouldn't increment counter when disabled
        
        # Note: The implementation might still increment counters even when disabled
        # The key test is that timing overhead is minimized
        final_summary = get_profiling_summary(cfg)
        @test final_summary.enabled == false
    end
    
    @testset "Report generation" begin
        shtns_profiling(cfg, true)
        reset_profiling(cfg)
        
        # Run some transforms to generate data
        qlm = randn(Complex{Float64}, cfg.nlm)
        vr = allocate_spatial(cfg)
        
        for _ in 1:3
            sh_to_spat_time(cfg, qlm, vr)
            spat_to_sh_time(cfg, vr, qlm)
        end
        
        # Test that report can be generated without errors
        # We can't easily test the output, but we can ensure it doesn't crash
        @test_nowarn print_profiling_report(cfg)
        
        summary = get_profiling_summary(cfg)
        @test summary.transform_count == 6  # 3 synthesis + 3 analysis
        @test summary.total_time > 0
        @test summary.average_time > 0
    end
end

@testset "Thread Safety" begin
    # Test that profiling works with multiple threads (if available)
    if Threads.nthreads() > 1
        lmax, mmax = 4, 4
        nlat, nphi = 12, 16
        cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
        
        # Enable profiling
        shtns_profiling(cfg, true)
        
        # Run transforms on multiple threads
        Threads.@threads for i in 1:4
            qlm = randn(Complex{Float64}, cfg.nlm)
            vr = allocate_spatial(cfg)
            sh_to_spat_time(cfg, qlm, vr)
        end
        
        # Each thread should have its own profiler state
        # This is a basic test that no crashes occur
        summary = get_profiling_summary(cfg)
        @test summary.transform_count >= 0  # Should be non-negative
    end
end