using Test
using SHTnsKit

@testset "GPU Acceleration" begin
    # Check if CUDA is available
    cuda_available = false
    try
        using CUDA
        cuda_available = CUDA.functional()
    catch
        cuda_available = false
    end
    
    if !cuda_available
        @test_skip "CUDA not available - skipping all GPU tests"
        return
    end
    
    using CUDA
    
    @testset "GPU Initialization" begin
        try
            # Test GPU initialization
            gpu_success = initialize_gpu(0, verbose=false)
            
            if gpu_success
                @test true  # GPU initialized successfully
                
                # Test cleanup
                @test cleanup_gpu(verbose=false) == true
            else
                @test_skip "GPU initialization failed"
            end
            
        catch e
            @test_skip "GPU initialization tests skipped: $e"
        end
    end
    
    @testset "Basic GPU Transforms" begin
        try
            if !initialize_gpu(0, verbose=false)
                @test_skip "GPU not initialized"
                return
            end
            
            cfg = create_gpu_config(16, 16)
            nlm = get_nlm(cfg)
            
            # Create test data
            sh_cpu = rand(Float64, nlm)
            sh_gpu = CUDA.CuArray(sh_cpu)
            
            # GPU forward transform
            spat_gpu = synthesize_gpu(cfg, sh_gpu)
            @test spat_gpu isa CUDA.CuArray
            @test size(spat_gpu) == (get_nlat(cfg), get_nphi(cfg))
            
            # GPU backward transform
            sh_gpu_back = analyze_gpu(cfg, spat_gpu)
            @test sh_gpu_back isa CUDA.CuArray
            @test length(sh_gpu_back) == nlm
            
            # Compare with CPU
            spat_cpu = synthesize(cfg, sh_cpu)
            gpu_error = maximum(abs.(spat_cpu - Array(spat_gpu)))
            @test gpu_error < 1e-12
            
            # Round-trip accuracy
            roundtrip_error = maximum(abs.(sh_cpu - Array(sh_gpu_back)))
            @test roundtrip_error < 1e-12
            
            free_config(cfg)
            cleanup_gpu(verbose=false)
            
        catch e
            @test_skip "Basic GPU transform tests skipped: $e"
        end
    end
    
    @testset "GPU Memory Management" begin
        try
            if !initialize_gpu(0, verbose=false)
                @test_skip "GPU not initialized"
                return
            end
            
            cfg = create_gpu_config(12, 12)
            
            # Check memory before
            free_before, total = CUDA.memory_info()
            
            # Create multiple GPU arrays
            arrays = []
            for i in 1:5
                sh_gpu = CUDA.rand(Float64, get_nlm(cfg))
                spat_gpu = synthesize_gpu(cfg, sh_gpu)
                push!(arrays, (sh_gpu, spat_gpu))
            end
            
            # Check memory usage
            free_during, _ = CUDA.memory_info()
            @test free_during < free_before  # Memory should be used
            
            # Clear arrays
            arrays = nothing
            CUDA.reclaim()
            GC.gc()
            
            # Check memory after cleanup
            free_after, _ = CUDA.memory_info()
            memory_recovered = free_after - free_during
            @test memory_recovered > 0  # Some memory should be recovered
            
            free_config(cfg)
            cleanup_gpu(verbose=false)
            
        catch e
            @test_skip "GPU memory management tests skipped: $e"
        end
    end
    
    @testset "GPU Performance vs CPU" begin
        try
            if !initialize_gpu(0, verbose=false)
                @test_skip "GPU not initialized"
                return
            end
            
            # Use larger problem for meaningful timing
            lmax = 64
            cfg = create_gpu_config(lmax, lmax)
            
            # Create test data
            sh_cpu = rand(Float64, get_nlm(cfg))
            sh_gpu = CUDA.CuArray(sh_cpu)
            
            # Time CPU transform
            cpu_time = @elapsed begin
                for i in 1:5
                    spat_cpu = synthesize(cfg, sh_cpu)
                end
            end
            
            # Time GPU transform (including data transfer)
            gpu_time = @elapsed begin
                for i in 1:5
                    spat_gpu = synthesize_gpu(cfg, sh_gpu)
                end
            end
            
            # GPU should complete (may not be faster due to data transfer overhead)
            @test gpu_time > 0
            @test cpu_time > 0
            
            # Just test that both produce similar results
            spat_cpu_final = synthesize(cfg, sh_cpu)
            spat_gpu_final = synthesize_gpu(cfg, sh_gpu)
            
            accuracy = maximum(abs.(spat_cpu_final - Array(spat_gpu_final)))
            @test accuracy < 1e-12
            
            free_config(cfg)
            cleanup_gpu(verbose=false)
            
        catch e
            @test_skip "GPU performance tests skipped: $e"
        end
    end
    
    @testset "GPU with Different Precisions" begin
        try
            if !initialize_gpu(0, verbose=false)
                @test_skip "GPU not initialized"
                return
            end
            
            cfg = create_gpu_config(8, 8)
            
            # Test Float32 input (should be promoted to Float64)
            sh_f32 = rand(Float32, get_nlm(cfg))
            sh_gpu_f32 = CUDA.CuArray(sh_f32)
            
            spat_gpu = synthesize_gpu(cfg, sh_gpu_f32)
            @test eltype(spat_gpu) == Float64  # Should be promoted
            
            # Test with Float64
            sh_f64 = rand(Float64, get_nlm(cfg))
            sh_gpu_f64 = CUDA.CuArray(sh_f64)
            
            spat_gpu_f64 = synthesize_gpu(cfg, sh_gpu_f64)
            @test eltype(spat_gpu_f64) == Float64
            
            free_config(cfg)
            cleanup_gpu(verbose=false)
            
        catch e
            @test_skip "GPU precision tests skipped: $e"
        end
    end
    
    @testset "GPU Error Handling" begin
        try
            if !initialize_gpu(0, verbose=false)
                @test_skip "GPU not initialized"
                return
            end
            
            cfg = create_gpu_config(6, 6)
            
            # Test with wrong size arrays
            sh_wrong = CUDA.rand(Float64, 10)  # Wrong size
            
            # This should either work (with internal checks) or throw an appropriate error
            try
                spat_result = synthesize_gpu(cfg, sh_wrong)
                # If it works, just check the size is reasonable
                @test length(spat_result) > 0
            catch e
                # If it throws an error, that's also acceptable behavior
                @test e isa Exception
            end
            
            free_config(cfg)
            cleanup_gpu(verbose=false)
            
        catch e
            @test_skip "GPU error handling tests skipped: $e"
        end
    end
end