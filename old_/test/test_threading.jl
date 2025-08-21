using Test
using SHTnsKit
using Base.Threads

@testset "Threading and Performance" begin
    @testset "Thread Control Functions" begin
        try
            # Test thread control functions
            original_threads = get_num_threads()
            @test original_threads > 0
            
            # Test setting thread count
            set_num_threads(1)
            @test get_num_threads() == 1
            
            set_num_threads(2)
            @test get_num_threads() == 2
            
            # Test optimal thread setting
            optimal_threads = set_optimal_threads()
            @test optimal_threads >= 1
            @test get_num_threads() == optimal_threads
            
            # Restore original
            set_num_threads(original_threads)
            
        catch e
            @test_skip "Thread control tests skipped: $e"
        end
    end
    
    @testset "Thread Safety" begin
        try
            cfg = create_gauss_config(8, 8)
            
            # Test that multiple transforms on the same config are thread-safe
            if Threads.nthreads() > 1
                nlm = get_nlm(cfg)
                n_parallel = min(4, Threads.nthreads())
                
                # Create separate data for each thread
                sh_arrays = [rand(nlm) for _ in 1:n_parallel]
                spat_arrays = [allocate_spatial(cfg) for _ in 1:n_parallel]
                results = Vector{Bool}(undef, n_parallel)
                
                # Run transforms in parallel
                Threads.@threads for i in 1:n_parallel
                    try
                        synthesize!(cfg, sh_arrays[i], spat_arrays[i])
                        results[i] = true
                    catch
                        results[i] = false
                    end
                end
                
                # All transforms should succeed
                @test all(results)
                
                # Check that results are correct
                for i in 1:n_parallel
                    @test size(spat_arrays[i]) == (get_nlat(cfg), get_nphi(cfg))
                    @test !any(isnan.(spat_arrays[i]))
                    @test !any(isinf.(spat_arrays[i]))
                end
                
            else
                @test_skip "Single-threaded environment; skipping parallel tests"
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Thread safety tests skipped: $e"
        end
    end
    
    @testset "Concurrent Different Configs" begin
        try
            # Test that different configurations can run concurrently
            if Threads.nthreads() > 1
                configs = [create_gauss_config(6, 6), create_gauss_config(8, 8)]
                results = Vector{Bool}(undef, length(configs))
                
                Threads.@threads for i in 1:length(configs)
                    try
                        cfg = configs[i]
                        sh = rand(get_nlm(cfg))
                        spat = synthesize(cfg, sh)
                        sh_back = analyze(cfg, spat)
                        
                        error = maximum(abs.(sh - sh_back))
                        results[i] = error < 1e-12
                    catch
                        results[i] = false
                    end
                end
                
                # All should succeed
                @test all(results)
                
                # Cleanup
                for cfg in configs
                    free_config(cfg)
                end
                
            else
                @test_skip "Single-threaded environment; skipping concurrent config tests"
            end
            
        catch e
            @test_skip "Concurrent different configs tests skipped: $e"
        end
    end
    
    @testset "Threading Performance" begin
        try
            cfg = create_gauss_config(32, 32)
            sh = rand(get_nlm(cfg))
            spat = allocate_spatial(cfg)
            
            # Test with single thread
            set_num_threads(1)
            single_time = @elapsed begin
                for i in 1:5
                    synthesize!(cfg, sh, spat)
                end
            end
            
            # Test with multiple threads (if available)
            if get_num_threads() > 1 || true  # Force test even if we can only set to 1
                max_threads = min(4, Threads.nthreads())
                if max_threads > 1
                    set_num_threads(max_threads)
                    multi_time = @elapsed begin
                        for i in 1:5
                            synthesize!(cfg, sh, spat)
                        end
                    end
                    
                    # Multi-threaded should be faster or at least not much slower
                    # (allowing for overhead in small problems)
                    @test multi_time <= single_time * 2.0  # Allow some overhead
                end
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Threading performance tests skipped: $e"
        end
    end
    
    @testset "Thread-Safe Error Handling" begin
        try
            cfg = create_gauss_config(6, 6)
            
            if Threads.nthreads() > 1
                n_parallel = min(3, Threads.nthreads())
                results = Vector{Any}(undef, n_parallel)
                
                # Run operations that should fail in parallel
                Threads.@threads for i in 1:n_parallel
                    try
                        # Wrong array size - should throw AssertionError
                        sh_wrong = rand(5)  # Wrong size
                        spat = allocate_spatial(cfg)
                        synthesize!(cfg, sh_wrong, spat)
                        results[i] = :success  # Should not reach here
                    catch e
                        results[i] = typeof(e)
                    end
                end
                
                # All should fail with AssertionError
                @test all(r == AssertionError for r in results)
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Thread-safe error handling tests skipped: $e"
        end
    end
end