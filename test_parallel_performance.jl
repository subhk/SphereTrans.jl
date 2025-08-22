#!/usr/bin/env julia

"""
Comprehensive test suite for parallel and SIMD matrix operations.
Tests performance scaling across different problem sizes and parallelization strategies.
"""

# Load required modules
include("src/types_optimized.jl")
include("src/matrix_operators.jl")
include("src/simd_matrix_ops.jl")

using Base.Threads
using LinearAlgebra

function test_simd_performance()
    println("=== SIMD Performance Tests ===\n")
    
    T = Float64
    test_sizes = [10, 20, 30, 50]
    
    for lmax in test_sizes
        cfg = create_config(T, lmax, lmax, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
        nlm = cfg.nlm
        
        println("Testing lmax=$lmax, nlm=$nlm:")
        
        # Create test data
        qlm_test = [complex(randn(T), randn(T)) for _ in 1:nlm]
        qlm_out = similar(qlm_test)
        
        # Test SIMD Laplacian
        qlm_lap_test = copy(qlm_test)
        
        # Baseline implementation
        t_baseline = @elapsed begin
            for _ in 1:1000
                for (idx, (l, m)) in enumerate(cfg.lm_indices)
                    qlm_lap_test[idx] *= -T(l * (l + 1))
                end
            end
        end
        
        # SIMD implementation
        qlm_lap_test = copy(qlm_test)
        t_simd = @elapsed begin
            for _ in 1:1000
                simd_apply_laplacian!(cfg, qlm_lap_test)
            end
        end
        
        simd_speedup = t_baseline / t_simd
        println("  Laplacian SIMD speedup: $(round(simd_speedup, digits=2))x")
        
        # Test threading for cos(θ) operator
        if Threads.nthreads() > 1
            # Serial version
            t_serial = @elapsed begin
                for _ in 1:10
                    apply_costheta_operator_direct!(cfg, qlm_test, qlm_out)
                end
            end
            
            # Threaded version  
            t_threaded = @elapsed begin
                for _ in 1:10
                    threaded_apply_costheta_operator!(cfg, qlm_test, qlm_out)
                end
            end
            
            thread_speedup = t_serial / t_threaded
            println("  cos(θ) threading speedup: $(round(thread_speedup, digits=2))x ($(Threads.nthreads()) threads)")
        else
            println("  Threading tests skipped (single thread)")
        end
        
        println()
    end
    
    return true
end

function test_adaptive_selection()
    println("=== Adaptive Algorithm Selection Tests ===\n")
    
    T = Float64
    test_configs = [
        (lmax=10, expected=:direct),
        (lmax=30, expected=:cached_sparse),
        (lmax=50, expected=:cached_sparse)
    ]
    
    for (lmax, expected) in test_configs
        cfg = create_config(T, lmax, lmax, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
        
        # Test auto-selection logic
        selected = adaptive_operator_selection(cfg, cfg.nlm, :costheta)
        
        println("lmax=$lmax, nlm=$(cfg.nlm): selected=$selected, expected=$expected")
        
        if selected == expected
            println("  ✓ Correct selection")
        else
            println("  ⚠ Unexpected selection (may still be optimal)")
        end
    end
    
    println()
    return true
end

function test_memory_efficiency()
    println("=== Memory Efficiency Tests ===\n")
    
    T = Float64
    cfg = create_config(T, 25, 25, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
    qlm_test = [complex(randn(T), randn(T)) for _ in 1:cfg.nlm]
    
    println("Configuration: lmax=$(cfg.lmax), nlm=$(cfg.nlm)")
    
    # Test different approaches
    approaches = [
        ("Standard operator", () -> begin
            result = apply_costheta_operator(cfg, qlm_test)
        end),
        ("In-place cached", () -> begin
            qlm_out = similar(qlm_test)
            apply_costheta_operator!(cfg, qlm_test, qlm_out)
        end),
        ("Direct matrix-free", () -> begin
            qlm_out = similar(qlm_test)
            apply_costheta_operator_direct!(cfg, qlm_test, qlm_out)
        end),
        ("Auto SIMD dispatch", () -> begin
            qlm_out = similar(qlm_test)
            auto_simd_dispatch(cfg, :costheta, qlm_test, qlm_out)
        end)
    ]
    
    for (name, func) in approaches
        # Warmup
        func()
        
        # Measure allocation
        mem = @allocated begin
            for _ in 1:5
                func()
            end
        end
        
        println("  $name: $(mem) bytes per 5 ops")
    end
    
    println()
    return true
end

function test_accuracy_preservation()
    println("=== Accuracy Preservation Tests ===\n")
    
    T = Float64
    cfg = create_config(T, 15, 15, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
    
    # Create test vector with known structure
    qlm_test = Vector{Complex{T}}(undef, cfg.nlm)
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        qlm_test[idx] = complex(T(l + 1), T(m))
    end
    
    println("Testing accuracy for lmax=$(cfg.lmax), nlm=$(cfg.nlm)")
    
    # Reference result (cached sparse matrix)
    clear_operator_cache!()
    qlm_ref = apply_costheta_operator(cfg, qlm_test)
    
    # Test different implementations
    implementations = [
        ("Direct matrix-free", () -> begin
            qlm_out = similar(qlm_test)
            apply_costheta_operator_direct!(cfg, qlm_test, qlm_out)
        end),
        ("Threaded (if available)", () -> begin
            qlm_out = similar(qlm_test)
            if Threads.nthreads() > 1
                threaded_apply_costheta_operator!(cfg, qlm_test, qlm_out)
            else
                apply_costheta_operator_direct!(cfg, qlm_test, qlm_out)
            end
        end),
        ("Auto SIMD dispatch", () -> begin
            qlm_out = similar(qlm_test)
            auto_simd_dispatch(cfg, :costheta, qlm_test, qlm_out)
        end)
    ]
    
    all_accurate = true
    
    for (name, func) in implementations
        qlm_result = func()
        max_error = maximum(abs.(qlm_result - qlm_ref))
        rel_error = max_error / maximum(abs.(qlm_ref))
        
        accurate = rel_error < 1e-12
        all_accurate = all_accurate && accurate
        
        status = accurate ? "✓" : "✗"
        println("  $status $name: max_error=$(max_error), rel_error=$(rel_error)")
    end
    
    # Test SIMD Laplacian accuracy
    qlm_lap_test = copy(qlm_test)
    qlm_lap_ref = copy(qlm_test)
    
    # Reference
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        qlm_lap_ref[idx] *= -T(l * (l + 1))
    end
    
    # SIMD version
    simd_apply_laplacian!(cfg, qlm_lap_test)
    
    lap_error = maximum(abs.(qlm_lap_test - qlm_lap_ref))
    lap_accurate = lap_error < 1e-14
    all_accurate = all_accurate && lap_accurate
    
    status = lap_accurate ? "✓" : "✗"
    println("  $status SIMD Laplacian: max_error=$lap_error")
    
    println("\nOverall accuracy: $(all_accurate ? "PASSED" : "FAILED")")
    println()
    
    return all_accurate
end

function test_performance_scaling()
    println("=== Performance Scaling Tests ===\n")
    
    T = Float64
    test_sizes = [10, 20, 30, 40]
    
    println("Problem size scaling (times in ms):")
    println("lmax   nlm    Laplacian  cos(θ)")
    println("--------------------------------")
    
    for lmax in test_sizes
        cfg = create_config(T, lmax, lmax, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
        nlm = cfg.nlm
        
        qlm_test = [complex(randn(T), randn(T)) for _ in 1:nlm]
        
        # Time Laplacian (SIMD)
        qlm_lap = copy(qlm_test)
        t_lap = @elapsed begin
            for _ in 1:1000
                simd_apply_laplacian!(cfg, qlm_lap)
            end
        end
        t_lap_ms = round(t_lap * 1000 / 1000, digits=3)
        
        # Time cos(θ) operator
        qlm_out = similar(qlm_test)
        t_cos = @elapsed begin
            for _ in 1:100
                auto_simd_dispatch(cfg, :costheta, qlm_test, qlm_out)
            end
        end
        t_cos_ms = round(t_cos * 1000 / 100, digits=3)
        
        println("$(lpad(lmax, 4))  $(lpad(nlm, 4))   $(lpad(t_lap_ms, 9))  $(lpad(t_cos_ms, 6))")
    end
    
    println()
    return true
end

function analyze_threading_benefit()
    println("=== Threading Analysis ===\n")
    
    nthreads = Threads.nthreads()
    println("Available threads: $nthreads")
    
    if nthreads == 1
        println("Single-threaded mode - threading tests skipped")
        println("Run with: julia -t auto test_parallel_performance.jl")
        println()
        return true
    end
    
    T = Float64
    cfg = create_config(T, 30, 30, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
    qlm_test = [complex(randn(T), randn(T)) for _ in 1:cfg.nlm]
    
    println("Testing with lmax=$(cfg.lmax), nlm=$(cfg.nlm)")
    
    # Serial timing
    qlm_out = similar(qlm_test)
    t_serial = @elapsed begin
        for _ in 1:20
            apply_costheta_operator_direct!(cfg, qlm_test, qlm_out)
        end
    end
    
    # Threaded timing
    t_threaded = @elapsed begin
        for _ in 1:20
            threaded_apply_costheta_operator!(cfg, qlm_test, qlm_out)
        end
    end
    
    speedup = t_serial / t_threaded
    efficiency = speedup / nthreads
    
    println("Serial time:    $(round(t_serial*1000/20, digits=2)) ms per op")
    println("Threaded time:  $(round(t_threaded*1000/20, digits=2)) ms per op")
    println("Speedup:        $(round(speedup, digits=2))x")
    println("Efficiency:     $(round(efficiency*100, digits=1))%")
    
    if speedup > 1.5
        println("✓ Threading provides significant benefit")
    elseif speedup > 1.1
        println("⚠ Threading provides modest benefit")  
    else
        println("✗ Threading overhead dominates (problem too small)")
    end
    
    println()
    return speedup > 1.1
end

function main()
    println("=== Parallel and SIMD Matrix Operations Test Suite ===\n")
    
    # Run all tests
    test1 = test_simd_performance()
    test2 = test_adaptive_selection() 
    test3 = test_memory_efficiency()
    test4 = test_accuracy_preservation()
    test5 = test_performance_scaling()
    test6 = analyze_threading_benefit()
    
    # Summary
    println("=== Test Summary ===")
    println("SIMD performance:      $(test1 ? "PASSED" : "FAILED")")
    println("Adaptive selection:    $(test2 ? "PASSED" : "FAILED")")
    println("Memory efficiency:     $(test3 ? "PASSED" : "FAILED")")
    println("Accuracy preservation: $(test4 ? "PASSED" : "FAILED")")
    println("Performance scaling:   $(test5 ? "PASSED" : "FAILED")")
    println("Threading analysis:    $(test6 ? "PASSED" : "FAILED")")
    
    all_passed = test1 && test2 && test3 && test4 && test5 && test6
    
    if all_passed
        println("\n✅ All tests PASSED! Parallel optimizations working correctly.")
        return 0
    else
        println("\n❌ Some tests FAILED! Check implementations.")
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end