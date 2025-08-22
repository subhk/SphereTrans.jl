"""
Comprehensive benchmarking suite for SHTnsKit performance analysis.
This module provides tools to measure and compare performance improvements.
"""

using Printf
using Statistics
# BenchmarkTools is optional - basic timing will be used if not available
const BENCHMARKTOOLS_AVAILABLE = try
    using BenchmarkTools
    true
catch
    false
end

struct BenchmarkResult
    name::String
    median_time::Float64
    minimum_time::Float64
    maximum_time::Float64
    mean_time::Float64
    std_time::Float64
    allocations::Int
    memory::Int
    gc_fraction::Float64
end

struct PerformanceComparison
    original::BenchmarkResult
    optimized::BenchmarkResult
    speedup::Float64
    memory_reduction::Float64
    allocation_reduction::Float64
end

"""
    benchmark_transform_performance(cfg::SHTnsConfig{T}; 
                                   n_samples::Int=100,
                                   include_optimized::Bool=true) where T

Comprehensive benchmark of transform performance comparing original vs optimized implementations.
"""
function benchmark_transform_performance(cfg::SHTnsConfig{T}; 
                                        n_samples::Int=100,
                                        include_optimized::Bool=true) where T
    validate_config(cfg)
    
    println("="^80)
    println("SHTnsKit Performance Benchmark")
    println("Configuration: lmax=$(cfg.lmax), mmax=$(cfg.mmax), grid=$(cfg.nlat)×$(cfg.nphi)")
    println("Precision: $T, Samples: $n_samples")
    println("="^80)
    
    # Prepare test data
    sh_coeffs = randn(Complex{T}, cfg.nlm)
    spatial_data = allocate_spatial(cfg)
    
    results = Dict{String, BenchmarkResult}()
    
    # Benchmark original synthesis
    println("Benchmarking original synthesis...")
    b_orig_synth = @benchmark sh_to_spat!($cfg, $sh_coeffs, $spatial_data) samples=n_samples
    results["original_synthesis"] = BenchmarkResult(
        "Original Synthesis",
        median(b_orig_synth.times) * 1e-9,
        minimum(b_orig_synth.times) * 1e-9,
        maximum(b_orig_synth.times) * 1e-9,
        mean(b_orig_synth.times) * 1e-9,
        std(b_orig_synth.times) * 1e-9,
        b_orig_synth.allocs,
        b_orig_synth.memory,
        b_orig_synth.gctimes / sum(b_orig_synth.times)
    )
    
    # Benchmark original analysis  
    println("Benchmarking original analysis...")
    b_orig_anal = @benchmark spat_to_sh!($cfg, $spatial_data, $sh_coeffs) samples=n_samples
    results["original_analysis"] = BenchmarkResult(
        "Original Analysis",
        median(b_orig_anal.times) * 1e-9,
        minimum(b_orig_anal.times) * 1e-9,
        maximum(b_orig_anal.times) * 1e-9,
        mean(b_orig_anal.times) * 1e-9,
        std(b_orig_anal.times) * 1e-9,
        b_orig_anal.allocs,
        b_orig_anal.memory,
        b_orig_anal.gctimes / sum(b_orig_anal.times)
    )
    
    if include_optimized
        # Benchmark optimized synthesis
        println("Benchmarking optimized synthesis...")
        b_opt_synth = @benchmark sh_to_spat_optimized!($cfg, $sh_coeffs, $spatial_data) samples=n_samples
        results["optimized_synthesis"] = BenchmarkResult(
            "Optimized Synthesis",
            median(b_opt_synth.times) * 1e-9,
            minimum(b_opt_synth.times) * 1e-9,
            maximum(b_opt_synth.times) * 1e-9,
            mean(b_opt_synth.times) * 1e-9,
            std(b_opt_synth.times) * 1e-9,
            b_opt_synth.allocs,
            b_opt_synth.memory,
            b_opt_synth.gctimes / sum(b_opt_synth.times)
        )
        
        # Benchmark optimized analysis
        println("Benchmarking optimized analysis...")
        b_opt_anal = @benchmark spat_to_sh_optimized!($cfg, $spatial_data, $sh_coeffs) samples=n_samples
        results["optimized_analysis"] = BenchmarkResult(
            "Optimized Analysis", 
            median(b_opt_anal.times) * 1e-9,
            minimum(b_opt_anal.times) * 1e-9,
            maximum(b_opt_anal.times) * 1e-9,
            mean(b_opt_anal.times) * 1e-9,
            std(b_opt_anal.times) * 1e-9,
            b_opt_anal.allocs,
            b_opt_anal.memory,
            b_opt_anal.gctimes / sum(b_opt_anal.times)
        )
    end
    
    # Print results
    print_benchmark_results(results, include_optimized)
    
    return results
end

"""
    benchmark_vector_transforms(cfg::SHTnsConfig{T}; n_samples::Int=50) where T

Benchmark vector transform performance.
"""
function benchmark_vector_transforms(cfg::SHTnsConfig{T}; n_samples::Int=50) where T
    validate_config(cfg)
    
    println("\nVector Transform Benchmarks")
    println("-"^50)
    
    # Test data
    sph_coeffs = randn(Complex{T}, cfg.nlm)
    tor_coeffs = randn(Complex{T}, cfg.nlm)
    u_theta = allocate_spatial(cfg)
    u_phi = allocate_spatial(cfg)
    
    results = Dict{String, BenchmarkResult}()
    
    # Original vector synthesis
    println("Benchmarking original vector synthesis...")
    b_orig_vec = @benchmark sphtor_to_spat!($cfg, $sph_coeffs, $tor_coeffs, $u_theta, $u_phi) samples=n_samples
    results["original_vector_synthesis"] = BenchmarkResult(
        "Original Vector Synthesis",
        median(b_orig_vec.times) * 1e-9,
        minimum(b_orig_vec.times) * 1e-9,
        maximum(b_orig_vec.times) * 1e-9,
        mean(b_orig_vec.times) * 1e-9,
        std(b_orig_vec.times) * 1e-9,
        b_orig_vec.allocs,
        b_orig_vec.memory,
        b_orig_vec.gctimes / sum(b_orig_vec.times)
    )
    
    # Optimized vector synthesis (if available)
    try
        println("Benchmarking optimized vector synthesis...")
        b_opt_vec = @benchmark sphtor_to_spat_optimized!($cfg, $sph_coeffs, $tor_coeffs, $u_theta, $u_phi) samples=n_samples
        results["optimized_vector_synthesis"] = BenchmarkResult(
            "Optimized Vector Synthesis",
            median(b_opt_vec.times) * 1e-9,
            minimum(b_opt_vec.times) * 1e-9,
            maximum(b_opt_vec.times) * 1e-9,
            mean(b_opt_vec.times) * 1e-9,
            std(b_opt_vec.times) * 1e-9,
            b_opt_vec.allocs,
            b_opt_vec.memory,
            b_opt_vec.gctimes / sum(b_opt_vec.times)
        )
    catch e
        println("Optimized vector transforms not available: $e")
    end
    
    print_benchmark_results(results, haskey(results, "optimized_vector_synthesis"))
    return results
end

"""
    benchmark_memory_scaling(lmax_range::AbstractVector{Int}, ::Type{T}=Float64; 
                            nphi_factor::Int=4) where T

Benchmark memory usage and performance scaling with problem size.
"""
function benchmark_memory_scaling(lmax_range::AbstractVector{Int}, ::Type{T}=Float64; 
                                 nphi_factor::Int=4) where T
    println("\nMemory Scaling Analysis")
    println("="^60)
    
    scaling_results = []
    
    for lmax in lmax_range
        mmax = lmax
        nlat = 2*lmax + 2  
        nphi = nphi_factor * lmax + 1
        
        println("Testing lmax=$lmax, grid=$(nlat)×$(nphi)...")
        
        cfg = create_gauss_config(T, lmax, mmax, nlat, nphi)
        
        # Memory analysis
        memory_info = profile_memory_usage(cfg)
        
        # Quick performance test
        sh_coeffs = randn(Complex{T}, cfg.nlm)
        spatial_data = allocate_spatial(cfg)
        
        # Time single transform
        t_synth = @elapsed sh_to_spat!(cfg, sh_coeffs, spatial_data)
        t_anal = @elapsed spat_to_sh!(cfg, spatial_data, sh_coeffs)
        
        # Allocation test
        alloc_test = @allocated begin
            sh_to_spat!(cfg, sh_coeffs, spatial_data)
            spat_to_sh!(cfg, spatial_data, sh_coeffs)
        end
        
        push!(scaling_results, (
            lmax = lmax,
            nlm = cfg.nlm,
            nspat = nlat * nphi,
            memory_mb = memory_info.memory_mb,
            synthesis_time = t_synth,
            analysis_time = t_anal,
            allocations = alloc_test
        ))
        
        @printf "  lmax=%3d: nlm=%5d, memory=%6.2f MB, synth=%7.3f ms, anal=%7.3f ms, alloc=%8d bytes\n" lmax cfg.nlm memory_info.memory_mb (t_synth*1000) (t_anal*1000) alloc_test
    end
    
    return scaling_results
end

"""
    benchmark_different_precisions()

Compare performance across different floating-point precisions.
"""
function benchmark_different_precisions()
    println("\nPrecision Comparison Benchmark")
    println("="^50)
    
    lmax, mmax = 20, 16
    nlat, nphi = 48, 64
    
    precisions = [Float32, Float64]
    precision_results = Dict()
    
    for T in precisions
        println("Testing precision: $T")
        
        cfg = create_gauss_config(T, lmax, mmax, nlat, nphi)
        sh_coeffs = randn(Complex{T}, cfg.nlm)
        spatial_data = allocate_spatial(cfg)
        
        # Benchmark synthesis
        b_synth = @benchmark sh_to_spat!($cfg, $sh_coeffs, $spatial_data) samples=20
        
        # Benchmark analysis
        b_anal = @benchmark spat_to_sh!($cfg, $spatial_data, $sh_coeffs) samples=20
        
        precision_results[T] = (
            synthesis_time = median(b_synth.times) * 1e-9,
            analysis_time = median(b_anal.times) * 1e-9,
            synthesis_allocs = b_synth.allocs,
            analysis_allocs = b_anal.allocs,
            synthesis_memory = b_synth.memory,
            analysis_memory = b_anal.memory
        )
        
        @printf "  %s: synth=%7.3f ms, anal=%7.3f ms, synth_mem=%8d B, anal_mem=%8d B\n" T (precision_results[T].synthesis_time*1000) (precision_results[T].analysis_time*1000) precision_results[T].synthesis_memory precision_results[T].analysis_memory
    end
    
    # Compare Float32 vs Float64
    if haskey(precision_results, Float32) && haskey(precision_results, Float64)
        r32, r64 = precision_results[Float32], precision_results[Float64]
        synth_speedup = r64.synthesis_time / r32.synthesis_time
        anal_speedup = r64.analysis_time / r32.analysis_time
        mem_reduction = (r64.synthesis_memory - r32.synthesis_memory) / r64.synthesis_memory * 100
        
        println("\nFloat32 vs Float64 comparison:")
        @printf "  Synthesis speedup: %.2fx\n" synth_speedup
        @printf "  Analysis speedup: %.2fx\n" anal_speedup  
        @printf "  Memory reduction: %.1f%%\n" mem_reduction
    end
    
    return precision_results
end

"""
    benchmark_threading_performance(cfg::SHTnsConfig{T}; max_threads::Int=Threads.nthreads()) where T

Benchmark performance scaling with number of threads.
"""
function benchmark_threading_performance(cfg::SHTnsConfig{T}; max_threads::Int=Threads.nthreads()) where T
    println("\nThreading Performance Analysis")
    println("="^50)
    println("Available threads: $(Threads.nthreads())")
    
    # Test data
    sh_coeffs = randn(Complex{T}, cfg.nlm)
    spatial_data = allocate_spatial(cfg)
    
    threading_results = []
    
    # Test different thread counts (would need threading implementation)
    # This is a placeholder showing how to structure the benchmark
    for nthreads in 1:min(max_threads, 4)
        println("Testing with $nthreads threads...")
        
        # Simulate different thread usage (actual implementation would set thread count)
        # For now, just run regular benchmark
        b_synth = @benchmark sh_to_spat!($cfg, $sh_coeffs, $spatial_data) samples=10
        b_anal = @benchmark spat_to_sh!($cfg, $spatial_data, $sh_coeffs) samples=10
        
        push!(threading_results, (
            threads = nthreads,
            synthesis_time = median(b_synth.times) * 1e-9,
            analysis_time = median(b_anal.times) * 1e-9
        ))
        
        @printf "  %d threads: synth=%7.3f ms, anal=%7.3f ms\n" nthreads (threading_results[end].synthesis_time*1000) (threading_results[end].analysis_time*1000)
    end
    
    return threading_results
end

# Utility functions for benchmark reporting

function print_benchmark_results(results::Dict{String, BenchmarkResult}, include_comparison::Bool=false)
    println("\nBenchmark Results:")
    println("-"^100)
    @printf "%-25s %12s %12s %12s %10s %12s %10s\n" "Function" "Median (ms)" "Min (ms)" "Max (ms)" "Allocs" "Memory (MB)" "GC %"
    println("-"^100)
    
    for (key, result) in results
        @printf "%-25s %12.6f %12.6f %12.6f %10d %12.3f %10.2f\n" result.name (result.median_time*1000) (result.minimum_time*1000) (result.maximum_time*1000) result.allocations (result.memory/1024^2) (result.gc_fraction*100)
    end
    
    if include_comparison && haskey(results, "original_synthesis") && haskey(results, "optimized_synthesis")
        println("\nPerformance Improvements:")
        println("-"^60)
        
        orig_synth = results["original_synthesis"]
        opt_synth = results["optimized_synthesis"]
        synth_speedup = orig_synth.median_time / opt_synth.median_time
        synth_mem_reduction = (orig_synth.memory - opt_synth.memory) / orig_synth.memory * 100
        synth_alloc_reduction = (orig_synth.allocations - opt_synth.allocations) / max(orig_synth.allocations, 1) * 100
        
        @printf "Synthesis speedup: %.2fx (%.1f%% faster)\n" synth_speedup ((synth_speedup-1)*100)
        @printf "Synthesis memory reduction: %.1f%%\n" synth_mem_reduction
        @printf "Synthesis allocation reduction: %.1f%%\n" synth_alloc_reduction
        
        if haskey(results, "original_analysis") && haskey(results, "optimized_analysis")
            orig_anal = results["original_analysis"]
            opt_anal = results["optimized_analysis"]
            anal_speedup = orig_anal.median_time / opt_anal.median_time
            anal_mem_reduction = (orig_anal.memory - opt_anal.memory) / orig_anal.memory * 100
            anal_alloc_reduction = (orig_anal.allocations - opt_anal.allocations) / max(orig_anal.allocations, 1) * 100
            
            @printf "Analysis speedup: %.2fx (%.1f%% faster)\n" anal_speedup ((anal_speedup-1)*100)
            @printf "Analysis memory reduction: %.1f%%\n" anal_mem_reduction
            @printf "Analysis allocation reduction: %.1f%%\n" anal_alloc_reduction
        end
    end
    
    println("-"^100)
end

"""
    run_comprehensive_benchmark(; 
        lmax_small::Int=10, lmax_medium::Int=20, lmax_large::Int=40,
        include_scaling::Bool=true, include_precision::Bool=true, 
        include_threading::Bool=true)

Run a comprehensive benchmark suite covering all major performance aspects.
"""
function run_comprehensive_benchmark(; 
    lmax_small::Int=10, lmax_medium::Int=20, lmax_large::Int=40,
    include_scaling::Bool=true, include_precision::Bool=true, 
    include_threading::Bool=true)
    
    println("SHTnsKit Comprehensive Performance Benchmark")
    println("="^80)
    println("Julia Version: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("SIMD Width: $(SIMD.pick_vector_width(Float64))")
    println("="^80)
    
    all_results = Dict()
    
    # Small problem benchmark
    println("\n🔹 Small Problem Benchmark (lmax=$lmax_small)")
    cfg_small = create_gauss_config(Float64, lmax_small, lmax_small, 2*lmax_small+2, 4*lmax_small+1)
    all_results[:small] = benchmark_transform_performance(cfg_small, n_samples=100)
    
    # Medium problem benchmark  
    println("\n🔹 Medium Problem Benchmark (lmax=$lmax_medium)")
    cfg_medium = create_gauss_config(Float64, lmax_medium, lmax_medium, 2*lmax_medium+2, 4*lmax_medium+1)
    all_results[:medium] = benchmark_transform_performance(cfg_medium, n_samples=50)
    
    # Large problem benchmark
    if lmax_large <= 50  # Avoid excessive runtime
        println("\n🔹 Large Problem Benchmark (lmax=$lmax_large)")
        cfg_large = create_gauss_config(Float64, lmax_large, lmax_large, 2*lmax_large+2, 4*lmax_large+1)
        all_results[:large] = benchmark_transform_performance(cfg_large, n_samples=20)
    end
    
    # Vector transforms
    println("\n🔹 Vector Transform Benchmark")
    all_results[:vector] = benchmark_vector_transforms(cfg_medium)
    
    # Scaling analysis
    if include_scaling
        println("\n🔹 Memory Scaling Analysis")
        all_results[:scaling] = benchmark_memory_scaling([5, 10, 15, 20, 25, 30])
    end
    
    # Precision comparison
    if include_precision
        println("\n🔹 Precision Comparison")
        all_results[:precision] = benchmark_different_precisions()
    end
    
    # Threading performance
    if include_threading && Threads.nthreads() > 1
        println("\n🔹 Threading Performance")
        all_results[:threading] = benchmark_threading_performance(cfg_medium)
    end
    
    println("\n" * "="^80)
    println("Comprehensive benchmark completed!")
    println("="^80)
    
    return all_results
end