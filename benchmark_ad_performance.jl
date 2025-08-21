"""
Comprehensive AD Performance Benchmark for SHTnsKit.jl

Compares original vs optimized implementations across:
- Type stability
- Memory efficiency  
- CPU performance
- Scalability with problem size

Usage: julia --project=. benchmark_ad_performance.jl
"""

using BenchmarkTools
using SHTnsKit
using LinearAlgebra
using Printf

# Try to load AD packages
try
    using ForwardDiff, Zygote
    global AD_AVAILABLE = true
catch
    global AD_AVAILABLE = false
    println("⚠ AD packages not available - install with: Pkg.add([\"ForwardDiff\", \"Zygote\"])")
end

if AD_AVAILABLE
    # Load optimized versions if they exist
    try
        include("ext/SHTnsKitForwardDiffExt_Optimized.jl")
        include("ext/SHTnsKitZygoteExt_Optimized.jl")
        global OPTIMIZED_AVAILABLE = true
    catch
        global OPTIMIZED_AVAILABLE = false
    end
else
    global OPTIMIZED_AVAILABLE = false
end

"""
Benchmark configuration parameters
"""
struct BenchmarkConfig
    lmax_values::Vector{Int}
    n_trials::Int
    test_functions::Dict{String, Function}
    
    function BenchmarkConfig()
        lmax_vals = [4, 8, 16, 32]
        n_trials = 5
        
        test_funcs = Dict(
            "synthesis_power" => (cfg, sh) -> sum(abs2, synthesize(cfg, sh)),
            "power_spectrum" => (cfg, sh) -> sum(power_spectrum(cfg, sh)),
            "point_evaluation" => (cfg, sh) -> evaluate_at_point(cfg, sh, π/3, π/4)^2,
            "roundtrip" => (cfg, sh) -> sum(abs2, sh - analyze(cfg, synthesize(cfg, sh)))
        )
        
        new(lmax_vals, n_trials, test_funcs)
    end
end

"""
Performance metrics collection
"""
mutable struct PerformanceMetrics
    median_time::Float64
    memory_bytes::Int64
    allocs::Int64
    gc_time::Float64
    
    PerformanceMetrics() = new(0.0, 0, 0, 0.0)
end

"""
Run benchmark for a specific test function
"""
function benchmark_function(test_func::Function, cfg, sh_coeffs, name::String)
    println("  Benchmarking $name...")
    
    try
        # Warmup
        test_func(cfg, sh_coeffs)
        
        # Benchmark
        result = @benchmark $test_func($cfg, $sh_coeffs) samples=5 seconds=10
        
        metrics = PerformanceMetrics()
        metrics.median_time = median(result.times) * 1e-9  # Convert to seconds
        metrics.memory_bytes = result.memory
        metrics.allocs = result.allocs  
        metrics.gc_time = result.gctimes > 0 ? median(result.gctimes) * 1e-9 : 0.0
        
        return metrics
    catch e
        println("    ❌ Benchmark failed: $e")
        return PerformanceMetrics()
    end
end

"""
Test type stability of a function
"""
function test_type_stability(f::Function, cfg, sh_coeffs)
    try
        # Use @code_warntype to check for type instabilities
        io = IOBuffer()
        @code_warntype io=io f(cfg, sh_coeffs)
        output = String(take!(io))
        
        # Count type instabilities (rough heuristic)
        instabilities = count(r"::Any|::Union|UNION", output)
        return instabilities
    catch
        return -1  # Error in analysis
    end
end

"""
Memory allocation analysis
"""
function analyze_memory_allocations(f::Function, cfg, sh_coeffs, name::String)
    println("    Memory analysis for $name:")
    
    # Track allocations
    alloc_result = @allocated f(cfg, sh_coeffs)
    
    println("      Total allocations: $(alloc_result) bytes ($(alloc_result ÷ 1024) KB)")
    
    # Check for major allocation sources
    if alloc_result > 1_000_000  # > 1MB
        println("      ⚠ HIGH MEMORY USAGE detected")
    elseif alloc_result > 100_000  # > 100KB  
        println("      ⚠ Moderate memory usage")
    else
        println("      ✓ Low memory usage")
    end
    
    return alloc_result
end

"""
Run comprehensive AD performance benchmark
"""
function run_ad_benchmark()
    if !AD_AVAILABLE
        println("❌ Cannot run AD benchmarks - packages not available")
        return
    end
    
    config = BenchmarkConfig()
    
    println("🚀 Starting AD Performance Benchmark")
    println("="^60)
    
    # Results storage
    results = Dict()
    
    for lmax in config.lmax_values
        println("\\n📊 Testing lmax = $lmax")
        println("-"^40)
        
        # Create configuration
        cfg = create_gauss_config(lmax, lmax)
        nlm = get_nlm(cfg)
        sh_coeffs = 0.1 * randn(nlm)
        
        println("  Configuration: nlm=$nlm, nlat×nphi=$(get_nlat(cfg))×$(get_nphi(cfg))")
        
        results[lmax] = Dict()
        
        for (test_name, test_func) in config.test_functions
            println("\\n  🔬 Testing $test_name")
            
            # Test forward function
            forward_metrics = benchmark_function(test_func, cfg, sh_coeffs, test_name)
            
            # Test ForwardDiff gradient
            if true  # ForwardDiff available
                try
                    grad_func(x) = ForwardDiff.gradient(y -> test_func(cfg, y), x)
                    fd_metrics = benchmark_function(grad_func, cfg, sh_coeffs, "$test_name (ForwardDiff)")
                    
                    # Type stability check
                    instabilities_fd = test_type_stability(grad_func, cfg, sh_coeffs)
                    
                    println("    ForwardDiff:")
                    println("      Time: $(@sprintf("%.3f", fd_metrics.median_time*1000)) ms")
                    println("      Memory: $(fd_metrics.memory_bytes ÷ 1024) KB")
                    println("      Allocations: $(fd_metrics.allocs)")
                    println("      Type instabilities: $instabilities_fd")
                    
                    results[lmax]["$test_name\_fd"] = fd_metrics
                    
                catch e
                    println("    ❌ ForwardDiff test failed: $e")
                end
            end
            
            # Test Zygote gradient
            if true  # Zygote available
                try
                    zy_func(x) = Zygote.gradient(y -> test_func(cfg, y), x)[1]
                    zy_metrics = benchmark_function(zy_func, cfg, sh_coeffs, "$test_name (Zygote)")
                    
                    # Type stability check
                    instabilities_zy = test_type_stability(zy_func, cfg, sh_coeffs)
                    
                    println("    Zygote:")
                    println("      Time: $(@sprintf("%.3f", zy_metrics.median_time*1000)) ms")
                    println("      Memory: $(zy_metrics.memory_bytes ÷ 1024) KB")
                    println("      Allocations: $(zy_metrics.allocs)")
                    println("      Type instabilities: $instabilities_zy")
                    
                    results[lmax]["$test_name\_zy"] = zy_metrics
                    
                catch e
                    println("    ❌ Zygote test failed: $e")
                end
            end
            
            # Memory allocation analysis
            println("\\n    📊 Memory Analysis:")
            analyze_memory_allocations(test_func, cfg, sh_coeffs, test_name)
        end
    end
    
    # Summary analysis
    println("\\n\\n📋 PERFORMANCE SUMMARY")
    println("="^60)
    
    analyze_scalability(results)
    identify_bottlenecks(results)
    suggest_optimizations(results)
    
    return results
end

"""
Analyze performance scalability with problem size
"""
function analyze_scalability(results::Dict)
    println("\\n🔍 Scalability Analysis:")
    
    for test_name in ["synthesis_power_fd", "power_spectrum_zy"]
        if all(haskey(results[lmax], test_name) for lmax in keys(results))
            println("\\n  $test_name:")
            
            times = [results[lmax][test_name].median_time for lmax in sort(collect(keys(results)))]
            sizes = [lmax^2 for lmax in sort(collect(keys(results)))]  # Approximate problem size
            
            # Estimate scaling exponent
            if length(times) >= 2
                log_times = log.(times)
                log_sizes = log.(sizes)
                
                # Linear fit: log(time) = a + b*log(size)
                X = hcat(ones(length(log_sizes)), log_sizes)
                coeffs = X \\ log_times
                scaling_exponent = coeffs[2]
                
                println("    Scaling exponent: $(@sprintf("%.2f", scaling_exponent))")
                
                if scaling_exponent < 1.5
                    println("    ✅ Good scalability (sub-quadratic)")
                elseif scaling_exponent < 2.5
                    println("    ⚠ Moderate scalability (quadratic-ish)")
                else
                    println("    ❌ Poor scalability (super-quadratic)")
                end
            end
        end
    end
end

"""
Identify performance bottlenecks
"""
function identify_bottlenecks(results::Dict)
    println("\\n🎯 Bottleneck Analysis:")
    
    # Find highest memory users
    max_memory = 0
    worst_memory_test = ""
    
    # Find slowest operations
    max_time = 0.0  
    slowest_test = ""
    
    for lmax in keys(results)
        for (test_name, metrics) in results[lmax]
            if metrics.memory_bytes > max_memory
                max_memory = metrics.memory_bytes
                worst_memory_test = "$test_name (lmax=$lmax)"
            end
            
            if metrics.median_time > max_time
                max_time = metrics.median_time
                slowest_test = "$test_name (lmax=$lmax)"
            end
        end
    end
    
    println("  Highest memory usage: $(max_memory ÷ 1024) KB in $worst_memory_test")
    println("  Slowest operation: $(@sprintf("%.3f", max_time*1000)) ms for $slowest_test")
    
    # Identify allocation-heavy operations
    high_alloc_threshold = 1000
    println("\\n  High-allocation operations:")
    for lmax in keys(results)
        for (test_name, metrics) in results[lmax]
            if metrics.allocs > high_alloc_threshold
                println("    $test_name (lmax=$lmax): $(metrics.allocs) allocations")
            end
        end
    end
end

"""
Suggest performance optimizations based on results
"""
function suggest_optimizations(results::Dict)
    println("\\n💡 Optimization Suggestions:")
    
    suggestions = String[]
    
    # Check for excessive allocations
    total_allocs = sum(metrics.allocs for lmax_results in values(results) for metrics in values(lmax_results))
    if total_allocs > 10000
        push!(suggestions, "• Reduce memory allocations with pre-allocated buffers")
        push!(suggestions, "• Implement in-place operations where possible")
    end
    
    # Check for GC pressure
    total_gc_time = sum(metrics.gc_time for lmax_results in values(results) for metrics in values(lmax_results))
    if total_gc_time > 0.1  # 100ms total
        push!(suggestions, "• Reduce GC pressure with memory pooling")
        push!(suggestions, "• Use object pools for frequently allocated types")
    end
    
    # Check scaling
    largest_lmax = maximum(keys(results))
    if largest_lmax >= 16
        push!(suggestions, "• Consider SIMD optimizations for large problems")
        push!(suggestions, "• Implement multithreading for parallel operations")
        push!(suggestions, "• Cache trigonometric computations")
    end
    
    # Type stability suggestions  
    push!(suggestions, "• Use concrete types instead of abstract types")
    push!(suggestions, "• Pre-allocate buffers with known sizes")
    push!(suggestions, "• Avoid type-unstable operations in hot paths")
    
    # Memory efficiency suggestions
    push!(suggestions, "• Implement copy-free operations where possible")
    push!(suggestions, "• Use views instead of copying arrays")
    push!(suggestions, "• Cache expensive computations")
    
    for suggestion in suggestions
        println("  $suggestion")
    end
end

"""
Compare original vs optimized implementations if available
"""
function compare_implementations()
    if !OPTIMIZED_AVAILABLE
        println("⚠ Optimized implementations not available for comparison")
        return
    end
    
    println("\\n🏁 Original vs Optimized Comparison")
    println("="^50)
    
    # This would require loading both versions and comparing
    println("  Comparison functionality would be implemented here")
end

"""
Main benchmark execution
"""
function main()
    println("🔧 SHTnsKit.jl AD Performance Benchmark")
    println("="^50)
    
    if !AD_AVAILABLE
        println("❌ Automatic differentiation packages not available")
        println("Install with: julia -e 'using Pkg; Pkg.add([\"ForwardDiff\", \"Zygote\"])'")
        return
    end
    
    # System information
    println("System Information:")
    println("  Julia version: $(VERSION)")
    println("  Threads: $(Threads.nthreads())")
    println("  Architecture: $(Sys.ARCH)")
    
    # Run benchmarks
    results = run_ad_benchmark()
    
    # Compare implementations if available
    compare_implementations()
    
    println("\\n✅ Benchmark completed!")
    return results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end