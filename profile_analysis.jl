#!/usr/bin/env julia

"""
Comprehensive profiling and type stability analysis for SHTnsKit.jl
This script identifies performance bottlenecks, memory allocations, and type instabilities.
"""

using Pkg
Pkg.activate(".")

using SHTnsKit
using BenchmarkTools
using Profile, ProfileView
using InteractiveUtils: @code_warntype, @code_llvm
using StaticAnalysis: analyze_method_instances
using Plots

println("=== SHTnsKit.jl Performance Analysis ===")
println("Julia version: ", VERSION)
println("Number of threads: ", Threads.nthreads())
println()

# Helper function to analyze type stability
function analyze_type_stability(func, args...)
    println("Type stability analysis for: ", func)
    println("=" ^ 50)
    
    try
        @code_warntype func(args...)
    catch e
        println("Error in type analysis: ", e)
    end
    println()
end

# Helper function to measure allocations
function measure_allocations(name::String, func, args...)
    println("Allocation analysis for: ", name)
    println("=" ^ 50)
    
    result = @benchmark $func($(args)...)
    
    println("Time: ", BenchmarkTools.prettytime(median(result.times)))
    println("Memory: ", BenchmarkTools.prettymemory(median(result.memory)))
    println("Allocations: ", median(result.allocs))
    println("GC time: ", BenchmarkTools.prettytime(median(result.gctimes)))
    
    # Profile for more detailed allocation info
    Profile.clear()
    Profile.clear_malloc_data()
    
    # Warmup
    func(args...)
    
    # Profile with allocations
    @profile for _ in 1:100
        func(args...)
    end
    
    println("Profiling results:")
    Profile.print(maxdepth=15, sortedby=:count)
    println()
    
    return result
end

# Create test configuration
println("Setting up test configuration...")
lmax, mmax = 63, 63
cfg = create_gauss_config(lmax, mmax; T=Float64)
println("Configuration: lmax=$lmax, mmax=$mmax")
println("Grid size: $(cfg.nlat) × $(cfg.nphi)")
println("Number of coefficients: $(cfg.nlm)")
println()

# Create test data
println("Creating test data...")
spatial_data = randn(Float64, cfg.nlat, cfg.nphi)
sh_coeffs = allocate_spectral(cfg)

# 1. Type stability analysis
println("\n1. TYPE STABILITY ANALYSIS")
println("=" ^ 60)

analyze_type_stability(spat_to_sh!, cfg, spatial_data, sh_coeffs)
analyze_type_stability(sh_to_spat!, cfg, sh_coeffs, spatial_data)

# 2. Memory allocation analysis
println("\n2. MEMORY ALLOCATION ANALYSIS")  
println("=" ^ 60)

results = Dict()

# Analysis transform
results[:analysis] = measure_allocations("spat_to_sh!", spat_to_sh!, cfg, spatial_data, sh_coeffs)

# Synthesis transform  
fill!(sh_coeffs, 0.0)
for i in 1:min(10, cfg.nlm)
    sh_coeffs[i] = randn()
end
results[:synthesis] = measure_allocations("sh_to_spat!", sh_to_spat!, cfg, sh_coeffs, spatial_data)

# FFT operations
println("\n3. FFT PERFORMANCE ANALYSIS")
println("=" ^ 60)

spatial_row = spatial_data[1, :]
fourier_coeffs = zeros(Complex{Float64}, cfg.nphi ÷ 2 + 1)

results[:fft_forward] = measure_allocations("azimuthal_fft_forward!", 
    SHTnsKit.azimuthal_fft_forward!, cfg, spatial_row, fourier_coeffs)
    
results[:fft_backward] = measure_allocations("azimuthal_fft_backward!", 
    SHTnsKit.azimuthal_fft_backward!, cfg, fourier_coeffs, spatial_row)

# 4. Legendre polynomial computation
println("\n4. LEGENDRE POLYNOMIAL ANALYSIS")
println("=" ^ 60)

cost = 0.5
results[:legendre] = measure_allocations("compute_associated_legendre",
    SHTnsKit.compute_associated_legendre, cfg.lmax, cost, cfg.norm)

# 5. Grid setup analysis  
println("\n5. GRID SETUP ANALYSIS")
println("=" ^ 60)

test_cfg = create_config(lmax, mmax; T=Float64)
results[:grid_setup] = measure_allocations("set_grid!", set_grid!, test_cfg, cfg.nlat, cfg.nphi)

# 6. Utility functions analysis
println("\n6. UTILITY FUNCTIONS ANALYSIS")
println("=" ^ 60)

results[:lmidx] = measure_allocations("lmidx", lmidx, cfg, 5, 3)
results[:power_spectrum] = measure_allocations("power_spectrum", power_spectrum, cfg, sh_coeffs)

# 7. Complex transforms analysis
println("\n7. COMPLEX TRANSFORMS ANALYSIS")
println("=" ^ 60)

cplx_spatial = Complex{Float64}.(spatial_data)
cplx_coeffs = SHTnsKit.allocate_spectral_complex(cfg)

results[:cplx_analysis] = measure_allocations("cplx_spat_to_sh", 
    SHTnsKit.cplx_spat_to_sh, cfg, cplx_spatial)
    
results[:cplx_synthesis] = measure_allocations("cplx_sh_to_spat",
    SHTnsKit.cplx_sh_to_spat, cfg, cplx_coeffs)

# Summary report
println("\n8. PERFORMANCE SUMMARY")
println("=" ^ 60)

for (name, result) in results
    println("$name:")
    println("  Time: ", BenchmarkTools.prettytime(median(result.times)))  
    println("  Memory: ", BenchmarkTools.prettymemory(median(result.memory)))
    println("  Allocations: ", median(result.allocs))
    if median(result.memory) > 1024
        println("    High memory usage detected!")
    end
    if median(result.allocs) > 100
        println("    High allocation count detected!")
    end
    println()
end

# Threading analysis
println("\n9. THREADING ANALYSIS")
println("=" ^ 60)

println("Current threading state:")
println("  SHTnsKit threading: ", get_threading())
println("  FFTW threads: ", get_fft_threads())
println("  Julia threads: ", Threads.nthreads())

# Test with threading disabled
set_threading!(false)
result_no_thread = @benchmark spat_to_sh!($cfg, $spatial_data, $sh_coeffs)

set_threading!(true)  
result_with_thread = @benchmark spat_to_sh!($cfg, $spatial_data, $sh_coeffs)

println("Performance comparison:")
println("  Without threading: ", BenchmarkTools.prettytime(median(result_no_thread.times)))
println("  With threading: ", BenchmarkTools.prettytime(median(result_with_thread.times)))

speedup = median(result_no_thread.times) / median(result_with_thread.times)
println("  Speedup: ", round(speedup, digits=2), "x")

# Memory layout analysis
println("\n10. MEMORY LAYOUT ANALYSIS")
println("=" ^ 60)

println("Array layouts:")
println("  spatial_data: ", typeof(spatial_data), " size: ", size(spatial_data))
println("  sh_coeffs: ", typeof(sh_coeffs), " size: ", size(sh_coeffs))
println("  cfg.plm_cache: ", typeof(cfg.plm_cache), " size: ", size(cfg.plm_cache))

# Check for type instabilities in core data structures
println("  Config field types:")
for field in fieldnames(typeof(cfg))
    field_type = typeof(getfield(cfg, field))
    println("    $field: $field_type")
    if field_type == Any || occursin("Union", string(field_type))
        println("        Potential type instability!")
    end
end

println("\nAnalysis complete!")
println("Check results above for optimization opportunities.")