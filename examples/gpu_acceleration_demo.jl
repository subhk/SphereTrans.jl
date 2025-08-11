#!/usr/bin/env julia

"""
GPU Acceleration Demo
=====================

This example demonstrates GPU-accelerated spherical harmonic transforms:
- Basic GPU transforms with CUDA
- Performance comparison CPU vs GPU
- GPU vector transforms
- GPU complex field transforms
- Memory management on GPU
"""

using SHTnsKit
using LinearAlgebra
using Printf

println("GPU Acceleration Demo")
println("====================")

# Check if CUDA is available
cuda_available = false
try
    using CUDA
    cuda_available = CUDA.functional()
    if cuda_available
        println("CUDA detected: $(CUDA.name()) with $(CUDA.device_count()) device(s)")
    else
        println("CUDA not functional")
    end
catch e
    println("CUDA not available: $e")
end

if !cuda_available
    println("This demo requires CUDA.jl and a working GPU setup.")
    println("Exiting...")
    exit(0)
end

using CUDA

# Initialize GPU
println("\n1. GPU Initialization")
println("-" * 21)

gpu_success = initialize_gpu(0, verbose=true)
if !gpu_success
    println("Failed to initialize GPU. Exiting...")
    exit(1)
end

# Create configurations
lmax = 64
println("\n2. Configuration Setup (lmax=$lmax)")
println("-" * 35)

cfg_cpu = create_gauss_config(lmax, lmax)
cfg_gpu = create_gpu_config(lmax, lmax)

nlat, nphi = get_nlat(cfg_cpu), get_nphi(cfg_cpu)
nlm = get_nlm(cfg_cpu)

println("Grid: $nlat × $nphi = $(nlat*nphi) spatial points")
println("Spectral: $nlm coefficients")

# 3. Basic GPU Transforms
println("\n3. Basic GPU Transforms")
println("-" * 23)

# Create test data
sh_cpu = rand(Float64, nlm)
println("Created random spectral coefficients")

# CPU transform
println("Performing CPU transform...")
@time spat_cpu = synthesize(cfg_cpu, sh_cpu)

# GPU transform
println("Performing GPU transform...")
sh_gpu = CUDA.CuArray(sh_cpu)
@time spat_gpu = synthesize_gpu(cfg_gpu, sh_gpu)

# Compare results
spat_gpu_host = Array(spat_gpu)
gpu_error = maximum(abs.(spat_cpu - spat_gpu_host))
println("GPU vs CPU difference: $(@sprintf("%.2e", gpu_error))")

# 4. Backward Transform Test
println("\n4. Backward Transform Test")
println("-" * 26)

# CPU backward
@time sh_cpu_back = analyze(cfg_cpu, spat_cpu)

# GPU backward
@time sh_gpu_back = analyze_gpu(cfg_gpu, spat_gpu)

# Compare
sh_gpu_back_host = Array(sh_gpu_back)
back_error = maximum(abs.(sh_cpu_back - sh_gpu_back_host))
println("Backward transform GPU vs CPU difference: $(@sprintf("%.2e", back_error))")

# Round-trip error
roundtrip_error = maximum(abs.(sh_cpu - sh_gpu_back_host))
println("Round-trip error: $(@sprintf("%.2e", roundtrip_error))")

# 5. Performance Benchmarking
println("\n5. Performance Benchmarking")
println("-" * 27)

# Multiple runs for better statistics
nruns = 10
println("Running $nruns iterations for benchmarking...")

# CPU timing
cpu_times = Float64[]
for i in 1:nruns
    t = @elapsed synthesize!(cfg_cpu, sh_cpu, spat_cpu)
    push!(cpu_times, t)
end
cpu_avg = mean(cpu_times)
cpu_std = std(cpu_times)

# GPU timing
gpu_times = Float64[]
spat_gpu_preallocated = similar(sh_gpu, Float64, nlat, nphi)
for i in 1:nruns
    t = @elapsed spat_gpu_result = synthesize_gpu(cfg_gpu, sh_gpu)
    push!(gpu_times, t)
end
gpu_avg = mean(gpu_times)
gpu_std = std(gpu_times)

println("CPU time: $(@sprintf("%.3f ± %.3f", cpu_avg, cpu_std)) seconds")
println("GPU time: $(@sprintf("%.3f ± %.3f", gpu_avg, gpu_std)) seconds")
println("Speedup: $(@sprintf("%.2f", cpu_avg / gpu_avg))x")

# 6. GPU Vector Transforms
println("\n6. GPU Vector Transforms")
println("-" * 24)

# Create vector field data
u_cpu = rand(Float64, nlat, nphi)
v_cpu = rand(Float64, nlat, nphi)
println("Created random vector field")

# CPU vector analysis
@time Slm_cpu, Tlm_cpu = analyze_vector(cfg_cpu, u_cpu, v_cpu)

# GPU vector analysis (using staging)
u_gpu = CUDA.CuArray(u_cpu)
v_gpu = CUDA.CuArray(v_cpu)

# Note: This uses host staging since we don't have native GPU vector transforms
println("GPU vector analysis (with host staging)...")
@time begin
    u_host = Array(u_gpu)
    v_host = Array(v_gpu)
    Slm_gpu_host, Tlm_gpu_host = analyze_vector(cfg_cpu, u_host, v_host)
    Slm_gpu = CUDA.CuArray(Slm_gpu_host)
    Tlm_gpu = CUDA.CuArray(Tlm_gpu_host)
end

# Compare results
slm_error = maximum(abs.(Slm_cpu - Array(Slm_gpu)))
tlm_error = maximum(abs.(Tlm_cpu - Array(Tlm_gpu)))
println("Vector transform errors - Slm: $(@sprintf("%.2e", slm_error)), Tlm: $(@sprintf("%.2e", tlm_error))")

# 7. GPU Complex Field Transforms
println("\n7. GPU Complex Field Transforms")
println("-" * 31)

# Create complex test data
sh_complex_cpu = rand(ComplexF64, nlm)
sh_complex_gpu = CUDA.CuArray(sh_complex_cpu)

println("Testing complex field transforms...")

# CPU complex transform
@time spat_complex_cpu = synthesize_complex(cfg_cpu, sh_complex_cpu)

# GPU complex transform (with staging)
@time begin
    sh_complex_host = Array(sh_complex_gpu)
    spat_complex_host = synthesize_complex(cfg_cpu, sh_complex_host)
    spat_complex_gpu = CUDA.CuArray(spat_complex_host)
end

# Compare
complex_error = maximum(abs.(spat_complex_cpu - Array(spat_complex_gpu)))
println("Complex field GPU vs CPU difference: $(@sprintf("%.2e", complex_error))")

# 8. Memory Usage Analysis
println("\n8. Memory Usage Analysis")
println("-" * 24)

# Check GPU memory usage
free_mem, total_mem = CUDA.memory_info()
used_mem = total_mem - free_mem

println("GPU memory usage:")
println("  Total: $(@sprintf("%.1f", total_mem / 1e9)) GB")
println("  Used:  $(@sprintf("%.1f", used_mem / 1e9)) GB")
println("  Free:  $(@sprintf("%.1f", free_mem / 1e9)) GB")

# Estimate memory for our arrays
sh_mem = sizeof(sh_gpu)
spat_mem = sizeof(spat_gpu)
total_array_mem = sh_mem + spat_mem

println("Our arrays use approximately $(@sprintf("%.1f", total_array_mem / 1e6)) MB")

# 9. Large Scale Test
println("\n9. Large Scale Test")
println("-" * 19)

# Test with larger problem if memory allows
lmax_large = 128
memory_needed = (lmax_large + 1)^2 * 8 + 2 * (lmax_large + 1) * (2 * lmax_large + 1) * 8  # Rough estimate
memory_needed_gb = memory_needed / 1e9

if free_mem > memory_needed * 2  # Safety factor
    println("Testing with lmax=$lmax_large (needs ~$(@sprintf("%.1f", memory_needed_gb)) GB)")
    
    cfg_large = create_gpu_config(lmax_large, lmax_large)
    nlm_large = get_nlm(cfg_large)
    nlat_large, nphi_large = get_nlat(cfg_large), get_nphi(cfg_large)
    
    sh_large_gpu = CUDA.rand(Float64, nlm_large)
    
    println("Large scale GPU transform...")
    @time spat_large_gpu = synthesize_gpu(cfg_large, sh_large_gpu)
    
    println("Large scale GPU analysis...")
    @time sh_large_back_gpu = analyze_gpu(cfg_large, spat_large_gpu)
    
    # Check accuracy
    large_error = maximum(abs.(Array(sh_large_gpu) - Array(sh_large_back_gpu)))
    println("Large scale round-trip error: $(@sprintf("%.2e", large_error))")
    
    free_config(cfg_large)
else
    println("Insufficient GPU memory for large scale test")
    println("Need ~$(@sprintf("%.1f", memory_needed_gb)) GB, have $(@sprintf("%.1f", free_mem / 1e9)) GB free")
end

# 10. Cleanup
println("\n10. Cleanup")
println("-" * 11)

# Free GPU memory
sh_gpu = nothing
spat_gpu = nothing
u_gpu = nothing
v_gpu = nothing
sh_complex_gpu = nothing
spat_complex_gpu = nothing

# Force garbage collection
CUDA.reclaim()
GC.gc()

# Check final memory state
free_mem_final, _ = CUDA.memory_info()
memory_freed = free_mem_final - free_mem
println("Freed $(@sprintf("%.1f", memory_freed / 1e6)) MB of GPU memory")

# Cleanup GPU
cleanup_gpu(verbose=true)

# Free configurations
free_config(cfg_cpu)
free_config(cfg_gpu)

println("All resources cleaned up successfully")
println("\nGPU acceleration demo completed!")