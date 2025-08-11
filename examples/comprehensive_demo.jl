#!/usr/bin/env julia

"""
Comprehensive SHTnsKit.jl Demo
===============================

This example demonstrates all the major features of SHTnsKit.jl including:
- Basic scalar transforms
- Complex field transforms  
- Vector field transforms
- Field rotations
- Power spectrum analysis
- GPU acceleration (if available)
- Multi-threading optimization
"""

using SHTnsKit
using LinearAlgebra
using Printf

println("SHTnsKit.jl Comprehensive Demo")
println("==============================")

# Set up optimal threading
println("Setting optimal OpenMP threads...")
nthreads = set_optimal_threads()
println("Using $nthreads OpenMP threads")

# 1. Basic Scalar Transforms
println("\n1. Basic Scalar Transforms")
println("-" * 25)

# Create configurations for different grid types
lmax, mmax = 16, 16
cfg_gauss = create_gauss_config(lmax, mmax)
cfg_regular = create_regular_config(lmax, mmax)

println("Created Gauss config: nlat=$(get_nlat(cfg_gauss)), nphi=$(get_nphi(cfg_gauss))")
println("Created regular config: nlat=$(get_nlat(cfg_regular)), nphi=$(get_nphi(cfg_regular))")

# Create a test function on the sphere (Y_2^1 harmonic)
function create_test_field(cfg)
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    spat = zeros(nlat, nphi)
    
    for i in 1:nlat
        theta = get_theta(cfg, i-1)  # SHTns uses 0-based indexing
        for j in 1:nphi
            phi = get_phi(cfg, j)
            # Y_2^1 = sqrt(15/(8π)) * sin(theta) * cos(theta) * cos(phi)
            spat[i, j] = sqrt(15/(8*π)) * sin(theta) * cos(theta) * cos(phi)
        end
    end
    return spat
end

# Analyze and synthesize
spat_test = create_test_field(cfg_gauss)
sh_coeffs = analyze(cfg_gauss, spat_test)
spat_reconstructed = synthesize(cfg_gauss, sh_coeffs)

error = maximum(abs.(spat_test - spat_reconstructed))
println("Round-trip error (Gauss grid): $(error)")

# 2. Complex Field Transforms
println("\n2. Complex Field Transforms")
println("-" * 27)

# Create complex test field
sh_complex = allocate_complex_spectral(cfg_gauss)
sh_complex[1] = 1.0 + 0.5im  # Set l=0, m=0 mode
if length(sh_complex) > 5
    sh_complex[6] = 0.3 + 0.2im  # Set another mode safely
end

spat_complex = synthesize_complex(cfg_gauss, sh_complex)
sh_complex_recovered = analyze_complex(cfg_gauss, spat_complex)

complex_error = maximum(abs.(sh_complex - sh_complex_recovered))
println("Complex field round-trip error: $(complex_error)")

# 3. Vector Field Transforms
println("\n3. Vector Field Transforms")
println("-" * 26)

# Create random spheroidal and toroidal coefficients
nlm = get_nlm(cfg_gauss)
Slm = rand(nlm) * 0.1  # Small random coefficients
Tlm = rand(nlm) * 0.1

# Vector synthesis: (Slm, Tlm) -> (Vtheta, Vphi)
Vt, Vp = synthesize_vector(cfg_gauss, Slm, Tlm)
println("Vector field synthesized: size = $(size(Vt))")

# Vector analysis: (Vtheta, Vphi) -> (Slm, Tlm)
Slm_recovered, Tlm_recovered = analyze_vector(cfg_gauss, Vt, Vp)

vector_error_S = maximum(abs.(Slm - Slm_recovered))
vector_error_T = maximum(abs.(Tlm - Tlm_recovered))
println("Vector transform errors - S: $(vector_error_S), T: $(vector_error_T)")

# Test gradient and curl
Vt_grad, Vp_grad = compute_gradient(cfg_gauss, Slm)
Vt_curl, Vp_curl = compute_curl(cfg_gauss, Tlm)
println("Gradient field computed: max|∇S| = $(@sprintf("%.3e", maximum(sqrt.(Vt_grad.^2 + Vp_grad.^2))))")
println("Curl field computed: max|∇×T| = $(@sprintf("%.3e", maximum(sqrt.(Vt_curl.^2 + Vp_curl.^2))))")

# 4. Field Rotations
println("\n4. Field Rotations")
println("-" * 18)

# Rotate the test field
alpha, beta, gamma = π/6, π/4, π/3  # Euler angles
sh_test = analyze(cfg_gauss, spat_test)
sh_rotated = rotate_field(cfg_gauss, sh_test, alpha, beta, gamma)

# Also test spatial rotation
spat_rotated = rotate_spatial_field(cfg_gauss, spat_test, alpha, beta, gamma)

rotation_spectral_diff = norm(sh_test - sh_rotated)
println("Spectral rotation difference: $(rotation_spectral_diff)")
println("Spatial rotation completed successfully")

# 5. Power Spectrum Analysis
println("\n5. Power Spectrum Analysis")
println("-" * 26)

# Compute power spectrum
power = power_spectrum(cfg_gauss, sh_test)
total_power = sum(power)
println("Total power: $(total_power)")

# Find dominant modes
if length(power) > 1
    threshold = 0.01 * maximum(power)
    dominant_modes = findall(x -> x > threshold, power)
    println("Dominant modes (l): $(dominant_modes .- 1)")  # Convert to 0-based
end

# 6. GPU Acceleration (if available)
println("\n6. GPU Acceleration")
println("-" * 19)

try
    using CUDA
    if CUDA.functional()
        # Initialize GPU
        gpu_success = initialize_gpu(0, verbose=true)
        
        if gpu_success
            # Test GPU transforms
            cfg_gpu = create_gpu_config(lmax, mmax)
            
            # CPU reference
            sh_cpu = rand(get_nlm(cfg_gpu))
            spat_cpu = synthesize(cfg_gpu, sh_cpu)
            
            # GPU computation
            sh_gpu = CUDA.CuArray(sh_cpu)
            spat_gpu = synthesize_gpu(cfg_gpu, sh_gpu)
            
            # Compare results
            gpu_error = maximum(abs.(spat_cpu - Array(spat_gpu)))
            println("GPU vs CPU error: $(gpu_error)")
            
            # Cleanup
            cleanup_gpu(verbose=true)
            free_config(cfg_gpu)
        else
            println("GPU initialization failed")
        end
    else
        println("CUDA not functional - skipping GPU tests")
    end
catch e
    println("CUDA not available - skipping GPU tests: $(e)")
end

# 7. Performance Comparison
println("\n7. Performance Comparison")
println("-" * 25)

# Benchmark different transform types
sh_bench = rand(get_nlm(cfg_gauss))
spat_bench = allocate_spatial(cfg_gauss)

println("Benchmarking transforms (lmax=$lmax)...")

# Forward transform benchmark
print("Forward transform:  ")
@time synthesize!(cfg_gauss, sh_bench, spat_bench)

# Backward transform benchmark
spat_bench2 = rand(get_nlat(cfg_gauss), get_nphi(cfg_gauss))
sh_bench2 = allocate_spectral(cfg_gauss)

print("Backward transform: ")
@time analyze!(cfg_gauss, spat_bench2, sh_bench2)

# 8. Cleanup
println("\n8. Cleanup")
println("-" * 10)

free_config(cfg_gauss)
free_config(cfg_regular)

println("All configurations freed successfully")
println("\nDemo completed! 🎉")

# Optional: Create a simple power spectrum output
try
    open("power_spectrum.dat", "w") do f
        println(f, "# l  Power")
        for (i, p) in enumerate(power)
            println(f, "$(i-1)  $p")
        end
    end
    println("Power spectrum saved as 'power_spectrum.dat'")
catch e
    println("Could not save power spectrum: $e")
end