#!/usr/bin/env julia

"""
MPI + SHTnsKit Comprehensive Demo
=================================

This example demonstrates MPI-distributed spherical harmonic transforms with:
- MPI-enabled scalar transforms
- MPI vector field transforms
- Distributed power spectrum analysis
- MPI + OpenMP hybrid parallelism
- Optional MPI + GPU acceleration
"""

using MPI
using SHTnsKit
using LinearAlgebra
using Printf
using Random

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Set up threading to avoid oversubscription
ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "2")

if rank == 0
    println("MPI + SHTnsKit Comprehensive Demo")
    println("=================================")
    println("Running on $size MPI ranks")
end

# 1. MPI Configuration Setup
if rank == 0
    println("\n1. MPI Configuration Setup")
    println("-" * 27)
end

# Try to enable native MPI transforms
mpi_enabled = false
try
    # Try common MPI symbol names
    SHTnsKit.SHTnsKitMPIExt.enable_native_mpi!(
        create="shtns_mpi_create_with_opts",
        set_grid="shtns_mpi_set_grid",
        sh2spat="shtns_mpi_sh_to_spat",
        spat2sh="shtns_mpi_spat_to_sh",
        free="shtns_mpi_free"
    )
    mpi_enabled = SHTnsKit.SHTnsKitMPIExt.is_native_mpi_enabled()
catch e
    if rank == 0
        println("Native MPI symbols not found, using fallback per-rank transforms")
    end
end

if rank == 0
    if mpi_enabled
        println("✓ Native MPI transforms enabled")
    else
        println("⚠ Using fallback per-rank transforms")
    end
end

# 2. Create MPI Configuration
if rank == 0
    println("\n2. Creating MPI Configuration")
    println("-" * 29)
end

lmax, mmax = 32, 32
cfg = SHTnsKit.SHTnsKitMPIExt.create_mpi_config(comm, lmax, mmax, 1)
set_grid(cfg, 2*lmax+1, 2*mmax+1, SHTnsFlags.SHT_REGULAR)

nlat = SHTnsKit.get_nlat(cfg.cfg)
nphi = SHTnsKit.get_nphi(cfg.cfg)
nlm = SHTnsKit.get_nlm(cfg.cfg)

if rank == 0
    println("Grid: $nlat × $nphi points per rank")
    println("Spectral: $nlm coefficients per rank")
end

# 3. Set up per-rank random data with known seed
if rank == 0
    println("\n3. Generating Test Data")
    println("-" * 23)
end

Random.seed!(42 + rank)  # Different but reproducible per rank
sh_local = SHTnsKit.allocate_spectral(cfg.cfg)
rand!(sh_local)

if rank == 0
    println("Generated random spectral data on each rank")
end

# 4. MPI Scalar Transforms
if rank == 0
    println("\n4. MPI Scalar Transforms")
    println("-" * 24)
end

# Forward transform
spat_local = SHTnsKit.allocate_spatial(cfg.cfg)
@time SHTnsKit.synthesize!(cfg, sh_local, spat_local)

# Backward transform
sh_recovered = SHTnsKit.allocate_spectral(cfg.cfg)
@time SHTnsKit.analyze!(cfg, spat_local, sh_recovered)

# Check local accuracy
local_error = maximum(abs.(sh_local - sh_recovered))
if rank == 0
    println("Transform completed on all ranks")
end

# Gather errors from all ranks
all_errors = MPI.Gather(local_error, 0, comm)
if rank == 0
    max_error = maximum(all_errors)
    avg_error = sum(all_errors) / length(all_errors)
    println("Round-trip errors - max: $(@sprintf("%.2e", max_error)), avg: $(@sprintf("%.2e", avg_error))")
end

# 5. MPI Vector Transforms (if enabled)
if rank == 0
    println("\n5. MPI Vector Transforms")
    println("-" * 24)
end

# Create vector field data
u_local = rand(nlat, nphi)
v_local = rand(nlat, nphi)

# Vector transforms
tor_local = SHTnsKit.allocate_spectral(cfg.cfg)
pol_local = SHTnsKit.allocate_spectral(cfg.cfg)

try
    # Try MPI vector transforms
    @time SHTnsKit.analyze_vec!(cfg, u_local, v_local, tor_local, pol_local)
    
    # Reconstruct
    u_recovered = SHTnsKit.allocate_spatial(cfg.cfg)
    v_recovered = SHTnsKit.allocate_spatial(cfg.cfg)
    @time SHTnsKit.synthesize_vec!(cfg, tor_local, pol_local, u_recovered, v_recovered)
    
    # Check accuracy
    u_error = maximum(abs.(u_local - u_recovered))
    v_error = maximum(abs.(v_local - v_recovered))
    
    all_u_errors = MPI.Gather(u_error, 0, comm)
    all_v_errors = MPI.Gather(v_error, 0, comm)
    
    if rank == 0
        println("Vector transform errors - u: $(@sprintf("%.2e", maximum(all_u_errors))), v: $(@sprintf("%.2e", maximum(all_v_errors)))")
    end
    
catch e
    if rank == 0
        println("MPI vector transforms not available, using fallback")
    end
    # Fallback to per-rank vector transforms
    tor_local, pol_local = SHTnsKit.analyze_vector(cfg.cfg, u_local, v_local)
    u_recovered, v_recovered = SHTnsKit.synthesize_vector(cfg.cfg, tor_local, pol_local)
end

# 6. Distributed Power Spectrum Analysis
if rank == 0
    println("\n6. Distributed Power Spectrum")
    println("-" * 27)
end

# Compute local power spectrum
local_power = SHTnsKit.power_spectrum(cfg.cfg, sh_local)

# Reduce across all ranks to get global power spectrum
global_power = MPI.Reduce(local_power, MPI.SUM, 0, comm)

if rank == 0
    total_power = sum(global_power)
    println("Global total power: $(@sprintf("%.2e", total_power))")
    
    # Find dominant modes globally
    if length(global_power) > 1
        threshold = 0.01 * maximum(global_power)
        dominant_modes = findall(x -> x > threshold, global_power)
        println("Global dominant modes (l): $(dominant_modes .- 1)")
    end
end

# 7. Performance Analysis
if rank == 0
    println("\n7. Performance Analysis")
    println("-" * 23)
end

# Time multiple transforms
nruns = 5
times = Float64[]

for i in 1:nruns
    MPI.Barrier(comm)  # Synchronize
    t_start = time()
    SHTnsKit.synthesize!(cfg, sh_local, spat_local)
    MPI.Barrier(comm)  # Ensure all ranks finish
    t_end = time()
    
    if rank == 0  # Only rank 0 records time
        push!(times, t_end - t_start)
    end
end

if rank == 0 && !isempty(times)
    avg_time = sum(times) / length(times)
    std_time = sqrt(sum((times .- avg_time).^2) / length(times))
    println("MPI transform time: $(@sprintf("%.3f ± %.3f", avg_time, std_time)) seconds")
end

# 8. Hybrid MPI + OpenMP Test
if rank == 0
    println("\n8. Hybrid MPI + OpenMP")
    println("-" * 22)
end

# Each rank sets optimal OpenMP threads
local_threads = SHTnsKit.set_optimal_threads(max_threads=2)  # Limit to avoid oversubscription

if rank == 0
    println("Each MPI rank using $local_threads OpenMP threads")
end

# Test with threading
MPI.Barrier(comm)
threaded_start = time()

# Run multiple transforms in parallel on each rank
Threads.@threads for i in 1:2
    local_sh = copy(sh_local)
    local_spat = SHTnsKit.allocate_spatial(cfg.cfg)
    SHTnsKit.synthesize!(cfg, local_sh, local_spat)
end

MPI.Barrier(comm)
threaded_end = time()

if rank == 0
    println("Hybrid MPI+OpenMP test completed in $(@sprintf("%.3f", threaded_end - threaded_start)) seconds")
end

# 9. Optional GPU Test
if rank == 0
    println("\n9. Optional GPU Test")
    println("-" * 20)
end

gpu_available = false
try
    using CUDA
    if CUDA.functional()
        # Select GPU based on MPI rank
        ndev = CUDA.device_count()
        if ndev > 0
            gpu_id = rank % ndev
            CUDA.device!(gpu_id + 1)  # CUDA devices are 1-indexed
            gpu_available = true
            
            if rank == 0
                println("GPU acceleration available on $ndev device(s)")
            end
        end
    end
catch e
    if rank == 0
        println("GPU not available: $e")
    end
end

if gpu_available
    using CUDA
    
    # Test GPU on each rank
    sh_gpu = CUDA.CuArray(sh_local)
    spat_gpu = SHTnsKit.synthesize_gpu(cfg.cfg, sh_gpu)
    
    # Compare with CPU result
    gpu_error = maximum(abs.(spat_local - Array(spat_gpu)))
    
    all_gpu_errors = MPI.Gather(gpu_error, 0, comm)
    if rank == 0
        max_gpu_error = maximum(all_gpu_errors)
        println("Max GPU vs CPU error across all ranks: $(@sprintf("%.2e", max_gpu_error))")
    end
end

# 10. Data Exchange and Global Operations
if rank == 0
    println("\n10. Global Data Operations")
    println("-" * 26)
end

# Compute some global statistics
local_max = maximum(abs.(spat_local))
local_norm = norm(sh_local)

global_max = MPI.Reduce(local_max, MPI.MAX, 0, comm)
global_norm_sq = MPI.Reduce(local_norm^2, MPI.SUM, 0, comm)

if rank == 0
    global_norm = sqrt(global_norm_sq)
    println("Global maximum spatial value: $(@sprintf("%.3e", global_max))")
    println("Global spectral norm: $(@sprintf("%.3e", global_norm))")
end

# 11. Save Results (rank 0 only)
if rank == 0
    println("\n11. Saving Results")
    println("-" * 18)
    
    try
        # Save global power spectrum
        open("mpi_power_spectrum.dat", "w") do f
            println(f, "# l  Global_Power")
            for (i, power) in enumerate(global_power)
                println(f, "$(i-1)  $power")
            end
        end
        println("Global power spectrum saved to 'mpi_power_spectrum.dat'")
    catch e
        println("Could not save results: $e")
    end
end

# 12. Cleanup
if rank == 0
    println("\n12. Cleanup")
    println("-" * 11)
end

# Free configuration
SHTnsKit.free_config(cfg)

MPI.Barrier(comm)
if rank == 0
    println("All MPI ranks completed successfully")
    println("\nMPI comprehensive demo completed! 🌐")
end

MPI.Finalize()