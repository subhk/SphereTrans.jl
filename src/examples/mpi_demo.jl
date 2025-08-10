#!/usr/bin/env julia

using MPI
using SHTnsKit
using LinearAlgebra

# Avoid oversubscription alongside SHTns/OpenMP in examples
ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Optional: per-rank CUDA device selection
gpu_enabled = false
try
    import CUDA
    if hasproperty(CUDA, :functional) && CUDA.functional()
        ndev = hasproperty(CUDA, :device_count) ? CUDA.device_count() : length(CUDA.devices())
        if ndev > 0
            dev = (rank % ndev) + 1
            CUDA.device!(dev)
            gpu_enabled = true
            if rank == 0
                @info "CUDA enabled" ndev
            end
        end
    end
catch
    gpu_enabled = false
end

# Create a per-rank SHTns configuration and grid
lmax, mmax, mres = 16, 16, 1
nlat, nphi = 64, 128
cfg = create_config(lmax, mmax, mres)
set_grid(cfg, nlat, nphi, 0)

# Prepare random spectral coefficients (Float64)
sh = allocate_spectral(cfg)
Random.seed!(0xC0FFEE + rank)
rand!(sh)

# Perform transforms per rank
if gpu_enabled
    # Device path via GPU-friendly helpers
    import CUDA
    shd = CUDA.CuArray(sh)
    spatd = synthesize_gpu(cfg, shd)
    shd2 = analyze_gpu(cfg, spatd)
    # Stage back to host for a quick consistency check
    sh2 = Array(shd2)
else
    # CPU path
    spat = synthesize(cfg, sh)
    sh2 = analyze(cfg, spat)
end

# Simple accuracy report per rank
relerr = norm(sh2 - sh) / max(norm(sh), eps())
println("[rank $(rank)/$(size)] nlm=$(length(sh)) relerr=$(relerr)")

free_config(cfg)
MPI.Barrier(comm)
MPI.Finalize()

