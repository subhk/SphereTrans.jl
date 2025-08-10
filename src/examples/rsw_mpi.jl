#!/usr/bin/env julia

using MPI
using SHTnsKit

# Limit oversubscription with any internal threading (OpenMP/BLAS)
ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OPENBLAS_NUM_THREADS"] = get(ENV, "OPENBLAS_NUM_THREADS", "1")

"""
Rotating Shallow Water (linear, constant f) on the sphere — CPU + MPI.

Notes
- Uses a linearized RSW system with constant Coriolis f0 and no advection.
- Evolves spectral coefficients for (η, ζ, δ) per (l,m) mode with RK4.
- Distributes (l,m) modes across MPI ranks; transforms are local (not MPI).
- Requires only scalar transforms from SHTnsKit.

Equations per mode (l,m), with L = l(l+1)/a^2:
  dη/dt = -H δ
  dζ/dt = -f0 δ
  dδ/dt =  g L η + f0 ζ

This simple linear model demonstrates structure; nonlinear terms are omitted.
"""

# --- Parameters ---
const a   = 6.371e6           # sphere radius (m)
const g   = 9.80616           # gravity (m/s^2)
const H   = 1.0e4             # mean depth (m)
const f0  = 1.0e-4            # constant Coriolis parameter (1/s)
const dt  = 50.0              # time step (s)
const nsteps = 200            # number of steps
const output_stride = 50      # how often to synthesize/print

const lmax = 31
const mmax = 31
const mres = 1
const nlat = 64
const nphi = 128

# --- Helpers ---
function lm_arrays(cfg)
    nlm = get_nlm(cfg)
    L = zeros(Float64, nlm)
    M = zeros(Int, nlm)
    lmax = get_lmax(cfg)
    mmax = get_mmax(cfg)
    for m in 0:mmax
        for l in m:lmax
            idx = lmidx(cfg, l, m) + 1
            L[idx] = l * (l + 1) / (a^2)
            M[idx] = m
        end
    end
    return L, M
end

"""Block partition of 1:n across size ranks; return (lo, hi) for rank r."""
function block_bounds(n, r, size)
    base = fld(n, size)
    rem = n % size
    lo = r * base + min(r, rem) + 1
    hi = lo + base - 1 + (r < rem ? 1 : 0)
    return lo, hi
end

function rhs_linear!(dη, dζ, dδ, η, ζ, δ, L)
    @inbounds for i in eachindex(η)
        dη[i] = -H * δ[i]
        dζ[i] = -f0 * δ[i]
        dδ[i] =  g * L[i] * η[i] + f0 * ζ[i]
    end
    return nothing
end

function rk4_step!(η, ζ, δ, L, dt, work)
    k1η, k1ζ, k1δ,
    k2η, k2ζ, k2δ,
    k3η, k3ζ, k3δ,
    k4η, k4ζ, k4δ,
    ηtmp, ζtmp, δtmp = work

    rhs_linear!(k1η, k1ζ, k1δ, η, ζ, δ, L)

    @. ηtmp = η + 0.5*dt*k1η;  @. ζtmp = ζ + 0.5*dt*k1ζ;  @. δtmp = δ + 0.5*dt*k1δ
    rhs_linear!(k2η, k2ζ, k2δ, ηtmp, ζtmp, δtmp, L)

    @. ηtmp = η + 0.5*dt*k2η;  @. ζtmp = ζ + 0.5*dt*k2ζ;  @. δtmp = δ + 0.5*dt*k2δ
    rhs_linear!(k3η, k3ζ, k3δ, ηtmp, ζtmp, δtmp, L)

    @. ηtmp = η + dt*k3η;      @. ζtmp = ζ + dt*k3ζ;      @. δtmp = δ + dt*k3δ
    rhs_linear!(k4η, k4ζ, k4δ, ηtmp, ζtmp, δtmp, L)

    @. η = η + (dt/6.0)*(k1η + 2k2η + 2k3η + k4η)
    @. ζ = ζ + (dt/6.0)*(k1ζ + 2k2ζ + 2k3ζ + k4ζ)
    @. δ = δ + (dt/6.0)*(k1δ + 2k2δ + 2k3δ + k4δ)
    return nothing
end

# --- Main ---
function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # SHTns scalar config (per-rank)
    cfg = create_config(lmax, mmax, mres)
    set_grid(cfg, nlat, nphi, 0)

    # Precompute L(l,m)
    Lfull, Mfull = lm_arrays(cfg)
    nlm = length(Lfull)

    # Partition the spectral indices across ranks
    lo, hi = block_bounds(nlm, rank, size)
    locidx = lo:hi

    # Allocate full arrays; we'll update only local slices and allreduce
    η = zeros(Float64, nlm)
    ζ = zeros(Float64, nlm)
    δ = zeros(Float64, nlm)

    # Initialize η from a simple zonal cosine in physical space (m_init)
    m_init = 2
    spat = allocate_spatial(cfg)
    @inbounds for j in 1:nphi
        λ = 2π * (j-1) / nphi
        val = cos(m_init * λ)
        @inbounds for i in 1:nlat
            spat[i,j] = val
        end
    end
    η .= analyze(cfg, spat)  # spectral coefficients
    fill!(ζ, 0.0)
    fill!(δ, 0.0)

    # Work arrays for RK4 (local to avoid extra allocations)
    k1η = zeros(Float64, nlm); k1ζ = similar(k1η); k1δ = similar(k1η)
    k2η = similar(k1η); k2ζ = similar(k1η); k2δ = similar(k1η)
    k3η = similar(k1η); k3ζ = similar(k1η); k3δ = similar(k1η)
    k4η = similar(k1η); k4ζ = similar(k1η); k4δ = similar(k1η)
    ηtmp = similar(k1η); ζtmp = similar(k1η); δtmp = similar(k1η)
    work = (k1η,k1ζ,k1δ,k2η,k2ζ,k2δ,k3η,k3ζ,k3δ,k4η,k4ζ,k4δ,ηtmp,ζtmp,δtmp)

    # Mask L outside local domain so only local modes update (others remain 0)
    L = zeros(Float64, nlm)
    L[locidx] .= Lfull[locidx]

    # Time stepping
    for step in 1:nsteps
        rk4_step!(η, ζ, δ, L, dt, work)
        # Allreduce to synchronize coefficients across ranks (sum combines disjoint updates)
        MPI.Allreduce!(η, η, MPI.SUM, comm)
        MPI.Allreduce!(ζ, ζ, MPI.SUM, comm)
        MPI.Allreduce!(δ, δ, MPI.SUM, comm)

        if step % output_stride == 0
            # Synthesize η to physical space for a quick metric
            spatη = synthesize(cfg, η)
            # Compute simple norms
            sumnorm = sum(abs, spatη)
            if rank == 0
                @info "step=$(step) sum|η|=$(sumnorm)"
            end
        end
    end

    if rank == 0
        # Final output as a simple checksum
        spatη = synthesize(cfg, η)
        println("Final sum(|η|) = ", sum(abs, spatη))
    end

    free_config(cfg)
    MPI.Barrier(comm)
    MPI.Finalize()
end

abspath(PROGRAM_FILE) == @__FILE__ && main()

