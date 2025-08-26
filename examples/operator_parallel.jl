#!/usr/bin/env julia

using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    lmax = 16
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    # Build a test field f(θ,φ) using a serial array (still running under MPI)
    fθφ = zeros(Float64, nlat, nlon)
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3*(iθ+1)) * cos(0.2*(iφ+1))
    end

    # Distributed analysis -> Alm
    aplan = DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    dist_analysis!(aplan, Alm, fθφ)

    # Build cosθ operator coefficients (packed) and apply in spectral space
    mx = zeros(Float64, 2*cfg.nlm)
    mul_ct_matrix(cfg, mx)
    use_halo = any(x -> x == "--halo", ARGS)
    if use_halo
        # Halo-exchange operator on distributed Alm
        Alm_p = PencilArray(Alm)
        R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
        # Detect path: neighbor-only when full m is local
        gl_m = globalindices(Alm_p, 2)
        full_m = (first(gl_m) == 1 && last(gl_m) == cfg.mmax+1 && length(axes(Alm_p,2)) == cfg.mmax+1)
        if rank == 0
            println(full_m ? "[halo] using neighbor Sendrecv halos along l" : "[halo] using per-m Allgatherv")
        end
        dist_SH_mul_mx!(cfg, mx, Alm_p, R_p)
        # Synthesize result back to grid
        spln = DistPlan(cfg, fθφ)
        fθφ_op = similar(fθφ)
        dist_synthesis!(spln, fθφ_op, R_p)
    else
        # Dense path
        Rlm = zeros(ComplexF64, size(Alm))
        dist_SH_mul_mx!(cfg, mx, Alm, Rlm)
        spln = DistPlan(cfg, fθφ)
        fθφ_op = similar(fθφ)
        dist_synthesis!(spln, fθφ_op, PencilArray(Rlm))
    end

    # Reference: multiply in grid-space by cosθ and compare
    gθφ = similar(fθφ)
    for iθ in 1:nlat
        ct = cos(cfg.θ[iθ])
        gθφ[iθ, :] .= ct .* fθφ[iθ, :]
    end

    # Analysis→synthesis to align normalization if needed (direct compare in grid space)
    # Compute relative error between spectral-operator result and grid-space multiplication
    op_out = Array(fθφ_op)
    ref = Array(gθφ)
    rel = sqrt(sum(abs2, op_out .- ref) / (sum(abs2, ref) + eps()))
    if rank == 0
        println("[cosθ operator] relative grid error: ", rel)
    end

    MPI.Finalize()
end

main()
