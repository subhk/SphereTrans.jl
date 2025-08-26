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
    β = 0.35

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    Pθφ = Pencil((:θ, :φ), (nlat, nlon); comm)

    # Build a test real field f(θ,φ)
    fθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3*(iθ+1)) + 0.7*cos(0.2*(iφ+1))
    end

    # Distributed analysis -> Alm
    aplan = DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    dist_analysis!(aplan, Alm, fθφ)

    # Allgatherm distributed Y-rotation
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    dist_SH_Yrotate_allgatherm!(cfg, Alm_p, β, R_p)

    # Dense gather/apply/scatter rotation for reference
    R_dense = zeros(ComplexF64, size(Alm))
    dist_SH_Yrotate(cfg, Alm, β, R_dense)

    # Compare pencil vs dense
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    num = 0.0; den = 0.0
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        x = R_p[il, jm] - R_dense[gl_l[ii], gl_m[jj]]
        num += abs2(x)
        den += abs2(R_dense[gl_l[ii], gl_m[jj]])
    end
    rel = sqrt(num / (den + eps()))
    if rank == 0
        println("[Y-rotation allgatherm] relative spectral error vs dense: ", rel)
    end

    # Synthesize rotated field (optional)
    spln = DistPlan(cfg, fθφ)
    fθφ_rot = similar(fθφ)
    dist_synthesis!(spln, fθφ_rot, R_p)

    MPI.Finalize()
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
