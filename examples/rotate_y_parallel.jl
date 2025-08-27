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
    # Build a distributed PencilArray field
    function _procgrid(p)
        best=(1,p); diff=p-1
        for d in 1:p
            if p % d == 0
                d2 = div(p,d)
                if abs(d-d2) < diff
                    best=(d,d2); diff=abs(d-d2)
                end
            end
        end
        return best
    end
    p = MPI.Comm_size(comm)
    pθ,pφ = _procgrid(p)
    topo = Pencil((:θ, :φ), (nlat, nlon); comm)
    # test real field
    fθφ = PencilArrays.allocate(topo; dims=(:θ, :φ), eltype=Float64)
    fill!(fθφ, 0)
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3*(iθ+1)) + 0.7*cos(0.2*(iφ+1))
    end

    # Distributed analysis -> Alm
    aplan = DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    dist_analysis!(aplan, Alm, fθφ)

    # Allgatherm Y-rotation using PencilArray
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    dist_SH_Yrotate_allgatherm!(cfg, Alm_p, β, R_p)

    # Synthesize rotated field (optional)
    spln = DistPlan(cfg, fθφ)
    fθφ_rot = similar(fθφ)
    dist_synthesis!(spln, fθφ_rot, R_p)

    MPI.Finalize()
end

main()
