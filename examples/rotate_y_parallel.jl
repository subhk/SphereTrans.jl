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

    # Build a test real field f(θ,φ)
    fθφ = zeros(Float64, nlat, nlon)
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3*(iθ+1)) + 0.7*cos(0.2*(iφ+1))
    end

    # Analysis -> Alm (serial path)
    Alm = analysis(cfg, fθφ)

    # Dense Y-rotation for reference
    R_dense = zeros(ComplexF64, size(Alm))
    dist_SH_Yrotate(cfg, Alm, β, R_dense)

    # Synthesize rotated field (optional)
    fθφ_rot = synthesis(cfg, R_dense; real_output=true)

    MPI.Finalize()
end

main()
