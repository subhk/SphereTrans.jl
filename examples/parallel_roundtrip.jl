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
    do_vector = any(x -> x == "--vector", ARGS)

    cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    Pθφ = PencilArrays.Pencil((:θ, :φ), (nlat, nlon); comm)

    # Scalar roundtrip
    fθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
    # Deterministic-ish local fill
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3 * (iθ + rank + 1)) + cos(0.2 * (iφ + 2))
    end
    rel_local, rel_global = SHTnsKit.dist_scalar_roundtrip!(cfg, fθφ)
    if rank == 0
        println("[scalar] rel_local≈$rel_local rel_global≈$rel_global")
    end

    if do_vector
        Vtθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        Vpθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        for (iθ, iφ) in zip(eachindex(axes(Vtθφ,1)), eachindex(axes(Vtθφ,2)))
            Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)
            Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))
        end
        (rl_t, rg_t), (rl_p, rg_p) = SHTnsKit.dist_vector_roundtrip!(cfg, Vtθφ, Vpθφ)
        if rank == 0
            println("[vector] Vt rel_local≈$rl_t rel_global≈$rg_t; Vp rel_local≈$rl_p rel_global≈$rg_p")
        end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

abspath(PROGRAM_FILE) == @__FILE__ && main()

