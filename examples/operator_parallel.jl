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

    # Analysis -> Alm (serial transform)
    Alm = analysis(cfg, fθφ)

    # Build cosθ operator coefficients (packed) and apply in spectral space
    mx = zeros(Float64, 2*cfg.nlm)
    mul_ct_matrix(cfg, mx)
    # Apply operator in spectral space (dense path)
    Rlm = zeros(ComplexF64, size(Alm))
    dist_SH_mul_mx!(cfg, mx, Alm, Rlm)
    # Synthesize back to grid (serial)
    fθφ_op = synthesis(cfg, Rlm; real_output=true)

    # Reference: multiply in grid-space by cosθ and compare
    gθφ = similar(fθφ)
    for iθ in 1:nlat
        ct = cos(cfg.θ[iθ])
        gθφ[iθ, :] .= ct .* fθφ[iθ, :]
    end

    # Analysis→synthesis to align normalization if needed (direct compare in grid space)
    # Compute relative error between spectral-operator result and grid-space multiplication
    op_out = fθφ_op
    ref = gθφ
    rel = sqrt(sum(abs2, op_out .- ref) / (sum(abs2, ref) + eps()))
    if rank == 0
        println("[cosθ operator] relative grid error: ", rel)
    end

    MPI.Finalize()
end

main()
