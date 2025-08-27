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
    # Build a test field f(θ,φ) using a PencilArray (distributed)
    function _procgrid(p)
        best = (1,p); diff = p-1
        for d in 1:p
            if p % d == 0
                d2 = div(p,d)
                if abs(d-d2) < diff
                    best = (d,d2); diff = abs(d-d2)
                end
            end
        end
        return best
    end
    p = MPI.Comm_size(comm)
    pθ,pφ = _procgrid(p)
    topo = Pencil((nlat, nlon), (pθ, pφ), comm)
    fθφ = PencilArrays.allocate(topo; eltype=Float64)
    fill!(fθφ, 0)
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
    # Apply operator in spectral space (pencil path)
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    dist_SH_mul_mx!(cfg, mx, Alm_p, R_p)
    # Synthesize back to grid (distributed)
    spln = DistPlan(cfg, fθφ)
    fθφ_op = similar(fθφ)
    dist_synthesis!(spln, fθφ_op, R_p)

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
