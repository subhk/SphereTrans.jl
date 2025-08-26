module SHTnsKitParallelExt

using MPI
using PencilArrays
using PencilFFTs
using ..SHTnsKit

"""
    dist_analysis(cfg, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)

Distributed scalar analysis using PencilArrays/PencilFFTs.
Pipeline: (θ,φ) -> FFT_φ -> (θ,k) -> transpose to (θ,m) -> per-m Legendre dot-products -> (l,m).
If pencils are effectively local (single-process), falls back to serial analysis.
"""
function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    comm = PencilArrays.communicator(fθφ)
    np = MPI.Comm_size(comm)
    # Fallback: single-process or degenerate pencil
    if np == 1
        f = Array(fθφ)
        return SHTnsKit.analysis(cfg, f)
    end
    # 1) FFT along φ (pencil-friendly)
    pfft = PencilFFTs.plan_fft(fθφ; dims=2)  # assumes 2nd dim is φ
    Fθk = PencilFFTs.fft(fθφ, pfft)
    # 2) Transpose (θ,k) -> (θ,m) (PencilArrays manages data motion)
    Fθm = PencilArrays.transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    # 3) Per-m Legendre stage: partial sums per rank if θ is split
    # Build local alm buffer for owned m
    lmax, mmax = cfg.lmax, cfg.mmax
    Alm_local = zeros(ComplexF64, lmax+1, mmax+1)
    θrange = axes(Fθm, 1)
    mrange = axes(Fθm, 2)
    # Use precomputed Plm tables if available and local θ indices align
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, lmax + 1)
    for m in mrange
        mm = m - first(mrange)  # local index offset
        mglob = PencilArrays.globalindices(Fθm, 2)[mm+1]  # global m index (1-based)
        mval = mglob - 1
        col = mval + 1
        for (ii,i) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm, 1)[ii]
            if use_tbl
                # use table column
                tblcol = view(cfg.plm_tables[col], :, iglob)
                @inbounds for l in mval:lmax
                    Alm_local[l+1, col] += cfg.w[iglob] * tblcol[l+1] * Fθm[i, m]
                end
            else
                SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                @inbounds for l in mval:lmax
                    Alm_local[l+1, col] += cfg.w[iglob] * P[l+1] * Fθm[i, m]
                end
            end
        end
    end
    # 4) Reduce across θ-pencil to sum contributions
    MPI.Allreduce!(Alm_local, +, comm)
    # 5) Apply normalization and φ scaling (2π/nlon)
    scaleφ = cfg.cphi
    @inbounds for m in 0:mmax, l in m:lmax
        Alm_local[l+1, m+1] *= cfg.Nlm[l+1, m+1] * scaleφ
    end
    return Alm_local
end

"""
    dist_synthesis(cfg, Alm::AbstractMatrix)

Distributed scalar synthesis. Assumes `Alm` is globally consistent on all ranks,
or that the caller distributes `(l,m)` identically across ranks.
Falls back to serial when single-process.
"""
function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix)
    # Simple initial version: all ranks run serial synthesis locally.
    # Next iteration can distribute m and use PencilFFTs + transposes inverse to analysis.
    return SHTnsKit.synthesis(cfg, Alm)
end

end # module
