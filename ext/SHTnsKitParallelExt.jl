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

"""
    dist_spat_to_SHsphtor(cfg, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)

Distributed vector analysis. Returns local dense Slm,Tlm matrices reduced across θ-pencil communicators.
For single rank, falls back to serial.
"""
function SHTnsKit.dist_spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    comm = PencilArrays.communicator(Vtθφ)
    np = MPI.Comm_size(comm)
    if np == 1
        return SHTnsKit.spat_to_SHsphtor(cfg, Array(Vtθφ), Array(Vpθφ))
    end
    # FFT along φ
    pfft_t = PencilFFTs.plan_fft(Vtθφ; dims=2)
    pfft_p = PencilFFTs.plan_fft(Vpθφ; dims=2)
    Fθk_t = PencilFFTs.fft(Vtθφ, pfft_t)
    Fθk_p = PencilFFTs.fft(Vpθφ, pfft_p)
    # Transpose to (θ,m)
    Fθm_t = PencilArrays.transpose(Fθk_t, (; dims=(1,2), names=(:θ,:m)))
    Fθm_p = PencilArrays.transpose(Fθk_p, (; dims=(1,2), names=(:θ,:m)))
    # Per-m vector projections
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm_local = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm_local = zeros(ComplexF64, lmax+1, mmax+1)
    θrange = axes(Fθm_t, 1)
    mrange = axes(Fθm_t, 2)
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    for m in mrange
        mm = m - first(mrange)
        mglob = PencilArrays.globalindices(Fθm_t, 2)[mm+1]
        mval = mglob - 1
        col = mval + 1
        for (ii,i) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm_t, 1)[ii]
            x = cfg.x[iglob]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Ft = Fθm_t[i, m]
            Fp = Fθm_p[i, m]
            wi = cfg.w[iglob]
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglob]
                    Y = N * tblP[l+1, iglob]
                    coeff = wi * cfg.cphi / (l*(l+1))
                    Slm_local[l+1, col] += coeff * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * tbld[l+1, iglob]))
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    coeff = wi * cfg.cphi / (l*(l+1))
                    Slm_local[l+1, col] += coeff * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * dPdx[l+1]))
                end
            end
        end
    end
    MPI.Allreduce!(Slm_local, +, comm)
    MPI.Allreduce!(Tlm_local, +, comm)
    # Convert to cfg norm/CS
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_local); T2 = similar(Tlm_local)
        SHTnsKit.convert_alm_norm!(S2, Slm_local, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(T2, Tlm_local, cfg; to_internal=false)
        return S2, T2
    end
    return Slm_local, Tlm_local
end


end # module
