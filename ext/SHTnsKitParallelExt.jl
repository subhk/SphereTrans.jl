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
        # Only accumulate for resolved orders 0..mmax
        if mval > mmax
            continue
        end
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
    dist_scalar_roundtrip!(cfg, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables, real_output=true)

Run distributed scalar round-trip (analysis → synthesis) and return (relerr_local, relerr_global).
Relerr computed as sqrt( sum|f_out - f|^2 / sum|f|^2 ).
"""
function SHTnsKit.dist_scalar_roundtrip!(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; 
                                    use_tables=cfg.use_plm_tables, real_output::Bool=true)
    comm = PencilArrays.communicator(fθφ)
    # Save local original
    f0 = Array(fθφ)
    # Round-trip
    Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_tables)
    fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=real_output)
    fout = Array(fθφ_out)
    # Errors
    num = sum(abs2, fout .- f0)
    den = sum(abs2, f0) + eps()
    rel_local = sqrt(num / den)
    # Global reductions
    num_g = MPI.Allreduce(num, +, comm)
    den_g = MPI.Allreduce(den, +, comm)
    rel_global = sqrt(num_g / den_g)
    return rel_local, rel_global
end

"""
    dist_vector_roundtrip!(cfg, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray;
                           use_tables=cfg.use_plm_tables, real_output=true)

Run distributed vector round-trip and return ((rel_local_t, rel_global_t), (rel_local_p, rel_global_p)).
"""
function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, 
                                Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables, real_output::Bool=true)

    comm = PencilArrays.communicator(Vtθφ)
    T0 = Array(Vtθφ); P0 = Array(Vpθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ; use_tables)
    Vt_out, Vp_out = SHTnsKit.dist_SHsphtor_to_spat(cfg, PencilArrays.PencilArray(Slm), PencilArrays.PencilArray(Tlm); 
                                                prototype_θφ=Vtθφ, real_output=real_output)
                                                
    # Note: Slm/Tlm are dense; above converts to PencilArray by constructor. If not available, use local arrays:
    T1 = Array(Vt_out); P1 = Array(Vp_out)
    # Errors t component
    num_t = sum(abs2, T1 .- T0)
    den_t = sum(abs2, T0) + eps()
    rel_local_t = sqrt(num_t / den_t)
    num_tg = MPI.Allreduce(num_t, +, comm)
    den_tg = MPI.Allreduce(den_t, +, comm)
    rel_global_t = sqrt(num_tg / den_tg)
    # Errors p component
    num_p = sum(abs2, P1 .- P0)
    den_p = sum(abs2, P0) + eps()
    rel_local_p = sqrt(num_p / den_p)
    num_pg = MPI.Allreduce(num_p, +, comm)
    den_pg = MPI.Allreduce(den_p, +, comm)
    rel_global_p = sqrt(num_pg / den_pg)
    return (rel_local_t, rel_global_t), (rel_local_p, rel_global_p)
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
    dist_synthesis(cfg, Alm_pencil::PencilArrays.PencilArray; real_output=true)

Distributed scalar synthesis from a distributed `(l,m)` PencilArray.
Current implementation gathers local data to a dense Array and calls serial synthesis on each rank.
Future improvements will implement the full inverse pipeline with transposes and PencilFFTs.
"""
function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArrays.PencilArray; real_output::Bool=true)
    comm = PencilArrays.communicator(Alm_pencil)
    np = MPI.Comm_size(comm)
    if np == 1
        return SHTnsKit.synthesis(cfg, Array(Alm_pencil); real_output=real_output)
    end
    # Fallback: local serial synthesis
    Alm_local = Array(Alm_pencil)
    return SHTnsKit.synthesis(cfg, Alm_local; real_output=real_output)
end

"""
    dist_synthesis(cfg, Alm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output=true)

Prototype-based distributed synthesis. Builds (θ,m) spectra from Alm (:l,:m), reduces across l-pencil,
maps to (θ,k) with Hermitian, and inverse FFTs along φ to return a (θ,φ) PencilArray matching prototype_θφ.
"""
function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArrays.PencilArray; 
                            prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)

    # Convert Alm to internal normalization if needed
    Alm_mat = Array(Alm)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_int = similar(Alm_mat)
        SHTnsKit.convert_alm_norm!(Alm_int, Alm_mat, cfg; to_internal=true)
        Alm_mat = Alm_int
    end
    # Allocate (θ,m) spectrum pencil using prototype
    Gθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    fill!(Gθm, 0)
    # Local ranges
    θloc = axes(Gθm, 1)
    mloc = axes(Gθm, 2)
    lloc = axes(Alm, 1)
    # Accumulate over local l for owned θ,m
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    for (ii,iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Gθm, 1)[ii]
        xθ = cfg.x[iglobθ]
        for (jj,jm) in enumerate(mloc)
            iglobm = PencilArrays.globalindices(Gθm, 2)[jj]
            mval = iglobm - 1
            col = mval + 1
            acc = 0.0 + 0.0im
            if use_tbl
                tbl = cfg.plm_tables[col]
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Alm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= mval
                        acc += (cfg.Nlm[lval+1, col] * tbl[lval+1, iglobθ]) * Alm_mat[il, iglobm]
                    end
                end
            else
                SHTnsKit.Plm_row!(P, xθ, cfg.lmax, mval)
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Alm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= mval
                        acc += (cfg.Nlm[lval+1, col] * P[lval+1]) * Alm_mat[il, iglobm]
                    end
                end
            end
            # Gauss weight and φ scaling
            acc *= cfg.w[iglobθ] * cfg.cphi
            Gθm[iθ, jm] += acc
        end
    end
    # Reduce across l-pencil communicator (sum partials). Using global comm is acceptable as first pass.
    MPI.Allreduce!(Gθm, +, PencilArrays.communicator(Gθm))
    # Map to (θ,k)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    fill!(Fθk, 0)
    nlon = cfg.nlon
    θloc = axes(Fθk, 1)
    kloc = axes(Fθk, 2)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(Fθk, 2)[jj] - 1
            if kglob == 0
                # m=0 bin
                if first(mloc) <= 1 <= last(mloc)
                    Fθk[iθ, jk] = Gθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                # positive m
                mpos = kglob + 1
                if mpos in mloc
                    Fθk[iθ, jk] = Gθm[iθ, mpos]
                end
            else
                # negative m mirror if real_output: k = n - m
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        # assign conj of positive m (ensure both owners set same bin if overlap)
                        Fθk[iθ, jk] = conj(Gθm[iθ, mneg+1])
                    end
                end
            end
        end
    end
    # Inverse FFT along φ to produce (θ,φ) pencils
    pifft = PencilFFTs.plan_fft(Fθk; dims=2)  # assuming plan_fft handles inverse when applied with ifft
    fθφ = PencilFFTs.ifft(Fθk, pifft)
    return real_output ? real.(fθφ) : fθφ
end

"""
    dist_SHsphtor_to_spat(cfg, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; 
                    prototype_θφ::PencilArrays.PencilArray, real_output=true)

Prototype-based distributed vector synthesis (placeholder). Computes local dense result for now.
"""
function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::PencilArrays.PencilArray, 
                    Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    # Convert to internal normalization
    Slm_local = Array(Slm); Tlm_local = Array(Tlm)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_local); T2 = similar(Tlm_local)
        SHTnsKit.convert_alm_norm!(S2, Slm_local, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm_local, cfg; to_internal=true)
        Slm_local = S2; Tlm_local = T2
    end
    # (θ,m) spectra for Vt and Vp
    Vtθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vpθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    fill!(Vtθm, 0); fill!(Vpθm, 0)
    θloc = axes(Vtθm, 1); mloc = axes(Vtθm, 2)
    lloc = axes(Slm, 1)
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    for (ii,iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Vtθm, 1)[ii]
        xθ = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - xθ*xθ))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        for (jj,jm) in enumerate(mloc)
            iglobm = PencilArrays.globalindices(Vtθm, 2)[jj]
            mval = iglobm - 1
            col = mval + 1
            acc_t = 0.0 + 0.0im
            acc_p = 0.0 + 0.0im
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Slm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= max(1, mval)
                        N = cfg.Nlm[lval+1, col]
                        dθY = -sθ * N * tbld[lval+1, iglobθ]
                        Y = N * tblP[lval+1, iglobθ]
                        coeff = cfg.w[iglobθ] * cfg.cphi / (lval*(lval+1))
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_t += coeff * (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += coeff * ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[lval+1, iglobθ]) * Tl)
                    end
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, xθ, cfg.lmax, mval)
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Slm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= max(1, mval)
                        N = cfg.Nlm[lval+1, col]
                        dθY = -sθ * N * dPdx[lval+1]
                        Y = N * P[lval+1]
                        coeff = cfg.w[iglobθ] * cfg.cphi / (lval*(lval+1))
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_t += coeff * (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += coeff * ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[lval+1]) * Tl)
                    end
                end
            end
            Vtθm[iθ, jm] += acc_t
            Vpθm[iθ, jm] += acc_p
        end
    end
    # Reduce across l-pencil
    MPI.Allreduce!(Vtθm, +, PencilArrays.communicator(Vtθm))
    MPI.Allreduce!(Vpθm, +, PencilArrays.communicator(Vpθm))
    # Map to (θ,k) and inverse FFT
    Fθk_t = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    Fθk_p = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    fill!(Fθk_t, 0); fill!(Fθk_p, 0)
    nlon = cfg.nlon
    θloc = axes(Fθk_t, 1)
    kloc = axes(Fθk_t, 2)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(Fθk_t, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    Fθk_t[iθ, jk] = Vtθm[iθ, 1]
                    Fθk_p[iθ, jk] = Vpθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                mpos = kglob + 1
                if mpos in mloc
                    Fθk_t[iθ, jk] = Vtθm[iθ, mpos]
                    Fθk_p[iθ, jk] = Vpθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        Fθk_t[iθ, jk] = conj(Vtθm[iθ, mneg+1])
                        Fθk_p[iθ, jk] = conj(Vpθm[iθ, mneg+1])
                    end
                end
            end
        end
    end
    pifft_t = PencilFFTs.plan_fft(Fθk_t; dims=2)
    pifft_p = PencilFFTs.plan_fft(Fθk_p; dims=2)
    Vtθφ = PencilFFTs.ifft(Fθk_t, pifft_t)
    Vpθφ = PencilFFTs.ifft(Fθk_p, pifft_p)
    # Robert form scaling
    if cfg.robert_form
        θloc = axes(Vtθφ, 1)
        for (ii,iθ) in enumerate(θloc)
            iglobθ = PencilArrays.globalindices(Vtθφ, 1)[ii]
            sθ = sqrt(max(0.0, 1 - cfg.x[iglobθ]^2))
            Vtθφ[iθ, :] .*= sθ
            Vpθφ[iθ, :] .*= sθ
        end
    end
    return real_output ? (real.(Vtθφ), real.(Vpθφ)) : (Vtθφ, Vpθφ)
end

"""
    dist_SHqst_to_spat(cfg, Qlm::PencilArrays.PencilArray, Slm::PencilArrays.PencilArray, 
                        Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output=true)

Prototype-based distributed qst synthesis (placeholder). Computes local dense result for now.
"""
function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::PencilArrays.PencilArray, 
                                Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; 
                                prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)

    # Convert spectra to internal normalization if needed
    Qlm_local = Array(Qlm); Slm_local = Array(Slm); Tlm_local = Array(Tlm)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Q2 = similar(Qlm_local); S2 = similar(Slm_local); T2 = similar(Tlm_local)
        SHTnsKit.convert_alm_norm!(Q2, Qlm_local, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(S2, Slm_local, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm_local, cfg; to_internal=true)
        Qlm_local = Q2; Slm_local = S2; Tlm_local = T2
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    # Allocate (θ,m) pencils
    Vrθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vtθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vpθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    fill!(Vrθm, 0); fill!(Vtθm, 0); fill!(Vpθm, 0)
    θloc = axes(Vrθm, 1)
    mloc = axes(Vrθm, 2)
    lloc = axes(Qlm, 1)
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    for (ii,iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Vrθm, 1)[ii]
        xθ = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - xθ*xθ))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        wi = cfg.w[iglobθ]
        for (jj,jm) in enumerate(mloc)
            iglobm = PencilArrays.globalindices(Vrθm, 2)[jj]
            mval = iglobm - 1
            col = mval + 1
            acc_r = 0.0 + 0.0im
            acc_t = 0.0 + 0.0im
            acc_p = 0.0 + 0.0im
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Qlm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= mval
                        N = cfg.Nlm[lval+1, col]
                        Y = N * tblP[lval+1, iglobθ]
                        dθY = -sθ * N * tbld[lval+1, iglobθ]
                        coeff_r = wi * cfg.cphi
                        coeff_v = wi * cfg.cphi / (lval*(lval+1) == 0 ? 1 : (lval*(lval+1)))
                        Ql = Qlm_local[il, iglobm]
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_r += coeff_r * (Y * Ql)
                        acc_t += coeff_v * (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += coeff_v * ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[lval+1, iglobθ]) * Tl)
                    end
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, xθ, lmax, mval)
                @inbounds for (kk, il) in enumerate(lloc)
                    iglobl = PencilArrays.globalindices(Qlm, 1)[kk]
                    lval = iglobl - 1
                    if lval >= mval
                        N = cfg.Nlm[lval+1, col]
                        Y = N * P[lval+1]
                        dθY = -sθ * N * dPdx[lval+1]
                        coeff_r = wi * cfg.cphi
                        denom = lval*(lval+1)
                        coeff_v = wi * cfg.cphi / (denom == 0 ? 1 : denom)
                        Ql = Qlm_local[il, iglobm]
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_r += coeff_r * (Y * Ql)
                        acc_t += coeff_v * (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += coeff_v * ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[lval+1]) * Tl)
                    end
                end
            end
            Vrθm[iθ, jm] += acc_r
            Vtθm[iθ, jm] += acc_t
            Vpθm[iθ, jm] += acc_p
        end
    end
    # Reduce across l-pencil communicator
    MPI.Allreduce!(Vrθm, +, PencilArrays.communicator(Vrθm))
    MPI.Allreduce!(Vtθm, +, PencilArrays.communicator(Vtθm))
    MPI.Allreduce!(Vpθm, +, PencilArrays.communicator(Vpθm))
    # Map to (θ,k) and inverse FFT (enforce Hermitian if real)
    function θm_to_θφ!(Fθk, Vθm)
        fill!(Fθk, 0)
        θloc = axes(Fθk, 1); kloc = axes(Fθk, 2)
        mloc = axes(Vθm, 2)
        nlon = cfg.nlon
        for (ii,iθ) in enumerate(θloc)
            for (jj,jk) in enumerate(kloc)
                kglob = PencilArrays.globalindices(Fθk, 2)[jj] - 1
                if kglob == 0
                    if first(mloc) <= 1 <= last(mloc)
                        Fθk[iθ, jk] = Vθm[iθ, 1]
                    end
                elseif kglob <= mmax
                    mpos = kglob + 1
                    if mpos in mloc
                        Fθk[iθ, jk] = Vθm[iθ, mpos]
                    end
                else
                    if real_output
                        mneg = nlon - kglob
                        if 1 <= mneg <= mmax && (mneg+1) in mloc
                            Fθk[iθ, jk] = conj(Vθm[iθ, mneg+1])
                        end
                    end
                end
            end
        end
        return Fθk
    end
    Fθk_r = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    Fθk_t = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    Fθk_p = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    θm_to_θφ!(Fθk_r, Vrθm)
    θm_to_θφ!(Fθk_t, Vtθm)
    θm_to_θφ!(Fθk_p, Vpθm)
    # Inverse FFTs
    pr = PencilFFTs.plan_fft(Fθk_r; dims=2)
    pt = PencilFFTs.plan_fft(Fθk_t; dims=2)
    pp = PencilFFTs.plan_fft(Fθk_p; dims=2)
    Vrθφ = PencilFFTs.ifft(Fθk_r, pr)
    Vtθφ = PencilFFTs.ifft(Fθk_t, pt)
    Vpθφ = PencilFFTs.ifft(Fθk_p, pp)
    # Robert form on vector components
    if cfg.robert_form
        θloc = axes(Vtθφ, 1)
        for (ii,iθ) in enumerate(θloc)
            iglobθ = PencilArrays.globalindices(Vtθφ, 1)[ii]
            sθ = sqrt(max(0.0, 1 - cfg.x[iglobθ]^2))
            Vtθφ[iθ, :] .*= sθ
            Vpθφ[iθ, :] .*= sθ
        end
    end
    if real_output
        return real.(Vrθφ), real.(Vtθφ), real.(Vpθφ)
    else
        return Vrθφ, Vtθφ, Vpθφ
    end
end

"""
    dist_spat_to_SHsphtor(cfg, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)

Distributed vector analysis. Returns local dense Slm,Tlm matrices reduced across θ-pencil communicators.
For single rank, falls back to serial.
"""
function SHTnsKit.dist_spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, 
                                    Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)

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



"""
Minimal distributed plan to reuse allocated (θ,k) pencils and prototype metadata.
"""
struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArrays.PencilArray
    Gθm::PencilArrays.PencilArray
    Fθk::PencilArrays.PencilArray
    ifft_plan::Any
end

function DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray)
    Gθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    pifft = PencilFFTs.plan_fft(Fθk; dims=2)
    return DistPlan(cfg, prototype_θφ, Gθm, Fθk, pifft)
end

"""
    dist_synthesis!(plan::DistPlan, fθφ_out::PencilArrays.PencilArray, Alm::PencilArrays.PencilArray; real_output=true)

In-place-like wrapper that runs prototype-based dist_synthesis and writes into fθφ_out.
Current implementation delegates to dist_synthesis and copies; future versions will stream into plan.Fθk directly.
"""
function SHTnsKit.dist_synthesis!(plan::DistPlan, fθφ_out::PencilArrays.PencilArray, Alm::PencilArrays.PencilArray; real_output::Bool=true)
    cfg = plan.cfg
    fill!(plan.Gθm, 0)
    θrange = axes(plan.Gθm, 1)
    mrange = axes(plan.Gθm, 2)
    lr = axes(Alm, 1)
    Alm_loc = Array(Alm)
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    for (ii,iθ) in enumerate(θrange)
        iglobθ = PencilArrays.globalindices(plan.Gθm, 1)[ii]
        x = cfg.x[iglobθ]
        for (jj,jm) in enumerate(mrange)
            iglobm = PencilArrays.globalindices(plan.Gθm, 2)[jj]
            mval = iglobm - 1
            col = mval + 1
            acc = 0.0 + 0.0im
            if use_tbl
                tbl = cfg.plm_tables[col]
                for (kk, il) in enumerate(lr)
                    igl = PencilArrays.globalindices(Alm, 1)[kk]
                    lval = igl - 1
                    if lval >= mval
                        a = Alm_loc[il, iglobm]
                        if cfg.norm !== :orthonormal || cfg.cs_phase == false
                            k = SHTnsKit.norm_scale_from_orthonormal(lval, mval, cfg.norm)
                            α = SHTnsKit.cs_phase_factor(mval, true, cfg.cs_phase)
                            a *= (k * α)
                        end
                        acc += (cfg.Nlm[lval+1, col] * tbl[lval+1, iglobθ]) * a
                    end
                end
            else
                SHTnsKit.Plm_row!(P, x, cfg.lmax, mval)
                for (kk, il) in enumerate(lr)
                    igl = PencilArrays.globalindices(Alm, 1)[kk]
                    lval = igl - 1
                    if lval >= mval
                        a = Alm_loc[il, iglobm]
                        if cfg.norm !== :orthonormal || cfg.cs_phase == false
                            k = SHTnsKit.norm_scale_from_orthonormal(lval, mval, cfg.norm)
                            α = SHTnsKit.cs_phase_factor(mval, true, cfg.cs_phase)
                            a *= (k * α)
                        end
                        acc += (cfg.Nlm[lval+1, col] * P[lval+1]) * a
                    end
                end
            end
            plan.Gθm[iθ, jm] = cfg.w[iglobθ] * cfg.cphi * acc
        end
    end
    MPI.Allreduce!(plan.Gθm, +, PencilArrays.communicator(plan.Gθm))
    fill!(plan.Fθk, 0)
    θloc = axes(plan.Fθk, 1)
    kloc = axes(plan.Fθk, 2)
    mloc = axes(plan.Gθm, 2)
    nlon = cfg.nlon
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(plan.Fθk, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    plan.Fθk[iθ, jk] = plan.Gθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                mpos = kglob + 1
                if mpos in mloc
                    plan.Fθk[iθ, jk] = plan.Gθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        plan.Fθk[iθ, jk] = conj(plan.Gθm[iθ, mneg+1])
                    end
                end
            end
        end
    end
    fθφ_tmp = PencilFFTs.ifft(plan.Fθk, plan.ifft_plan)
    if real_output
        for I in eachindex(fθφ_tmp)
            fθφ_out[I] = real(fθφ_tmp[I])
        end
    else
        for I in eachindex(fθφ_tmp)
            fθφ_out[I] = fθφ_tmp[I]
        end
    end
    return fθφ_out
end

end # module
