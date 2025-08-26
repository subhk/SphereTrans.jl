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
Plan for distributed 3D (q,s,t) transforms: radial scalar Qlm and tangential Slm/Tlm.
Reuses (θ,m) accumulators for all three components and one (θ,k) buffer for ifft.
"""
struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    Vrθm::PencilArrays.PencilArray
    Vtθm::PencilArrays.PencilArray
    Vpθm::PencilArrays.PencilArray
    Fθk::PencilArrays.PencilArray
    pfft::Any
    pifft::Any
    P::Vector{Float64}
    dPdx::Vector{Float64}
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray)
    Vrθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vtθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vpθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    pfft = PencilFFTs.plan_fft(Fθk; dims=2)
    pifft = PencilFFTs.plan_fft(Fθk; dims=2)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    return DistQstPlan(cfg, Vrθm, Vtθm, Vpθm, Fθk, pfft, pifft, P, dPdx)
end

"""
    dist_spat_to_SHqst!(plan::DistQstPlan, Qlm_out, Slm_out, Tlm_out, Vrθφ, Vtθφ, Vpθφ; use_tables=plan.cfg.use_plm_tables)

In-place distributed QST analysis.
"""
function SHTnsKit.dist_spat_to_SHqst!(plan::DistQstPlan, Qlm_out::AbstractMatrix, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                      Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray;
                                      use_tables=plan.cfg.use_plm_tables)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Qlm_out,1)==lmax+1 && size(Qlm_out,2)==mmax+1 || throw(DimensionMismatch("Qlm_out dims"))
    size(Slm_out,1)==lmax+1 && size(Slm_out,2)==mmax+1 || throw(DimensionMismatch("Slm_out dims"))
    size(Tlm_out,1)==lmax+1 && size(Tlm_out,2)==mmax+1 || throw(DimensionMismatch("Tlm_out dims"))
    fill!(Qlm_out, 0); fill!(Slm_out, 0); fill!(Tlm_out, 0)
    # FFT all three and transpose to (θ,m)
    Fθm_r = PencilArrays.transpose(PencilFFTs.fft(Vrθφ, plan.pfft), (; dims=(1,2), names=(:θ,:m)))
    Fθm_t = PencilArrays.transpose(PencilFFTs.fft(Vtθφ, plan.pfft), (; dims=(1,2), names=(:θ,:m)))
    Fθm_p = PencilArrays.transpose(PencilFFTs.fft(Vpθφ, plan.pfft), (; dims=(1,2), names=(:θ,:m)))
    θrange = axes(Fθm_r, 1)
    mrange = axes(Fθm_r, 2)
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    for m in mrange
        mm = m - first(mrange)
        mglob = PencilArrays.globalindices(Fθm_r, 2)[mm+1]
        mval = mglob - 1
        mval > mmax && continue
        col = mval + 1
        for (ii, iθ) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm_r, 1)[ii]
            x = cfg.x[iglob]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Fr = Fθm_r[iθ, m]
            Ft = Fθm_t[iθ, m]
            Fp = Fθm_p[iθ, m]
            if cfg.robert_form && sθ > 0
                Ft /= sθ; Fp /= sθ
            end
            wi = cfg.w[iglob]
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in mval:lmax
                    N = cfg.Nlm[l+1, col]
                    Y = N * tblP[l+1, iglob]
                    dθY = -sθ * N * tbld[l+1, iglob]
                    coeffs = wi * cfg.cphi
                    Qlm_out[l+1, col] += coeffs * (Y * Fr)
                end
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    Y = N * tblP[l+1, iglob]
                    dθY = -sθ * N * tbld[l+1, iglob]
                    coeffv = wi * cfg.cphi / (l*(l+1))
                    Slm_out[l+1, col] += coeffv * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_out[l+1, col] += coeffv * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * tbld[l+1, iglob]))
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(plan.P, plan.dPdx, x, lmax, mval)
                @inbounds for l in mval:lmax
                    N = cfg.Nlm[l+1, col]
                    Y = N * plan.P[l+1]
                    Qlm_out[l+1, col] += wi * cfg.cphi * (Y * Fr)
                end
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * plan.dPdx[l+1]
                    Y = N * plan.P[l+1]
                    coeffv = wi * cfg.cphi / (l*(l+1))
                    Slm_out[l+1, col] += coeffv * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_out[l+1, col] += coeffv * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * plan.dPdx[l+1]))
                end
            end
        end
    end
    # Reduce across ranks
    c = PencilArrays.communicator(Vrθφ)
    MPI.Allreduce!(Qlm_out, +, c)
    MPI.Allreduce!(Slm_out, +, c)
    MPI.Allreduce!(Tlm_out, +, c)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Q2 = similar(Qlm_out); S2 = similar(Slm_out); T2 = similar(Tlm_out)
        SHTnsKit.convert_alm_norm!(Q2, Qlm_out, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(S2, Slm_out, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(T2, Tlm_out, cfg; to_internal=false)
        copyto!(Qlm_out, Q2); copyto!(Slm_out, S2); copyto!(Tlm_out, T2)
    end
    return Qlm_out, Slm_out, Tlm_out
end

"""
    dist_SHqst_to_spat!(plan::DistQstPlan, Vrθφ_out, Vtθφ_out, Vpθφ_out, Qlm, Slm, Tlm; real_output=true)

In-place distributed QST synthesis.
"""
function SHTnsKit.dist_SHqst_to_spat!(plan::DistQstPlan, Vrθφ_out::PencilArrays.PencilArray, Vtθφ_out::PencilArrays.PencilArray, Vpθφ_out::PencilArrays.PencilArray,
                                      Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Qlm,1)==lmax+1 && size(Qlm,2)==mmax+1 || throw(DimensionMismatch("Qlm dims"))
    size(Slm,1)==lmax+1 && size(Slm,2)==mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1)==lmax+1 && size(Tlm,2)==mmax+1 || throw(DimensionMismatch("Tlm dims"))
    # Convert to internal norm if needed
    Qlm_i, Slm_i, Tlm_i = Qlm, Slm, Tlm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Q2 = similar(Qlm); S2 = similar(Slm); T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(Q2, Qlm, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Qlm_i, Slm_i, Tlm_i = Q2, S2, T2
    end
    # Build (θ,m) accumulators locally
    fill!(plan.Vrθm, 0); fill!(plan.Vtθm, 0); fill!(plan.Vpθm, 0)
    θloc = axes(plan.Vrθm, 1)
    mloc = axes(plan.Vrθm, 2)
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    for (ii,iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(plan.Vrθm, 1)[ii]
        xθ = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - xθ*xθ))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        for (jj,jm) in enumerate(mloc)
            iglobm = PencilArrays.globalindices(plan.Vrθm, 2)[jj]
            mval = iglobm - 1
            mval > mmax && continue
            col = mval + 1
            acc_r = 0.0 + 0.0im
            acc_t = 0.0 + 0.0im
            acc_p = 0.0 + 0.0im
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in mval:lmax
                    N = cfg.Nlm[l+1, col]
                    Y = N * tblP[l+1, iglobθ]
                    acc_r += Y * Qlm_i[l+1, col]
                end
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    acc_t += dθY * Slm_i[l+1, col] + (0 + 1im) * mval * inv_sθ * Y * Tlm_i[l+1, col]
                    acc_p += (0 + 1im) * mval * inv_sθ * Y * Slm_i[l+1, col] + (sθ * N * tbld[l+1, iglobθ]) * Tlm_i[l+1, col]
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(plan.P, plan.dPdx, xθ, lmax, mval)
                @inbounds for l in mval:lmax
                    N = cfg.Nlm[l+1, col]
                    Y = N * plan.P[l+1]
                    acc_r += Y * Qlm_i[l+1, col]
                end
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * plan.dPdx[l+1]
                    Y = N * plan.P[l+1]
                    acc_t += dθY * Slm_i[l+1, col] + (0 + 1im) * mval * inv_sθ * Y * Tlm_i[l+1, col]
                    acc_p += (0 + 1im) * mval * inv_sθ * Y * Slm_i[l+1, col] + (sθ * N * plan.dPdx[l+1]) * Tlm_i[l+1, col]
                end
            end
            plan.Vrθm[iθ, jm] = acc_r
            plan.Vtθm[iθ, jm] = acc_t
            plan.Vpθm[iθ, jm] = acc_p
        end
    end
    # Reduce across ranks
    comm = PencilArrays.communicator(plan.Vrθm)
    MPI.Allreduce!(plan.Vrθm, +, comm)
    MPI.Allreduce!(plan.Vtθm, +, comm)
    MPI.Allreduce!(plan.Vpθm, +, comm)
    # Map to (θ,k), apply inv_scaleφ, iFFT for each component
    inv_scaleφ = cfg.nlon / (2π)
    θloc = axes(plan.Fθk, 1); kloc = axes(plan.Fθk, 2)
    mloc = axes(plan.Vrθm, 2)
    nlon = cfg.nlon
    # Helper to place and transform one component
    function place_ifft!(Fθk, Vθm)
        fill!(Fθk, 0)
        for (ii,iθ) in enumerate(θloc)
            for (jj,jk) in enumerate(kloc)
                kglob = PencilArrays.globalindices(Fθk, 2)[jj] - 1
                if kglob == 0
                    if first(mloc) <= 1 <= last(mloc)
                        Fθk[iθ, jk] = inv_scaleφ * Vθm[iθ, 1]
                    end
                elseif kglob <= mmax
                    mpos = kglob + 1
                    if mpos in mloc
                        Fθk[iθ, jk] = inv_scaleφ * Vθm[iθ, mpos]
                    end
                else
                    if real_output
                        mneg = nlon - kglob
                        if 1 <= mneg <= mmax && (mneg+1) in mloc
                            Fθk[iθ, jk] = conj(inv_scaleφ * Vθm[iθ, mneg+1])
                        end
                    end
                end
            end
        end
        return PencilFFTs.ifft(Fθk, plan.pifft)
    end
    Vrθφ_tmp = place_ifft!(plan.Fθk, plan.Vrθm)
    Vtθφ_tmp = place_ifft!(plan.Fθk, plan.Vtθm)
    Vpθφ_tmp = place_ifft!(plan.Fθk, plan.Vpθm)
    # Apply robert form and write to outputs
    if cfg.robert_form
        θl = axes(Vtθφ_tmp, 1)
        for (ii,iθ) in enumerate(θl)
            iglobθ = PencilArrays.globalindices(Vtθφ_tmp, 1)[ii]
            sθ = sqrt(max(0.0, 1 - cfg.x[iglobθ]^2))
            Vtθφ_tmp[iθ, :] .*= sθ
            Vpθφ_tmp[iθ, :] .*= sθ
        end
    end
    if real_output
        for I in eachindex(Vrθφ_out)
            Vrθφ_out[I] = real(Vrθφ_tmp[I])
        end
        for I in eachindex(Vtθφ_out)
            Vtθφ_out[I] = real(Vtθφ_tmp[I])
        end
        for I in eachindex(Vpθφ_out)
            Vpθφ_out[I] = real(Vpθφ_tmp[I])
        end
    else
        for I in eachindex(Vrθφ_out)
            Vrθφ_out[I] = Vrθφ_tmp[I]
        end
        for I in eachindex(Vtθφ_out)
            Vtθφ_out[I] = Vtθφ_tmp[I]
        end
        for I in eachindex(Vpθφ_out)
            Vpθφ_out[I] = Vpθφ_tmp[I]
        end
    end
    return Vrθφ_out, Vtθφ_out, Vpθφ_out
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
            # No Gauss weights here (synthesis); defer φ scaling to placement step.
            Gθm[iθ, jm] += acc
        end
    end
    # Reduce across l-pencil communicator (sum partials)
    MPI.Allreduce!(Gθm, +, PencilArrays.communicator(Gθm))
    # Map to (θ,k)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    fill!(Fθk, 0)
    nlon = cfg.nlon
    inv_scaleφ = nlon / (2π)
    θloc = axes(Fθk, 1)
    kloc = axes(Fθk, 2)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(Fθk, 2)[jj] - 1
            if kglob == 0
                # m=0 bin
                if first(mloc) <= 1 <= last(mloc)
                    Fθk[iθ, jk] = inv_scaleφ * Gθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                # positive m
                mpos = kglob + 1
                if mpos in mloc
                    Fθk[iθ, jk] = inv_scaleφ * Gθm[iθ, mpos]
                end
            else
                # negative m mirror if real_output: k = n - m
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        # assign conj of positive m (ensure both owners set same bin if overlap)
                        Fθk[iθ, jk] = conj(inv_scaleφ * Gθm[iθ, mneg+1])
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
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_t += (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[lval+1, iglobθ]) * Tl)
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
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_t += (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[lval+1]) * Tl)
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
    inv_scaleφ = nlon / (2π)
    θloc = axes(Fθk_t, 1)
    kloc = axes(Fθk_t, 2)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(Fθk_t, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    Fθk_t[iθ, jk] = inv_scaleφ * Vtθm[iθ, 1]
                    Fθk_p[iθ, jk] = inv_scaleφ * Vpθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                mpos = kglob + 1
                if mpos in mloc
                    Fθk_t[iθ, jk] = inv_scaleφ * Vtθm[iθ, mpos]
                    Fθk_p[iθ, jk] = inv_scaleφ * Vpθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        Fθk_t[iθ, jk] = conj(inv_scaleφ * Vtθm[iθ, mneg+1])
                        Fθk_p[iθ, jk] = conj(inv_scaleφ * Vpθm[iθ, mneg+1])
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
                        Ql = Qlm_local[il, iglobm]
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_r += (Y * Ql)
                        acc_t += (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[lval+1, iglobθ]) * Tl)
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
                        Ql = Qlm_local[il, iglobm]
                        Sl = Slm_local[il, iglobm]
                        Tl = Tlm_local[il, iglobm]
                        acc_r += (Y * Ql)
                        acc_t += (dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl)
                        acc_p += ((0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[lval+1]) * Tl)
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
        inv_scaleφ = nlon / (2π)
        for (ii,iθ) in enumerate(θloc)
            for (jj,jk) in enumerate(kloc)
                kglob = PencilArrays.globalindices(Fθk, 2)[jj] - 1
                if kglob == 0
                    if first(mloc) <= 1 <= last(mloc)
                        Fθk[iθ, jk] = inv_scaleφ * Vθm[iθ, 1]
                    end
                elseif kglob <= mmax
                    mpos = kglob + 1
                    if mpos in mloc
                        Fθk[iθ, jk] = inv_scaleφ * Vθm[iθ, mpos]
                    end
                else
                    if real_output
                        mneg = nlon - kglob
                        if 1 <= mneg <= mmax && (mneg+1) in mloc
                            Fθk[iθ, jk] = conj(inv_scaleφ * Vθm[iθ, mneg+1])
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
    # FFT along φ (no persistent plan here; initial version uses alloc)
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
        if mval > mmax
            continue
        end
        col = mval + 1
        for (ii,i) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm_t, 1)[ii]
            x = cfg.x[iglob]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Ft = Fθm_t[i, m]
            Fp = Fθm_p[i, m]
            wi = cfg.w[iglob]
            # Undo Robert form scaling for analysis if enabled
            if cfg.robert_form && sθ > 0
                Ft /= sθ
                Fp /= sθ
            end
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
            # No Gauss weights during synthesis; store raw Gθm
            plan.Gθm[iθ, jm] = acc
        end
    end
    MPI.Allreduce!(plan.Gθm, +, PencilArrays.communicator(plan.Gθm))
    fill!(plan.Fθk, 0)
    θloc = axes(plan.Fθk, 1)
    kloc = axes(plan.Fθk, 2)
    mloc = axes(plan.Gθm, 2)
    nlon = cfg.nlon
    inv_scaleφ = nlon / (2π)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(plan.Fθk, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Gθm[iθ, 1]
                end
            elseif kglob <= cfg.mmax
                mpos = kglob + 1
                if mpos in mloc
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Gθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= cfg.mmax && (mneg+1) in mloc
                        plan.Fθk[iθ, jk] = conj(inv_scaleφ * plan.Gθm[iθ, mneg+1])
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
"""
Minimal distributed analysis plan to reuse FFT plan and (θ,k) buffer.
Reuses caller-provided Alm output to avoid temporary matrix allocations.
"""
struct DistAnalysisPlan
    cfg::SHTnsKit.SHTConfig
    Fθk::PencilArrays.PencilArray
    pfft::Any
    P::Vector{Float64}
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    pfft = PencilFFTs.plan_fft(Fθk; dims=2)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    return DistAnalysisPlan(cfg, Fθk, pfft, P)
end

"""
Plan for distributed vector (spheroidal/toroidal) analysis and synthesis.
Reuses (θ,k) buffer and (θ,m) accumulators to avoid per-call allocations.
"""
struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    Vtθm::PencilArrays.PencilArray
    Vpθm::PencilArrays.PencilArray
    Fθk::PencilArrays.PencilArray
    pfft::Any
    pifft::Any
    P::Vector{Float64}
    dPdx::Vector{Float64}
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray)
    Vtθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Vpθm = PencilArrays.allocate(prototype_θφ; dims=(:θ, :m), eltype=ComplexF64)
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ, :k), eltype=ComplexF64)
    pfft = PencilFFTs.plan_fft(Fθk; dims=2)
    pifft = PencilFFTs.plan_fft(Fθk; dims=2)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    return DistSphtorPlan(cfg, Vtθm, Vpθm, Fθk, pfft, pifft, P, dPdx)
end

"""
    dist_spat_to_SHsphtor!(plan::DistSphtorPlan, Slm_out, Tlm_out, Vtθφ, Vpθφ; use_tables=plan.cfg.use_plm_tables)

In-place distributed vector analysis. Writes into preallocated Slm_out, Tlm_out (lmax+1, mmax+1).
"""
function SHTnsKit.dist_spat_to_SHsphtor!(plan::DistSphtorPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                         Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray;
                                         use_tables=plan.cfg.use_plm_tables)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm_out,1)==lmax+1 && size(Slm_out,2)==mmax+1 || throw(DimensionMismatch("Slm_out dims"))
    size(Tlm_out,1)==lmax+1 && size(Tlm_out,2)==mmax+1 || throw(DimensionMismatch("Tlm_out dims"))
    fill!(Slm_out, 0); fill!(Tlm_out, 0)
    # FFT along φ then transpose to (θ,m)
    Fθk_t = PencilFFTs.fft(Vtθφ, plan.pfft)
    Fθm_t = PencilArrays.transpose(Fθk_t, (; dims=(1,2), names=(:θ,:m)))
    Fθk_p = PencilFFTs.fft(Vpθφ, plan.pfft)
    Fθm_p = PencilArrays.transpose(Fθk_p, (; dims=(1,2), names=(:θ,:m)))
    θrange = axes(Fθm_t, 1)
    mrange = axes(Fθm_t, 2)
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    for m in mrange
        mm = m - first(mrange)
        mglob = PencilArrays.globalindices(Fθm_t, 2)[mm+1]
        mval = mglob - 1
        mval > mmax && continue
        col = mval + 1
        for (ii, iθ) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm_t, 1)[ii]
            x = cfg.x[iglob]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Ft = Fθm_t[iθ, m]
            Fp = Fθm_p[iθ, m]
            if cfg.robert_form && sθ > 0
                Ft /= sθ; Fp /= sθ
            end
            wi = cfg.w[iglob]
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglob]
                    Y = N * tblP[l+1, iglob]
                    coeff = wi * cfg.cphi / (l*(l+1))
                    Slm_out[l+1, col] += coeff * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_out[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * tbld[l+1, iglob]))
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(plan.P, plan.dPdx, x, lmax, mval)
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * plan.dPdx[l+1]
                    Y = N * plan.P[l+1]
                    coeff = wi * cfg.cphi / (l*(l+1))
                    Slm_out[l+1, col] += coeff * (Ft * dθY - (0 + 1im) * mval * inv_sθ * Y * Fp)
                    Tlm_out[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Ft + Fp * (+sθ * N * plan.dPdx[l+1]))
                end
            end
        end
    end
    MPI.Allreduce!(Slm_out, +, PencilArrays.communicator(Vtθφ))
    MPI.Allreduce!(Tlm_out, +, PencilArrays.communicator(Vtθφ))
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_out); T2 = similar(Tlm_out)
        SHTnsKit.convert_alm_norm!(S2, Slm_out, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(T2, Tlm_out, cfg; to_internal=false)
        copyto!(Slm_out, S2); copyto!(Tlm_out, T2)
    end
    return Slm_out, Tlm_out
end

"""
    dist_SHsphtor_to_spat!(plan::DistSphtorPlan, Vtθφ_out, Vpθφ_out, Slm, Tlm; real_output=true)

In-place distributed vector synthesis. Writes into Vtθφ_out, Vpθφ_out (pencils like prototype used to build the plan).
"""
function SHTnsKit.dist_SHsphtor_to_spat!(plan::DistSphtorPlan, Vtθφ_out::PencilArrays.PencilArray, Vpθφ_out::PencilArrays.PencilArray,
                                         Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1)==lmax+1 && size(Slm,2)==mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1)==lmax+1 && size(Tlm,2)==mmax+1 || throw(DimensionMismatch("Tlm dims"))
    # Convert to internal normalization if needed
    Slm_int = Slm; Tlm_int = Tlm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Slm_int, Tlm_int = S2, T2
    end
    # Build (θ,m) accumulators locally from our l-slice
    fill!(plan.Vtθm, 0); fill!(plan.Vpθm, 0)
    θloc = axes(plan.Vtθm, 1)
    mloc = axes(plan.Vtθm, 2)
    use_tbl = cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    for (ii,iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(plan.Vtθm, 1)[ii]
        xθ = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - xθ*xθ))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        for (jj,jm) in enumerate(mloc)
            iglobm = PencilArrays.globalindices(plan.Vtθm, 2)[jj]
            mval = iglobm - 1
            mval > mmax && continue
            col = mval + 1
            acc_t = 0.0 + 0.0im
            acc_p = 0.0 + 0.0im
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    acc_t += dθY * Slm_int[l+1, col] + (0 + 1im) * mval * inv_sθ * Y * Tlm_int[l+1, col]
                    acc_p += (0 + 1im) * mval * inv_sθ * Y * Slm_int[l+1, col] + (sθ * N * tbld[l+1, iglobθ]) * Tlm_int[l+1, col]
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(plan.P, plan.dPdx, xθ, lmax, mval)
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * plan.dPdx[l+1]
                    Y = N * plan.P[l+1]
                    acc_t += dθY * Slm_int[l+1, col] + (0 + 1im) * mval * inv_sθ * Y * Tlm_int[l+1, col]
                    acc_p += (0 + 1im) * mval * inv_sθ * Y * Slm_int[l+1, col] + (sθ * N * plan.dPdx[l+1]) * Tlm_int[l+1, col]
                end
            end
            plan.Vtθm[iθ, jm] = acc_t
            plan.Vpθm[iθ, jm] = acc_p
        end
    end
    # Reduce across l-pencil
    MPI.Allreduce!(plan.Vtθm, +, PencilArrays.communicator(plan.Vtθm))
    MPI.Allreduce!(plan.Vpθm, +, PencilArrays.communicator(plan.Vpθm))
    # Map to (θ,k) with inv_scaleφ and iFFT
    fill!(plan.Fθk, 0)
    inv_scaleφ = cfg.nlon / (2π)
    θloc = axes(plan.Fθk, 1)
    kloc = axes(plan.Fθk, 2)
    mloc = axes(plan.Vtθm, 2)
    nlon = cfg.nlon
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(plan.Fθk, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Vtθm[iθ, 1]
                end
            elseif kglob <= mmax
                mpos = kglob + 1
                if mpos in mloc
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Vtθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= mmax && (mneg+1) in mloc
                        plan.Fθk[iθ, jk] = conj(inv_scaleφ * plan.Vtθm[iθ, mneg+1])
                    end
                end
            end
        end
    end
    Vtθφ_tmp = PencilFFTs.ifft(plan.Fθk, plan.pifft)
    # Reuse Fθk for Vp
    fill!(plan.Fθk, 0)
    for (ii,iθ) in enumerate(θloc)
        for (jj,jk) in enumerate(kloc)
            kglob = PencilArrays.globalindices(plan.Fθk, 2)[jj] - 1
            if kglob == 0
                if first(mloc) <= 1 <= last(mloc)
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Vpθm[iθ, 1]
                end
            elseif kglob <= mmax
                mpos = kglob + 1
                if mpos in mloc
                    plan.Fθk[iθ, jk] = inv_scaleφ * plan.Vpθm[iθ, mpos]
                end
            else
                if real_output
                    mneg = nlon - kglob
                    if 1 <= mneg <= mmax && (mneg+1) in mloc
                        plan.Fθk[iθ, jk] = conj(inv_scaleφ * plan.Vpθm[iθ, mneg+1])
                    end
                end
            end
        end
    end
    Vpθφ_tmp = PencilFFTs.ifft(plan.Fθk, plan.pifft)
    # Robert form scaling
    if cfg.robert_form
        θl = axes(Vtθφ_tmp, 1)
        for (ii,iθ) in enumerate(θl)
            iglobθ = PencilArrays.globalindices(Vtθφ_tmp, 1)[ii]
            sθ = sqrt(max(0.0, 1 - cfg.x[iglobθ]^2))
            Vtθφ_tmp[iθ, :] .*= sθ
            Vpθφ_tmp[iθ, :] .*= sθ
        end
    end
    if real_output
        for I in eachindex(Vtθφ_out)
            Vtθφ_out[I] = real(Vtθφ_tmp[I])
        end
        for I in eachindex(Vpθφ_out)
            Vpθφ_out[I] = real(Vpθφ_tmp[I])
        end
    else
        for I in eachindex(Vtθφ_out)
            Vtθφ_out[I] = Vtθφ_tmp[I]
        end
        for I in eachindex(Vpθφ_out)
            Vpθφ_out[I] = Vpθφ_tmp[I]
        end
    end
    return Vtθφ_out, Vpθφ_out
end

"""
    dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)

In-place distributed scalar analysis. Writes coefficients into Alm_out (lmax+1, mmax+1).
Avoids internal Alm allocations and reuses FFT plan and buffers.
"""
function SHTnsKit.dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm_out,1) == lmax+1 || throw(DimensionMismatch("Alm_out rows must be lmax+1"))
    size(Alm_out,2) == mmax+1 || throw(DimensionMismatch("Alm_out cols must be mmax+1"))
    # 1) FFT along φ into (θ,k)
    Fθk_local = PencilFFTs.fft(fθφ, plan.pfft)
    # 2) Transpose to (θ,m)
    Fθm = PencilArrays.transpose(Fθk_local, (; dims=(1,2), names=(:θ,:m)))
    # 3) Accumulate contributions into Alm_out
    fill!(Alm_out, 0)
    θrange = axes(Fθm, 1)
    mrange = axes(Fθm, 2)
    use_tbls = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    for m in mrange
        mm = m - first(mrange)
        mglob = PencilArrays.globalindices(Fθm, 2)[mm+1]
        mval = mglob - 1
        mval > mmax && continue
        col = mval + 1
        for (ii, iθ) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm, 1)[ii]
            Fi = Fθm[iθ, m]
            wi = cfg.w[iglob]
            if use_tbls
                tbl = cfg.plm_tables[col]
                @inbounds for l in mval:lmax
                    Alm_out[l+1, col] += (wi * tbl[l+1, iglob]) * Fi
                end
            else
                SHTnsKit.Plm_row!(plan.P, cfg.x[iglob], lmax, mval)
                @inbounds for l in mval:lmax
                    Alm_out[l+1, col] += (wi * plan.P[l+1]) * Fi
                end
            end
        end
    end
    # 4) Global reduction over θ-pencil
    MPI.Allreduce!(Alm_out, +, PencilArrays.communicator(fθφ))
    # 5) Apply normalization and φ scaling
    scaleφ = cfg.cphi
    @inbounds for m in 0:mmax, l in m:lmax
        Alm_out[l+1, m+1] *= cfg.Nlm[l+1, m+1] * scaleφ
    end
    return Alm_out
end
