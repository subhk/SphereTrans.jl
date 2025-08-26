##########
# Distributed transforms using PencilFFTs/PencilArrays (scalar) and safe fallbacks for vector/QST
##########

function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false)
    comm = PencilArrays.communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Choose FFT path
    if use_rfft && eltype(fθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, fθφ)
        Fθk = PencilFFTs.rfft(fθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, fθφ)
        Fθk = PencilFFTs.fft(fθφ, pfft)
    end
    Fθm = PencilArrays.transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    Alm_local = zeros(ComplexF64, lmax+1, mmax+1)
    θrange = axes(Fθm, 1); mrange = axes(Fθm, 2)
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, lmax + 1)
    for m in mrange
        mm = m - first(mrange)
        mglob = PencilArrays.globalindices(Fθm, 2)[mm+1]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        for (ii,iθ) in enumerate(θrange)
            iglob = PencilArrays.globalindices(Fθm, 1)[ii]
            Fi = Fθm[iθ, m]
            wi = cfg.w[iglob]
            if use_tbl
                tblcol = view(cfg.plm_tables[col], :, iglob)
                @inbounds for l in mval:lmax
                    Alm_local[l+1, col] += wi * tblcol[l+1] * Fi
                end
            else
                SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                @inbounds for l in mval:lmax
                    Alm_local[l+1, col] += wi * P[l+1] * Fi
                end
            end
        end
    end
    MPI.Allreduce!(Alm_local, +, comm)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm_local[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
    end
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_out = similar(Alm_local)
        SHTnsKit.convert_alm_norm!(Alm_out, Alm_local, cfg; to_internal=false)
        return Alm_out
    else
        return Alm_local
    end
end

function SHTnsKit.dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)
    Alm = SHTnsKit.dist_analysis(plan.cfg, fθφ; use_tables, use_rfft=plan.use_rfft)
    copyto!(Alm_out, Alm)
    return Alm_out
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
    fill!(Fθk, 0)
    θloc = axes(Fθk, 1); kloc = axes(Fθk, 2)
    mloc = axes(PencilArrays.allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)
    nlon = cfg.nlon
    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{ComplexF64}(undef, length(θloc))
    for (jj, jm) in enumerate(mloc)
        mglob = PencilArrays.globalindices(PencilArrays.allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)[jj]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        if cfg.use_plm_tables && !isempty(cfg.plm_tables)
            tbl = cfg.plm_tables[col]
            for (ii,iθ) in enumerate(θloc)
                g = 0.0 + 0.0im
                iglobθ = PencilArrays.globalindices(Fθk, 1)[ii]
                @inbounds for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * tbl[l+1, iglobθ]) * Alm[l+1, col]
                end
                G[ii] = g
            end
        else
            for (ii,iθ) in enumerate(θloc)
                iglobθ = PencilArrays.globalindices(Fθk, 1)[ii]
                SHTnsKit.Plm_row!(P, cfg.x[iglobθ], lmax, mval)
                g = 0.0 + 0.0im
                @inbounds for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * P[l+1]) * Alm[l+1, col]
                end
                G[ii] = g
            end
        end
        inv_scaleφ = nlon / (2π)
        for (ii,iθ) in enumerate(θloc)
            Fθk[iθ, col] = inv_scaleφ * G[ii]
        end
        if real_output && mval > 0
            conj_index = nlon - mval + 1
            for (ii,iθ) in enumerate(θloc)
                Fθk[iθ, conj_index] = conj(Fθk[iθ, col])
            end
        end
    end
    pifft = PencilFFTs.plan_fft(Fθk; dims=2)
    fθφ = PencilFFTs.ifft(Fθk, pifft)
    return real_output ? real.(fθφ) : fθφ
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_synthesis(cfg, Array(Alm); prototype_θφ, real_output)
end

function SHTnsKit.dist_synthesis!(plan::DistPlan, fθφ_out::PencilArrays.PencilArray, Alm::PencilArrays.PencilArray; real_output::Bool=true)
    f = SHTnsKit.dist_synthesis(plan.cfg, Alm; prototype_θφ=plan.prototype_θφ, real_output)
    copyto!(fθφ_out, f)
    return fθφ_out
end

## Vector/QST distributed implementations

# Distributed vector analysis (spheroidal/toroidal)
function SHTnsKit.dist_spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false)
    comm = PencilArrays.communicator(Vtθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Choose FFT path per input eltype
    if use_rfft && eltype(Vtθφ) <: Real && eltype(Vpθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, Vtθφ)
        Ftθk = PencilFFTs.rfft(Vtθφ, pfft)
        Fpθk = PencilFFTs.rfft(Vpθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, Vtθφ)
        Ftθk = PencilFFTs.fft(Vtθφ, pfft)
        Fpθk = PencilFFTs.fft(Vpθφ, pfft)
    end
    Ftθm = PencilArrays.transpose(Ftθk, (; dims=(1,2), names=(:θ,:m)))
    Fpθm = PencilArrays.transpose(Fpθk, (; dims=(1,2), names=(:θ,:m)))

    Slm_local = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm_local = zeros(ComplexF64, lmax+1, mmax+1)

    θrange = axes(Ftθm, 1); mrange = axes(Ftθm, 2)
    gl_θ = PencilArrays.globalindices(Ftθm, 1)
    gl_m = PencilArrays.globalindices(Ftθm, 2)

    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi

    @inbounds for (jj, jm) in enumerate(mrange)
        mglob = gl_m[jj]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        for (ii, iθ) in enumerate(θrange)
            iglobθ = gl_θ[ii]
            x = cfg.x[iglobθ]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Fθ_i = Ftθm[iθ, jm]
            Fφ_i = Fpθm[iθ, jm]
            if cfg.robert_form && sθ > 0
                Fθ_i /= sθ
                Fφ_i /= sθ
            end
            wi = cfg.w[iglobθ]
            if use_tbl
                tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    coeff = wi * scaleφ / (l*(l+1))
                    Slm_local[l+1, col] += coeff * (Fθ_i * dθY - (0 + 1im) * mval * inv_sθ * Y * Fφ_i)
                    Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * tbld[l+1, iglobθ]))
                end
            else
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    coeff = wi * scaleφ / (l*(l+1))
                    Slm_local[l+1, col] += coeff * (Fθ_i * dθY - (0 + 1im) * mval * inv_sθ * Y * Fφ_i)
                    Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * dPdx[l+1]))
                end
            end
        end
    end
    MPI.Allreduce!(Slm_local, +, comm)
    MPI.Allreduce!(Tlm_local, +, comm)
    # Convert to cfg's requested normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_local); T2 = similar(Tlm_local)
        SHTnsKit.convert_alm_norm!(S2, Slm_local, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(T2, Tlm_local, cfg; to_internal=false)
        return S2, T2
    else
        return Slm_local, Tlm_local
    end
end

function SHTnsKit.dist_spat_to_SHsphtor!(plan::DistSphtorPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                         Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(plan.cfg, Vtθφ, Vpθφ; use_tables, use_rfft=plan.use_rfft)
    copyto!(Slm_out, Slm); copyto!(Tlm_out, Tlm)
    return Slm_out, Tlm_out
end

# Distributed vector synthesis (spheroidal/toroidal) from dense spectra
function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Slm = S2; Tlm = T2
    end

    Fθk = PencilArrays.allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
    Fφk = PencilArrays.allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
    fill!(Fθk, 0); fill!(Fφk, 0)

    θloc = axes(Fθk, 1)
    mloc = axes(PencilArrays.allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)
    gl_θ = PencilArrays.globalindices(Fθk, 1)
    gl_m = PencilArrays.globalindices(PencilArrays.allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)

    nlon = cfg.nlon
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    Gθ = Vector{ComplexF64}(undef, length(θloc))
    Gφ = Vector{ComplexF64}(undef, length(θloc))
    inv_scaleφ = nlon / (2π)

    @inbounds for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        if cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
            tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
            for (ii, iθ) in enumerate(θloc)
                iglobθ = gl_θ[ii]
                x = cfg.x[iglobθ]
                sθ = sqrt(max(0.0, 1 - x*x))
                inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
                gθ = 0.0 + 0.0im
                gφ = 0.0 + 0.0im
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[l+1, iglobθ]) * Tl
                end
                Gθ[ii] = gθ; Gφ[ii] = gφ
            end
        else
            for (ii, iθ) in enumerate(θloc)
                iglobθ = gl_θ[ii]
                x = cfg.x[iglobθ]
                sθ = sqrt(max(0.0, 1 - x*x))
                inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
                gθ = 0.0 + 0.0im
                gφ = 0.0 + 0.0im
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[l+1]) * Tl
                end
                Gθ[ii] = gθ; Gφ[ii] = gφ
            end
        end
        # Place positive m Fourier modes
        for (ii, iθ) in enumerate(θloc)
            Fθk[iθ, col] = inv_scaleφ * Gθ[ii]
            Fφk[iθ, col] = inv_scaleφ * Gφ[ii]
        end
        # Hermitian conjugate for negative m to ensure real output
        if real_output && mval > 0
            conj_index = nlon - mval + 1
            for (ii, iθ) in enumerate(θloc)
                Fθk[iθ, conj_index] = conj(Fθk[iθ, col])
                Fφk[iθ, conj_index] = conj(Fφk[iθ, col])
            end
        end
    end

    pifft = SHTnsKitParallelExt._get_or_plan(:ifft, Fθk)
    Vtθφ = PencilFFTs.ifft(Fθk, pifft)
    Vpθφ = PencilFFTs.ifft(Fφk, pifft)
    if real_output
        Vtθφ = real.(Vtθφ)
        Vpθφ = real.(Vpθφ)
    end
    if cfg.robert_form
        θloc2 = axes(Vtθφ, 1)
        gl_θ2 = PencilArrays.globalindices(Vtθφ, 1)
        for (ii, iθ) in enumerate(θloc2)
            x = cfg.x[gl_θ2[ii]]
            sθ = sqrt(max(0.0, 1 - x*x))
            Vtθφ[iθ, :] .*= sθ
            Vpθφ[iθ, :] .*= sθ
        end
    end
    return Vtθφ, Vpθφ
end

# Convenience: spectral inputs as PencilArray (dense layout (:l,:m))
function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_SHsphtor_to_spat(cfg, Array(Slm), Array(Tlm); prototype_θφ, real_output)
end

function SHTnsKit.dist_SHsphtor_to_spat!(plan::DistSphtorPlan, Vtθφ_out::PencilArrays.PencilArray, Vpθφ_out::PencilArrays.PencilArray,
                                         Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(plan.cfg, Slm, Tlm; prototype_θφ=plan.prototype_θφ, real_output)
    copyto!(Vtθφ_out, Vt); copyto!(Vpθφ_out, Vp)
    return Vtθφ_out, Vpθφ_out
end

# QST distributed implementations by composition
function SHTnsKit.dist_spat_to_SHqst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    Qlm = SHTnsKit.dist_analysis(cfg, Vrθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ)
    return Qlm, Slm, Tlm
end

function SHTnsKit.dist_spat_to_SHqst!(plan::DistQstPlan, Qlm_out::AbstractMatrix, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                      Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    Q, S, T = SHTnsKit.dist_spat_to_SHqst(plan.cfg, Vrθφ, Vtθφ, Vpθφ)
    copyto!(Qlm_out, Q); copyto!(Slm_out, S); copyto!(Tlm_out, T)
    return Qlm_out, Slm_out, Tlm_out
end

# Synthesis to distributed fields from dense spectra
function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    Vr = SHTnsKit.dist_synthesis(cfg, Qlm; prototype_θφ, real_output)
    Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ, real_output)
    return Vr, Vt, Vp
end

function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::PencilArrays.PencilArray, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    Vr, Vt, Vp = SHTnsKit.dist_SHqst_to_spat(cfg, Array(Qlm), Array(Slm), Array(Tlm); prototype_θφ, real_output)
    return Vr, Vt, Vp
end

##########
# Simple roundtrip diagnostics (optional helpers)
##########

function SHTnsKit.dist_scalar_roundtrip!(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray)
    comm = PencilArrays.communicator(fθφ)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
    # Local and global relative errors
    local_diff2 = 0.0; local_ref2 = 0.0
    for i in axes(fθφ,1), j in axes(fθφ,2)
        d = fθφ_out[i,j] - fθφ[i,j]
        local_diff2 += abs2(d)
        local_ref2 += abs2(fθφ[i,j])
    end
    global_diff2 = MPI.Allreduce(local_diff2, +, comm)
    global_ref2 = MPI.Allreduce(local_ref2, +, comm)
    rel_local = sqrt(local_diff2 / (local_ref2 + eps()))
    rel_global = sqrt(global_diff2 / (global_ref2 + eps()))
    return rel_local, rel_global
end

function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    comm = PencilArrays.communicator(Vtθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ)
    Vt2, Vp2 = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vtθφ, real_output=true)
    # θ component
    lt_d2 = 0.0; lt_r2 = 0.0
    lp_d2 = 0.0; lp_r2 = 0.0
    for i in axes(Vtθφ,1), j in axes(Vtθφ,2)
        dt = Vt2[i,j] - Vtθφ[i,j]; dp = Vp2[i,j] - Vpθφ[i,j]
        lt_d2 += abs2(dt); lt_r2 += abs2(Vtθφ[i,j])
        lp_d2 += abs2(dp); lp_r2 += abs2(Vpθφ[i,j])
    end
    gt_d2 = MPI.Allreduce(lt_d2, +, comm); gt_r2 = MPI.Allreduce(lt_r2, +, comm)
    gp_d2 = MPI.Allreduce(lp_d2, +, comm); gp_r2 = MPI.Allreduce(lp_r2, +, comm)
    rl_t = sqrt(lt_d2 / (lt_r2 + eps())); rg_t = sqrt(gt_d2 / (gt_r2 + eps()))
    rl_p = sqrt(lp_d2 / (lp_r2 + eps())); rg_p = sqrt(gp_d2 / (gp_r2 + eps()))
    return (rl_t, rg_t), (rl_p, rg_p)
end
