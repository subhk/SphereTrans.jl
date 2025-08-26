##########
# Distributed transforms using PencilFFTs/PencilArrays (scalar) and safe fallbacks for vector/QST
##########

function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    comm = PencilArrays.communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    pfft = PencilFFTs.plan_fft(fθφ; dims=2)
    Fθk = PencilFFTs.fft(fθφ, pfft)
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
    Alm = SHTnsKit.dist_analysis(plan.cfg, fθφ; use_tables)
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

# Vector/QST fallbacks for now

function SHTnsKit.dist_spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    return SHTnsKit.spat_to_SHsphtor(cfg, Array(Vtθφ), Array(Vpθφ))
end

function SHTnsKit.dist_spat_to_SHsphtor!(plan::DistSphtorPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                         Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(plan.cfg, Vtθφ, Vpθφ; use_tables)
    copyto!(Slm_out, Slm); copyto!(Tlm_out, Tlm)
    return Slm_out, Tlm_out
end

function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    Vt, Vp = SHTnsKit.SHsphtor_to_spat(cfg, Array(Slm), Array(Tlm); real_output)
    Vt_p = similar(prototype_θφ); Vp_p = similar(prototype_θφ)
    θloc = axes(Vt_p, 1); φloc = axes(Vt_p, 2)
    for (ii, iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Vt_p, 1)[ii]
        for (jj, jφ) in enumerate(φloc)
            iglobφ = PencilArrays.globalindices(Vt_p, 2)[jj]
            Vt_p[iθ, jφ] = Vt[iglobθ, iglobφ]
            Vp_p[iθ, jφ] = Vp[iglobθ, iglobφ]
        end
    end
    return Vt_p, Vp_p
end

function SHTnsKit.dist_SHsphtor_to_spat!(plan::DistSphtorPlan, Vtθφ_out::PencilArrays.PencilArray, Vpθφ_out::PencilArrays.PencilArray,
                                         Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    Vt, Vp = SHTnsKit.SHsphtor_to_spat(plan.cfg, Slm, Tlm; real_output)
    θloc = axes(Vtθφ_out, 1); φloc = axes(Vtθφ_out, 2)
    for (ii, iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Vtθφ_out, 1)[ii]
        for (jj, jφ) in enumerate(φloc)
            iglobφ = PencilArrays.globalindices(Vtθφ_out, 2)[jj]
            Vtθφ_out[iθ, jφ] = Vt[iglobθ, iglobφ]
            Vpθφ_out[iθ, jφ] = Vp[iglobθ, iglobφ]
        end
    end
    return Vtθφ_out, Vpθφ_out
end

function SHTnsKit.dist_spat_to_SHqst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    return SHTnsKit.spat_to_SHqst(cfg, Array(Vrθφ), Array(Vtθφ), Array(Vpθφ))
end

function SHTnsKit.dist_spat_to_SHqst!(plan::DistQstPlan, Qlm_out::AbstractMatrix, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                      Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    Q, S, T = SHTnsKit.dist_spat_to_SHqst(plan.cfg, Vrθφ, Vtθφ, Vpθφ)
    copyto!(Qlm_out, Q); copyto!(Slm_out, S); copyto!(Tlm_out, T)
    return Qlm_out, Slm_out, Tlm_out
end

function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::PencilArrays.PencilArray, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    Vr, Vt, Vp = SHTnsKit.SHqst_to_spat(cfg, Array(Qlm), Array(Slm), Array(Tlm); real_output)
    Vr_p = similar(prototype_θφ); Vt_p = similar(prototype_θφ); Vp_p = similar(prototype_θφ)
    θloc = axes(Vr_p, 1); φloc = axes(Vr_p, 2)
    for (ii, iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(Vr_p, 1)[ii]
        for (jj, jφ) in enumerate(φloc)
            iglobφ = PencilArrays.globalindices(Vr_p, 2)[jj]
            Vr_p[iθ, jφ] = Vr[iglobθ, iglobφ]
            Vt_p[iθ, jφ] = Vt[iglobθ, iglobφ]
            Vp_p[iθ, jφ] = Vp[iglobθ, iglobφ]
        end
    end
    return Vr_p, Vt_p, Vp_p
end
