##########
# Minimal distributed transforms using safe Array fallbacks
##########

function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    return SHTnsKit.analysis(cfg, Array(fθφ))
end

function SHTnsKit.dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArrays.PencilArray; use_tables=plan.cfg.use_plm_tables)
    Alm = SHTnsKit.dist_analysis(plan.cfg, fθφ; use_tables)
    copyto!(Alm_out, Alm)
    return Alm_out
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    f = SHTnsKit.synthesis(cfg, Alm; real_output)
    f_p = similar(prototype_θφ)
    θloc = axes(f_p, 1); φloc = axes(f_p, 2)
    for (ii, iθ) in enumerate(θloc)
        iglobθ = PencilArrays.globalindices(f_p, 1)[ii]
        for (jj, jφ) in enumerate(φloc)
            iglobφ = PencilArrays.globalindices(f_p, 2)[jj]
            f_p[iθ, jφ] = f[iglobθ, iglobφ]
        end
    end
    return f_p
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_synthesis(cfg, Array(Alm); prototype_θφ, real_output)
end

function SHTnsKit.dist_synthesis!(plan::DistPlan, fθφ_out::PencilArrays.PencilArray, Alm::PencilArrays.PencilArray; real_output::Bool=true)
    f = SHTnsKit.dist_synthesis(plan.cfg, Alm; prototype_θφ=plan.prototype_θφ, real_output)
    copyto!(fθφ_out, f)
    return fθφ_out
end

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
