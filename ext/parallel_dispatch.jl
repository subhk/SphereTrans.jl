##########
# Unified dispatch helpers for PencilArray inputs
##########

SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray) = SHTnsKit.dist_analysis(cfg, fθφ)

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArrays.PencilArray; 
                    prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)

    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; 
                    return_type::Symbol=:matrix, prototype_θφ=nothing)

    Alm = SHTnsKit.dist_analysis(cfg, fθφ)

    return return_type === :pencil ? PencilArrays.PencilArray(Alm) : Alm
end

# Convenience: synthesis dispatch sugar from dense Alm to PencilArray
function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; 
                prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
                
    return SHTnsKit.dist_synthesis(cfg, PencilArrays.PencilArray(Alm); prototype_θφ, real_output)
end

##########
# Vector/QST dispatch for PencilArrays
##########

function SHTnsKit.spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray; use_tables=cfg.use_plm_tables)
    return SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ; use_tables)
end

function SHTnsKit.SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; 
                                   prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ, real_output)
end

function SHTnsKit.SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; 
                                   prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ, real_output)
end

function SHTnsKit.spat_to_SHqst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArrays.PencilArray, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    return SHTnsKit.dist_spat_to_SHqst(cfg, Vrθφ, Vtθφ, Vpθφ)
end

function SHTnsKit.SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::PencilArrays.PencilArray, Slm::PencilArrays.PencilArray, Tlm::PencilArrays.PencilArray; 
                                prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_SHqst_to_spat(cfg, Qlm, Slm, Tlm; prototype_θφ, real_output)
end

function SHTnsKit.SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; 
                                prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_SHqst_to_spat(cfg, Qlm, Slm, Tlm; prototype_θφ, real_output)
end
