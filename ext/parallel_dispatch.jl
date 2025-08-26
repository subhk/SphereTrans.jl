##########
# Unified dispatch helpers for PencilArray inputs
##########

SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray) = SHTnsKit.dist_analysis(cfg, fθφ)

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArrays.PencilArray; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray; return_type::Symbol=:matrix, prototype_θφ=nothing)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    return return_type === :pencil ? PencilArrays.PencilArray(Alm) : Alm
end

# Convenience: synthesis dispatch sugar from dense Alm to PencilArray
function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; prototype_θφ::PencilArrays.PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_synthesis(cfg, PencilArrays.PencilArray(Alm); prototype_θφ, real_output)
end
