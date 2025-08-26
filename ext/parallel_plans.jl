##########
# Minimal plan structs to keep API stable
##########

struct DistAnalysisPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArrays.PencilArray
    use_rfft::Bool
end

DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray; use_rfft::Bool=false) = DistAnalysisPlan(cfg, prototype_θφ, use_rfft)

struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArrays.PencilArray
end

DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray) = DistPlan(cfg, prototype_θφ)

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArrays.PencilArray
    use_rfft::Bool
end

DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false) = DistSphtorPlan(cfg, prototype_θφ, use_rfft)

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArrays.PencilArray
    use_rfft::Bool
end

DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArrays.PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false) = DistQstPlan(cfg, prototype_θφ, use_rfft)
