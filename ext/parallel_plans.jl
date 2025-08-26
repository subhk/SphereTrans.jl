##########
# Minimal plan structs to keep API stable
##########

struct DistAnalysisPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false) = DistAnalysisPlan(cfg, prototype_θφ, use_rfft)

struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false) = DistPlan(cfg, prototype_θφ, use_rfft)

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false) = DistSphtorPlan(cfg, prototype_θφ, use_rfft)

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false) = DistQstPlan(cfg, prototype_θφ, use_rfft)
