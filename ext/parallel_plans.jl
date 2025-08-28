##########
# Minimal plan structs to keep API stable
##########

struct DistAnalysisPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    # Pre-computed index maps for performance
    θ_local_to_global::Vector{Int}
    m_local_to_global::Vector{Int}
    m_local_range::UnitRange{Int}
    θ_local_range::UnitRange{Int}
    # Memory layout optimization
    use_packed_storage::Bool
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false, use_packed_storage::Bool=true)
    # Pre-compute index mappings to avoid expensive lookups in tight loops
    temp_pencil = allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64)
    
    θ_range = axes(temp_pencil, 1)
    m_range = axes(temp_pencil, 2)
    
    θ_local_to_global = collect(globalindices(temp_pencil, 1))
    m_local_to_global = collect(globalindices(temp_pencil, 2))
    
    return DistAnalysisPlan(cfg, prototype_θφ, use_rfft, θ_local_to_global, m_local_to_global, 
                           m_range, θ_range, use_packed_storage)
end

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
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,Tuple{PencilArray,PencilArray}}
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffers for IFFT operations
        scratch_θ = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        scratch_φ = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        (scratch_θ, scratch_φ)
    else
        nothing
    end
    return DistSphtorPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch)
end

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,PencilArray}
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffer for scalar IFFT operations
        allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
    else
        nothing
    end
    return DistQstPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch)
end
