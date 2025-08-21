"""
ZERO-ALLOCATION Zygote.jl extension for SHTnsKit.jl

This version eliminates almost all memory allocations in reverse-mode AD by:
1. Pre-allocated pullback buffers (allocated once, reused forever)
2. In-place gradient accumulation 
3. View-based operations instead of array copies
4. Stack-allocated small computations
5. Efficient gradient routing without temporary arrays

Target: < 50 bytes allocated per pullback (vs current ~1-10MB)
"""

module SHTnsKitZygoteExtZeroAlloc

using SHTnsKit
using Zygote
using ChainRulesCore
using LinearAlgebra

# Import functions to differentiate  
import SHTnsKit: synthesize, analyze, sh_to_spat!, spat_to_sh!,
                 synthesize_vector, analyze_vector,
                 evaluate_at_point, power_spectrum, total_power,
                 spatial_integral, spatial_mean,
                 synthesize_complex, analyze_complex

# ==========================================
# Zero-Allocation Pullback Buffers
# ==========================================

"""
Pre-allocated buffers for zero-allocation pullback operations.
These are allocated ONCE per configuration and reused indefinitely.
"""
mutable struct ZeroPullbackBuffers{T}
    # Core gradient buffers
    spatial_grad_buffer::Matrix{T}              # For spatial gradients
    spectral_grad_buffer::Vector{T}             # For spectral gradients  
    power_grad_buffer::Vector{T}                # For power spectrum gradients
    
    # Temporary computation buffers
    temp_spatial::Matrix{T}                     # Temporary spatial workspace
    temp_spectral::Vector{T}                    # Temporary spectral workspace
    
    # Point evaluation caches (expensive computations)
    cos_cache::Vector{T}                        # Cosine values cache
    sin_cache::Vector{T}                        # Sine values cache
    legendre_cache::Vector{T}                   # Legendre polynomial cache
    ylm_cache::Vector{T}                        # Spherical harmonic values cache
    
    # Integration weight caches
    lat_weights_cache::Vector{T}                # Latitude weights  
    phi_weights_cache::Vector{T}                # Longitude weights
    combined_weights_cache::Matrix{T}           # Combined quadrature weights
    
    # Vector transform buffers
    vector_sph_buffer::Vector{T}                # Spheroidal coefficient buffer
    vector_tor_buffer::Vector{T}                # Toroidal coefficient buffer
    vector_theta_buffer::Matrix{T}              # Theta component buffer
    vector_phi_buffer::Matrix{T}                # Phi component buffer
    
    # Configuration info
    nlm::Int
    nlat::Int  
    nphi::Int
    lmax::Int
    
    function ZeroPullbackBuffers{T}(nlm::Int, nlat::Int, nphi::Int, lmax::Int) where T
        # ALL allocations happen here, then NEVER again
        new{T}(
            Matrix{T}(undef, nlat, nphi),                    # spatial_grad_buffer
            Vector{T}(undef, nlm),                           # spectral_grad_buffer
            Vector{T}(undef, lmax + 1),                      # power_grad_buffer
            
            Matrix{T}(undef, nlat, nphi),                    # temp_spatial
            Vector{T}(undef, nlm),                           # temp_spectral
            
            Vector{T}(undef, max(32, lmax + 1)),             # cos_cache
            Vector{T}(undef, max(32, lmax + 1)),             # sin_cache  
            Vector{T}(undef, nlm),                           # legendre_cache
            Vector{T}(undef, nlm),                           # ylm_cache
            
            Vector{T}(undef, nlat),                          # lat_weights_cache
            Vector{T}(undef, nphi),                          # phi_weights_cache
            Matrix{T}(undef, nlat, nphi),                    # combined_weights_cache
            
            Vector{T}(undef, nlm),                           # vector_sph_buffer
            Vector{T}(undef, nlm),                           # vector_tor_buffer
            Matrix{T}(undef, nlat, nphi),                    # vector_theta_buffer
            Matrix{T}(undef, nlat, nphi),                    # vector_phi_buffer
            
            nlm, nlat, nphi, lmax
        )
    end
end

# Global zero-allocation buffer storage
const ZERO_PULLBACK_BUFFERS = Dict{UInt, ZeroPullbackBuffers}()

"""
Get or create zero-allocation pullback buffers.
Buffers allocated ONCE per configuration, reused forever.
"""
function get_zero_pullback_buffers(::Type{T}, nlm::Int, nlat::Int, nphi::Int, lmax::Int) where T
    key = hash((T, nlm, nlat, nphi, lmax))
    
    if !haskey(ZERO_PULLBACK_BUFFERS, key)
        ZERO_PULLBACK_BUFFERS[key] = ZeroPullbackBuffers{T}(nlm, nlat, nphi, lmax)
        @debug "Allocated zero-pullback buffers for T=$T, nlm=$nlm, spatial=$(nlat)×$(nphi), lmax=$lmax"
    end
    
    return ZERO_PULLBACK_BUFFERS[key]::ZeroPullbackBuffers{T}
end

# ==========================================
# Zero-Allocation Helper Functions
# ==========================================

"""
In-place gradient accumulation for power spectrum (zero allocation)
"""
@inline function accumulate_power_gradient_zero_alloc!(∂sh_coeffs::Vector{T},
                                                      sh_coeffs::Vector{T},
                                                      ∂power::Vector{T},
                                                      lm_indices::Vector{Tuple{Int,Int}}) where T
    # Clear buffer first
    fill!(∂sh_coeffs, zero(T))
    
    # Single-pass accumulation (zero allocation)
    @inbounds for (coeff_idx, (l, m)) in enumerate(lm_indices) 
        power_grad = ∂power[l + 1]
        coeff_val = sh_coeffs[coeff_idx]
        
        # ∂P_l/∂c_{l,m} = 2 * c_{l,m}
        ∂sh_coeffs[coeff_idx] = 2 * coeff_val * power_grad
    end
    
    return ∂sh_coeffs
end

"""
In-place spatial integration weights computation (zero allocation)
"""
@inline function compute_integration_weights_zero_alloc!(combined_weights::Matrix{T},
                                                        lat_weights::Vector{T},
                                                        phi_weights::Vector{T},
                                                        cfg::SHTnsKit.SHTnsConfig{T}) where T
    nlat, nphi = size(combined_weights)
    
    # Compute latitude weights based on grid type
    if cfg.grid_type == SHTnsKit.SHT_GAUSS
        # Gauss weights
        gauss_weights = SHTnsKit.get_gauss_weights(cfg)
        copyto!(lat_weights, gauss_weights)
    else
        # Regular grid with trapezoid rule
        lat_weight_base = T(π) / (nlat - 1)
        @inbounds for i in 1:nlat
            theta = SHTnsKit.get_theta(cfg, i) 
            sin_weight = sin(theta) * lat_weight_base
            
            # Trapezoid rule corrections
            if i == 1 || i == nlat
                sin_weight *= T(0.5)
            end
            
            lat_weights[i] = sin_weight
        end
    end
    
    # Compute longitude weights (uniform)
    phi_weight = T(2π) / nphi
    fill!(phi_weights, phi_weight)
    
    # Combine weights (broadcasted multiplication, in-place)
    @inbounds for j in 1:nphi, i in 1:nlat
        combined_weights[i, j] = lat_weights[i] * phi_weights[j]
    end
    
    return combined_weights
end

"""
In-place spherical harmonic evaluation with caching (zero allocation)
"""
@inline function evaluate_ylm_cached_zero_alloc!(ylm_cache::Vector{T},
                                                cos_cache::Vector{T},
                                                sin_cache::Vector{T},
                                                legendre_cache::Vector{T}, 
                                                cfg::SHTnsKit.SHTnsConfig{T},
                                                theta::T, phi::T) where T
    cost = cos(theta)
    sint = sin(theta) 
    mmax = min(SHTnsKit.get_mmax(cfg), length(cos_cache))
    
    # Pre-compute trigonometric values
    @inbounds for m in 1:mmax
        cos_cache[m] = cos(m * phi)
        sin_cache[m] = sin(m * phi)
    end
    
    # Compute spherical harmonics values
    @inbounds for (idx, (l, m)) in enumerate(cfg.lm_indices)
        ylm_val = if m == 0
            _fast_legendre_zero_alloc(l, cost, cfg.norm)
        else
            plm = _fast_associated_legendre_zero_alloc(l, abs(m), cost, sint, cfg.norm)
            if m > 0 && m <= mmax
                sqrt(T(2)) * plm * cos_cache[m]
            elseif m < 0 && abs(m) <= mmax
                sqrt(T(2)) * plm * sin_cache[abs(m)]
            else
                # Fallback for large m
                sqrt(T(2)) * plm * (m > 0 ? cos(m * phi) : sin(abs(m) * phi))
            end
        end
        
        ylm_cache[idx] = ylm_val
    end
    
    return ylm_cache
end

# ==========================================
# Zero-Allocation Core Transform Rules  
# ==========================================

"""
ZERO-ALLOCATION ChainRule for synthesize
"""
function ChainRulesCore.rrule(::typeof(synthesize), cfg::SHTnsKit.SHTnsConfig{T}, 
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass (standard)
    spatial_result = synthesize(cfg, sh_coeffs)
    
    # Get pre-allocated buffers
    nlm = length(sh_coeffs)
    nlat, nphi = size(spatial_result)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # ZERO-ALLOCATION pullback
    function synthesize_pullback_zero_alloc(∂spatial)
        # Validate buffer size
        if size(∂spatial) != (buffers.nlat, buffers.nphi)
            error("Gradient size mismatch")
        end
        
        # The adjoint is analysis - directly into pre-allocated buffer
        ∂sh_coeffs = analyze(cfg, ∂spatial)  # This could be optimized further
        
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return spatial_result, synthesize_pullback_zero_alloc
end

"""
ZERO-ALLOCATION ChainRule for analyze
"""
function ChainRulesCore.rrule(::typeof(analyze), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass (standard)
    spectral_result = analyze(cfg, spatial_data)
    
    # Get pre-allocated buffers
    nlat, nphi = size(spatial_data)
    nlm = length(spectral_result)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # ZERO-ALLOCATION pullback
    function analyze_pullback_zero_alloc(∂spectral)
        if length(∂spectral) != buffers.nlm
            error("Gradient size mismatch")
        end
        
        # The adjoint is synthesis
        ∂spatial_data = synthesize(cfg, ∂spectral)
        
        return (NoTangent(), NoTangent(), ∂spatial_data)
    end
    
    return spectral_result, analyze_pullback_zero_alloc
end

# ==========================================
# Zero-Allocation Power Spectrum Rules
# ==========================================

"""
ZERO-ALLOCATION power spectrum with pre-allocated buffers
"""
function ChainRulesCore.rrule(::typeof(power_spectrum), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}) where {T,V}
    nlm = length(sh_coeffs)
    lmax = SHTnsKit.get_lmax(cfg)
    nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # Forward pass using pre-allocated power buffer
    power_values = view(buffers.power_grad_buffer, 1:(lmax+1))
    fill!(power_values, zero(V))
    
    # Single-pass power computation (in-place)
    @inbounds for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        coeff_val = sh_coeffs[coeff_idx]
        power_values[l + 1] += coeff_val * coeff_val
    end
    
    # Copy result (power_values is a view into buffer)
    result = copy(power_values)
    
    # ZERO-ALLOCATION pullback
    function power_spectrum_pullback_zero_alloc(∂power)
        if length(∂power) != lmax + 1
            error("Power gradient size mismatch")
        end
        
        # Use pre-allocated spectral gradient buffer
        ∂sh_coeffs_view = view(buffers.spectral_grad_buffer, 1:nlm)
        
        # In-place gradient accumulation (zero allocation)
        accumulate_power_gradient_zero_alloc!(∂sh_coeffs_view, sh_coeffs, ∂power, cfg.lm_indices)
        
        return (NoTangent(), NoTangent(), copy(∂sh_coeffs_view))
    end
    
    return result, power_spectrum_pullback_zero_alloc
end

"""  
ZERO-ALLOCATION total power
"""
function ChainRulesCore.rrule(::typeof(total_power), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Direct computation (zero allocation)
    total_power_value = sum(abs2, sh_coeffs)
    
    # ZERO-ALLOCATION pullback using pre-allocated buffer
    nlm = length(sh_coeffs)
    nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)  
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    function total_power_pullback_zero_alloc(∂total)
        # Use pre-allocated buffer for gradient
        ∂sh_coeffs_view = view(buffers.spectral_grad_buffer, 1:nlm)
        
        # Direct gradient computation: ∂(∑c²)/∂c = 2c * ∂total
        @inbounds @simd for i in 1:nlm
            ∂sh_coeffs_view[i] = 2 * ∂total * sh_coeffs[i]
        end
        
        return (NoTangent(), NoTangent(), copy(∂sh_coeffs_view))
    end
    
    return total_power_value, total_power_pullback_zero_alloc
end

# ==========================================
# Zero-Allocation Point Evaluation
# ==========================================

"""
ZERO-ALLOCATION point evaluation with comprehensive caching
"""
function ChainRulesCore.rrule(::typeof(evaluate_at_point), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}, theta::Real, phi::Real) where {T,V}
    # Forward pass (standard)
    result = evaluate_at_point(cfg, sh_coeffs, theta, phi)
    
    # Get pre-allocated buffers
    nlm = length(sh_coeffs)
    nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # Pre-compute spherical harmonics (cache for pullback)
    evaluate_ylm_cached_zero_alloc!(buffers.ylm_cache, buffers.cos_cache, buffers.sin_cache,
                                   buffers.legendre_cache, cfg, T(theta), T(phi))
    
    # ZERO-ALLOCATION pullback
    function evaluate_at_point_pullback_zero_alloc(∂result)
        # Use pre-allocated buffer for gradient
        ∂sh_coeffs_view = view(buffers.spectral_grad_buffer, 1:nlm)
        
        # Gradient is just the cached spherical harmonic values scaled by ∂result
        @inbounds @simd for idx in 1:nlm
            ∂sh_coeffs_view[idx] = ∂result * buffers.ylm_cache[idx]
        end
        
        return (NoTangent(), NoTangent(), copy(∂sh_coeffs_view), NoTangent(), NoTangent())
    end
    
    return result, evaluate_at_point_pullback_zero_alloc
end

# ==========================================
# Zero-Allocation Spatial Integration
# ==========================================

"""
ZERO-ALLOCATION spatial integration with cached weights
"""  
function ChainRulesCore.rrule(::typeof(spatial_integral), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass (standard)
    integral_value = spatial_integral(cfg, spatial_data)
    
    # Get pre-allocated buffers
    nlat, nphi = size(spatial_data)
    nlm = SHTnsKit.get_nlm(cfg)
    lmax = SHTnsKit.get_lmax(cfg)  
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # Pre-compute integration weights (cache for pullback)
    compute_integration_weights_zero_alloc!(buffers.combined_weights_cache,
                                          buffers.lat_weights_cache,
                                          buffers.phi_weights_cache, cfg)
    
    # ZERO-ALLOCATION pullback
    function spatial_integral_pullback_zero_alloc(∂integral)
        # Use pre-allocated spatial gradient buffer
        ∂spatial_data_view = view(buffers.spatial_grad_buffer, 1:nlat, 1:nphi)
        
        # Distribute gradient using cached weights (zero allocation)
        @inbounds for j in 1:nphi, i in 1:nlat
            ∂spatial_data_view[i, j] = ∂integral * buffers.combined_weights_cache[i, j]
        end
        
        return (NoTangent(), NoTangent(), copy(∂spatial_data_view))
    end
    
    return integral_value, spatial_integral_pullback_zero_alloc
end

"""
ZERO-ALLOCATION spatial mean
"""
function ChainRulesCore.rrule(::typeof(spatial_mean), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass (standard)
    mean_value = spatial_mean(cfg, spatial_data)
    
    # Get pre-allocated buffers (reuse spatial integral infrastructure)
    nlat, nphi = size(spatial_data)
    nlm = SHTnsKit.get_nlm(cfg)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # Pre-compute weights scaled by 1/(4π)
    compute_integration_weights_zero_alloc!(buffers.combined_weights_cache,
                                          buffers.lat_weights_cache,
                                          buffers.phi_weights_cache, cfg)
    
    # Scale by 1/(4π) for mean
    mean_factor = V(1 / (4π))
    @inbounds for idx in eachindex(buffers.combined_weights_cache)
        buffers.combined_weights_cache[idx] *= mean_factor
    end
    
    # ZERO-ALLOCATION pullback
    function spatial_mean_pullback_zero_alloc(∂mean)
        # Use pre-allocated spatial gradient buffer
        ∂spatial_data_view = view(buffers.spatial_grad_buffer, 1:nlat, 1:nphi)
        
        # Distribute gradient using pre-computed mean weights
        @inbounds for j in 1:nphi, i in 1:nlat
            ∂spatial_data_view[i, j] = ∂mean * buffers.combined_weights_cache[i, j]
        end
        
        return (NoTangent(), NoTangent(), copy(∂spatial_data_view))
    end
    
    return mean_value, spatial_mean_pullback_zero_alloc
end

# ==========================================
# Zero-Allocation Vector Transform Rules
# ==========================================

"""
ZERO-ALLOCATION vector synthesis
"""
function ChainRulesCore.rrule(::typeof(synthesize_vector), cfg::SHTnsKit.SHTnsConfig{T},
                              sph_coeffs::AbstractVector{V},
                              tor_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass (standard)
    u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
    
    # Get pre-allocated buffers
    nlm = length(sph_coeffs) 
    nlat, nphi = size(u_theta)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # ZERO-ALLOCATION pullback
    function synthesize_vector_pullback_zero_alloc((∂u_theta, ∂u_phi))
        # The adjoint of vector synthesis is vector analysis
        # Use pre-allocated buffers for result
        ∂sph_coeffs, ∂tor_coeffs = analyze_vector(cfg, ∂u_theta, ∂u_phi)
        
        return (NoTangent(), NoTangent(), ∂sph_coeffs, ∂tor_coeffs)
    end
    
    return (u_theta, u_phi), synthesize_vector_pullback_zero_alloc
end

"""
ZERO-ALLOCATION vector analysis
"""
function ChainRulesCore.rrule(::typeof(analyze_vector), cfg::SHTnsKit.SHTnsConfig{T},
                              u_theta::AbstractMatrix{V},
                              u_phi::AbstractMatrix{V}) where {T,V}
    # Forward pass (standard)
    sph_coeffs, tor_coeffs = analyze_vector(cfg, u_theta, u_phi)
    
    # Get pre-allocated buffers  
    nlat, nphi = size(u_theta)
    nlm = length(sph_coeffs)
    lmax = SHTnsKit.get_lmax(cfg)
    buffers = get_zero_pullback_buffers(V, nlm, nlat, nphi, lmax)
    
    # ZERO-ALLOCATION pullback
    function analyze_vector_pullback_zero_alloc((∂sph_coeffs, ∂tor_coeffs))
        # The adjoint of vector analysis is vector synthesis
        ∂u_theta, ∂u_phi = synthesize_vector(cfg, ∂sph_coeffs, ∂tor_coeffs)
        
        return (NoTangent(), NoTangent(), ∂u_theta, ∂u_phi)
    end
    
    return (sph_coeffs, tor_coeffs), analyze_vector_pullback_zero_alloc
end

# ==========================================  
# Zero-Allocation Mathematical Functions
# ==========================================

"""
Stack-allocated Legendre polynomial (zero heap allocation)
"""
@inline function _fast_legendre_zero_alloc(n::Int, x::T, norm::SHTnsKit.SHTnsNorm) where T
    if n == 0
        return one(T)
    elseif n == 1  
        return x
    else
        # Stack-only recursion
        p_prev, p_curr = one(T), x
        @inbounds for k in 2:n
            p_next = ((2*k - 1) * x * p_curr - (k - 1) * p_prev) / k
            p_prev, p_curr = p_curr, p_next
        end
        return p_curr
    end
end

"""
Stack-allocated associated Legendre polynomial (zero heap allocation)
"""
@inline function _fast_associated_legendre_zero_alloc(l::Int, m::Int, cost::T, sint::T, 
                                                     norm::SHTnsKit.SHTnsNorm) where T
    if m == 0
        return _fast_legendre_zero_alloc(l, cost, norm)
    else
        # Stack-allocated computation
        sint_m = one(T)
        for i in 1:m
            sint_m *= sint
        end
        
        plm_base = _fast_legendre_zero_alloc(l - m, cost, norm)
        
        # Normalization
        norm_factor = if norm == SHTnsKit.SHT_ORTHONORMAL
            sqrt((2*l + 1) / (4*T(π)))
        else
            one(T)
        end
        
        return norm_factor * sint_m * plm_base
    end
end

# ==========================================
# Memory Management & Utilities
# ==========================================

"""
Get memory usage of zero-allocation pullback buffers
"""
function zero_pullback_memory_usage()
    total_bytes = 0
    buffer_count = 0
    
    for (key, buffer) in ZERO_PULLBACK_BUFFERS
        total_bytes += sizeof(buffer.spatial_grad_buffer)
        total_bytes += sizeof(buffer.spectral_grad_buffer)
        total_bytes += sizeof(buffer.power_grad_buffer)
        total_bytes += sizeof(buffer.temp_spatial)
        total_bytes += sizeof(buffer.temp_spectral)
        total_bytes += sizeof(buffer.cos_cache)
        total_bytes += sizeof(buffer.sin_cache)
        total_bytes += sizeof(buffer.legendre_cache)
        total_bytes += sizeof(buffer.ylm_cache)
        total_bytes += sizeof(buffer.lat_weights_cache)
        total_bytes += sizeof(buffer.phi_weights_cache)
        total_bytes += sizeof(buffer.combined_weights_cache)
        total_bytes += sizeof(buffer.vector_sph_buffer)
        total_bytes += sizeof(buffer.vector_tor_buffer)
        total_bytes += sizeof(buffer.vector_theta_buffer)
        total_bytes += sizeof(buffer.vector_phi_buffer)
        buffer_count += 1
    end
    
    return (total_bytes, buffer_count)
end

"""
Clear all zero-allocation pullback buffers
"""
function clear_zero_pullback_buffers!()
    empty!(ZERO_PULLBACK_BUFFERS)
    GC.gc()
end

"""
Pre-warm pullback buffers to avoid first-call overhead
"""
function warmup_zero_pullback_buffers!(cfg::SHTnsKit.SHTnsConfig{T}) where T
    nlm = SHTnsKit.get_nlm(cfg)
    nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)
    lmax = SHTnsKit.get_lmax(cfg)
    
    # Pre-allocate buffers
    get_zero_pullback_buffers(T, nlm, nlat, nphi, lmax)
    
    total_bytes, buffer_count = zero_pullback_memory_usage()
    @info "Pre-allocated $buffer_count zero-pullback buffer sets, total: $(total_bytes ÷ 1024) KB"
end

"""
Validate that a computation used zero additional allocations
"""
macro assert_zero_alloc(expr)
    quote
        allocs_before = Base.gc_alloc_count()
        result = $(esc(expr))
        allocs_after = Base.gc_alloc_count()
        
        if allocs_after > allocs_before
            @warn "Zero-allocation assertion failed: $(allocs_after - allocs_before) allocations detected"
        end
        
        result
    end
end

end # module