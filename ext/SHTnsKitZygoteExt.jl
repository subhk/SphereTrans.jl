"""
Zygote.jl extension for SHTnsKit.jl

This extension provides automatic differentiation support using Zygote.jl
for reverse-mode automatic differentiation of spherical harmonic transforms.

Key features:
- Reverse-mode AD for all SHT functions  
- Efficient gradient computation for optimization
- ChainRules compatibility
- Neural network and machine learning support
"""

module SHTnsKitZygoteExt

using SHTnsKit
using Zygote
using ChainRulesCore
using LinearAlgebra

# Import the functions we want to differentiate
import SHTnsKit: synthesize, analyze, sh_to_spat!, spat_to_sh!,
                 synthesize_vector, analyze_vector,
                 evaluate_at_point, power_spectrum, total_power,
                 spatial_integral, spatial_mean,
                 synthesize_complex, analyze_complex

# Re-export for convenience
using Zygote: @adjoint, pullback, gradient

# ==========================================
# Core Transform Rules (ChainRules)
# ==========================================

"""
ChainRule for synthesize (spectral → spatial).

Since synthesis is linear: y = A * x, we have ∂L/∂x = A^T * ∂L/∂y
where A represents the synthesis operation.
"""
function ChainRulesCore.rrule(::typeof(synthesize), cfg::SHTnsKit.SHTnsConfig{T}, 
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass
    spatial_result = synthesize(cfg, sh_coeffs)
    
    # Reverse pass: gradient w.r.t. sh_coeffs
    function synthesize_pullback(∂spatial)
        # Since synthesis is linear, the adjoint is analysis
        # ∂L/∂sh_coeffs = analyze(cfg, ∂L/∂spatial)
        ∂sh_coeffs = analyze(cfg, ∂spatial)
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return spatial_result, synthesize_pullback
end

"""
ChainRule for analyze (spatial → spectral).

Since analysis is linear: y = B * x, we have ∂L/∂x = B^T * ∂L/∂y  
where B represents the analysis operation, and B^T is synthesis.
"""
function ChainRulesCore.rrule(::typeof(analyze), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass
    spectral_result = analyze(cfg, spatial_data)
    
    # Reverse pass: gradient w.r.t. spatial_data
    function analyze_pullback(∂spectral)
        # The adjoint of analysis is synthesis
        # ∂L/∂spatial_data = synthesize(cfg, ∂L/∂spectral)
        ∂spatial_data = synthesize(cfg, ∂spectral)
        return (NoTangent(), NoTangent(), ∂spatial_data)
    end
    
    return spectral_result, analyze_pullback
end

# ==========================================
# In-place Transform Rules
# ==========================================

"""
ChainRule for in-place synthesis.
"""
function ChainRulesCore.rrule(::typeof(sh_to_spat!), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}, 
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass
    result = sh_to_spat!(cfg, sh_coeffs, spatial_data)
    
    function sh_to_spat_pullback(∂result)
        # Gradient w.r.t. sh_coeffs is analysis of output gradient
        ∂sh_coeffs = analyze(cfg, ∂result)
        # spatial_data is modified in-place, so gradient flows through
        ∂spatial_data = ∂result
        return (NoTangent(), NoTangent(), ∂sh_coeffs, ∂spatial_data)
    end
    
    return result, sh_to_spat_pullback
end

"""
ChainRule for in-place analysis.
"""
function ChainRulesCore.rrule(::typeof(spat_to_sh!), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V},
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass
    result = spat_to_sh!(cfg, spatial_data, sh_coeffs)
    
    function spat_to_sh_pullback(∂result)
        # Gradient w.r.t. spatial_data is synthesis of output gradient
        ∂spatial_data = synthesize(cfg, ∂result)
        # sh_coeffs is modified in-place, so gradient flows through
        ∂sh_coeffs = ∂result
        return (NoTangent(), NoTangent(), ∂spatial_data, ∂sh_coeffs)
    end
    
    return result, spat_to_sh_pullback
end

# ==========================================
# Vector Transform Rules
# ==========================================

"""
ChainRule for vector synthesis.
"""
function ChainRulesCore.rrule(::typeof(synthesize_vector), cfg::SHTnsKit.SHTnsConfig{T},
                              sph_coeffs::AbstractVector{V},
                              tor_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass
    u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
    
    function synthesize_vector_pullback((∂u_theta, ∂u_phi))
        # The adjoint of vector synthesis is vector analysis
        ∂sph_coeffs, ∂tor_coeffs = analyze_vector(cfg, ∂u_theta, ∂u_phi)
        return (NoTangent(), NoTangent(), ∂sph_coeffs, ∂tor_coeffs)
    end
    
    return (u_theta, u_phi), synthesize_vector_pullback
end

"""
ChainRule for vector analysis.
"""
function ChainRulesCore.rrule(::typeof(analyze_vector), cfg::SHTnsKit.SHTnsConfig{T},
                              u_theta::AbstractMatrix{V},
                              u_phi::AbstractMatrix{V}) where {T,V}
    # Forward pass
    sph_coeffs, tor_coeffs = analyze_vector(cfg, u_theta, u_phi)
    
    function analyze_vector_pullback((∂sph_coeffs, ∂tor_coeffs))
        # The adjoint of vector analysis is vector synthesis
        ∂u_theta, ∂u_phi = synthesize_vector(cfg, ∂sph_coeffs, ∂tor_coeffs)
        return (NoTangent(), NoTangent(), ∂u_theta, ∂u_phi)
    end
    
    return (sph_coeffs, tor_coeffs), analyze_vector_pullback
end

# ==========================================
# Complex Transform Rules
# ==========================================

"""
ChainRule for complex synthesis.
"""
function ChainRulesCore.rrule(::typeof(synthesize_complex), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{Complex{V}}) where {T,V}
    # Forward pass
    spatial_result = synthesize_complex(cfg, sh_coeffs)
    
    function synthesize_complex_pullback(∂spatial)
        # The adjoint is complex analysis
        ∂sh_coeffs = analyze_complex(cfg, ∂spatial)
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return spatial_result, synthesize_complex_pullback
end

"""
ChainRule for complex analysis.
"""
function ChainRulesCore.rrule(::typeof(analyze_complex), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{Complex{V}}) where {T,V}
    # Forward pass
    spectral_result = analyze_complex(cfg, spatial_data)
    
    function analyze_complex_pullback(∂spectral)
        # The adjoint is complex synthesis
        ∂spatial_data = synthesize_complex(cfg, ∂spectral)
        return (NoTangent(), NoTangent(), ∂spatial_data)
    end
    
    return spectral_result, analyze_complex_pullback
end

# ==========================================
# Point Evaluation Rules
# ==========================================

"""
ChainRule for point evaluation.
"""
function ChainRulesCore.rrule(::typeof(evaluate_at_point), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}, theta::Real, phi::Real) where {T,V}
    # Forward pass
    result = evaluate_at_point(cfg, sh_coeffs, theta, phi)
    
    function evaluate_at_point_pullback(∂result)
        # Gradient w.r.t. sh_coeffs: evaluate basis functions at the point
        ∂sh_coeffs = _evaluate_point_gradient(cfg, T(theta), T(phi), ∂result)
        return (NoTangent(), NoTangent(), ∂sh_coeffs, NoTangent(), NoTangent())
    end
    
    return result, evaluate_at_point_pullback
end

"""
Helper function to compute gradient for point evaluation.
"""
function _evaluate_point_gradient(cfg::SHTnsKit.SHTnsConfig{T}, theta::T, phi::T, 
                                 ∂result::V) where {T,V}
    cost = cos(theta)
    # Compute Legendre polynomials at the point
    plm_values = SHTnsKit.compute_associated_legendre(SHTnsKit.get_lmax(cfg), cost, cfg.norm)
    
    # Gradient is the basis functions evaluated at the point
    ∂sh_coeffs = zeros(V, SHTnsKit.get_nlm(cfg))
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if m == 0
            ∂sh_coeffs[idx] = ∂result * plm_values[idx]
        else
            # For m > 0, include the azimuthal part
            # This is simplified - full implementation needs proper complex exponentials
            phase_factor = cos(m * phi)  # Simplified for real harmonics
            ∂sh_coeffs[idx] = ∂result * plm_values[idx] * phase_factor
        end
    end
    
    return ∂sh_coeffs
end

# ==========================================
# Analysis Functions Rules
# ==========================================

"""
ChainRule for power spectrum computation.
"""
function ChainRulesCore.rrule(::typeof(power_spectrum), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass
    power_values = power_spectrum(cfg, sh_coeffs)
    
    function power_spectrum_pullback(∂power)
        # ∂P_l/∂c_{l,m} = 2 * c_{l,m} for power P_l = Σ_m |c_{l,m}|²
        ∂sh_coeffs = zeros(V, length(sh_coeffs))
        
        for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
            power_grad = ∂power[l + 1]  # Gradient w.r.t. P_l
            coeff_val = sh_coeffs[coeff_idx]
            
            if m == 0
                ∂sh_coeffs[coeff_idx] = 2 * coeff_val * power_grad
            else
                ∂sh_coeffs[coeff_idx] = 4 * coeff_val * power_grad  # Factor of 2 from m>0 doubling
            end
        end
        
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return power_values, power_spectrum_pullback
end

"""
ChainRule for total power computation.
"""
function ChainRulesCore.rrule(::typeof(total_power), cfg::SHTnsKit.SHTnsConfig{T},
                              sh_coeffs::AbstractVector{V}) where {T,V}
    # Forward pass  
    total_power_value = total_power(cfg, sh_coeffs)
    
    function total_power_pullback(∂total)
        # Total power is sum of power spectrum, so gradient distributes evenly
        power_spec = power_spectrum(cfg, sh_coeffs)
        ones_vec = ones(V, length(power_spec))
        _, power_pullback = ChainRulesCore.rrule(power_spectrum, cfg, sh_coeffs)
        _, _, ∂sh_coeffs = power_pullback(ones_vec .* ∂total)
        
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return total_power_value, total_power_pullback
end

# ==========================================
# Spatial Integration Rules
# ==========================================

"""
ChainRule for spatial integration.
"""
function ChainRulesCore.rrule(::typeof(spatial_integral), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass
    integral_value = spatial_integral(cfg, spatial_data)
    
    function spatial_integral_pullback(∂integral)
        # Gradient w.r.t. spatial_data: distribute gradient by quadrature weights
        lat_weights = SHTnsKit.create_latitude_weights(cfg)
        ∂spatial_data = zeros(V, size(spatial_data))
        
        for i in 1:SHTnsKit.get_nlat(cfg)
            for j in 1:SHTnsKit.get_nphi(cfg)
                ∂spatial_data[i, j] = ∂integral * lat_weights[i]
            end
        end
        
        return (NoTangent(), NoTangent(), ∂spatial_data)
    end
    
    return integral_value, spatial_integral_pullback
end

"""
ChainRule for spatial mean computation.
"""
function ChainRulesCore.rrule(::typeof(spatial_mean), cfg::SHTnsKit.SHTnsConfig{T},
                              spatial_data::AbstractMatrix{V}) where {T,V}
    # Forward pass
    mean_value = spatial_mean(cfg, spatial_data)
    
    function spatial_mean_pullback(∂mean)
        # Mean is integral / (4π), so gradient scales by 1/(4π)
        _, integral_pullback = ChainRulesCore.rrule(spatial_integral, cfg, spatial_data)
        _, _, ∂spatial_data = integral_pullback(∂mean / (4π))
        
        return (NoTangent(), NoTangent(), ∂spatial_data)
    end
    
    return mean_value, spatial_mean_pullback
end

# ==========================================
# High-level Zygote Interface
# ==========================================

"""
    gradient(loss, cfg::SHTnsConfig, params...)

Compute gradients using Zygote for optimization and machine learning.

# Examples
```julia
cfg = create_gauss_config(8, 8)
x = rand(get_nlm(cfg))

# Gradient of total power
∇P = gradient(params -> total_power(cfg, params), x)[1]

# Gradient of point evaluation
θ, φ = π/4, π/2  
∇f = gradient(params -> evaluate_at_point(cfg, params, θ, φ), x)[1]

# Gradient of synthesis-based loss
target = rand(get_nlat(cfg), get_nphi(cfg))
loss(params) = sum(abs2, synthesize(cfg, params) - target)
∇L = gradient(loss, x)[1]
```
"""
Zygote.gradient

"""
    pullback(f, cfg::SHTnsConfig, params...)

Get both forward value and pullback function for reverse-mode AD.

# Examples
```julia
cfg = create_gauss_config(8, 8)
x = rand(get_nlm(cfg))

# Get value and pullback
f = params -> total_power(cfg, params)
val, back = pullback(f, x)
∇ = back(1.0)[1]  # Gradient
```
"""
Zygote.pullback

# ==========================================
# Machine Learning Utilities
# ==========================================

"""
    sht_layer(cfg::SHTnsConfig, sh_coeffs::AbstractVector) -> AbstractMatrix

A differentiable "SHT layer" that can be used in neural networks.
"""
function sht_layer(cfg::SHTnsKit.SHTnsConfig{T}, sh_coeffs::AbstractVector{V}) where {T,V}
    return synthesize(cfg, sh_coeffs)
end

"""
    inverse_sht_layer(cfg::SHTnsConfig, spatial_data::AbstractMatrix) -> AbstractVector

A differentiable "inverse SHT layer" for neural networks.
"""
function inverse_sht_layer(cfg::SHTnsKit.SHTnsConfig{T}, spatial_data::AbstractMatrix{V}) where {T,V}
    return analyze(cfg, spatial_data)
end

"""
    spectral_regularizer(cfg::SHTnsConfig, sh_coeffs::AbstractVector, p::Real=2) -> Real

L^p regularization in spectral domain.
"""
function spectral_regularizer(cfg::SHTnsKit.SHTnsConfig{T}, sh_coeffs::AbstractVector{V}, p::Real=2) where {T,V}
    if p == 2
        return sum(abs2, sh_coeffs)
    elseif p == 1
        return sum(abs, sh_coeffs)
    else
        return sum(x -> abs(x)^p, sh_coeffs)
    end
end

"""
    power_regularizer(cfg::SHTnsConfig, sh_coeffs::AbstractVector, weights::AbstractVector) -> Real

Weighted power spectrum regularization.
"""
function power_regularizer(cfg::SHTnsKit.SHTnsConfig{T}, sh_coeffs::AbstractVector{V}, 
                          weights::AbstractVector{W}) where {T,V,W}
    power = power_spectrum(cfg, sh_coeffs)
    return sum(weights .* power)
end

# ==========================================
# Optimization Utilities  
# ==========================================

"""
    gradient_descent_step!(cfg::SHTnsConfig, params::AbstractVector, 
                          loss_fn::Function, learning_rate::Real) -> Real

Perform one gradient descent step on spectral coefficients.
"""
function gradient_descent_step!(cfg::SHTnsKit.SHTnsConfig{T}, params::AbstractVector{V},
                                loss_fn::Function, learning_rate::Real) where {T,V}
    # Compute loss and gradient
    loss_val, ∇ = Zygote.withgradient(loss_fn, params)
    
    # Update parameters
    params .-= learning_rate .* ∇[1]
    
    return loss_val
end

"""
    adam_step!(cfg::SHTnsConfig, params, loss_fn, m, v, β1, β2, lr, t) -> Real

Perform one Adam optimization step on spectral coefficients.
"""
function adam_step!(cfg::SHTnsKit.SHTnsConfig{T}, params::AbstractVector{V},
                   loss_fn::Function, m::AbstractVector{V}, v::AbstractVector{V},
                   β1::Real=0.9, β2::Real=0.999, lr::Real=0.001, t::Int=1) where {T,V}
    # Compute loss and gradient
    loss_val, ∇ = Zygote.withgradient(loss_fn, params)
    g = ∇[1]
    
    # Update biased first and second moment estimates
    m .= β1 .* m .+ (1 - β1) .* g
    v .= β2 .* v .+ (1 - β2) .* (g .^ 2)
    
    # Compute bias-corrected moment estimates
    m_hat = m ./ (1 - β1^t)
    v_hat = v ./ (1 - β2^t)
    
    # Update parameters
    params .-= lr .* m_hat ./ (sqrt.(v_hat) .+ 1e-8)
    
    return loss_val
end

end # module