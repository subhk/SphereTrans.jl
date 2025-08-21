"""
ForwardDiff.jl extension for SHTnsKit.jl

This extension provides automatic differentiation support using ForwardDiff.jl
for forward-mode automatic differentiation of spherical harmonic transforms.

Key features:
- Forward-mode AD for all SHT functions
- Efficient gradient computation
- Support for arbitrary-order derivatives
- Gradient-based optimization compatibility
"""

module SHTnsKitForwardDiffExt

using SHTnsKit
using ForwardDiff
using LinearAlgebra

# Import the functions we want to differentiate
import SHTnsKit: synthesize, analyze, sh_to_spat!, spat_to_sh!,
                 synthesize_vector, analyze_vector,
                 evaluate_at_point, power_spectrum, total_power,
                 spatial_integral, spatial_mean

"""
    _extract_dual_values(x::AbstractArray{<:ForwardDiff.Dual})

Extract the values from a Dual number array for use in non-AD computations.
"""
function _extract_dual_values(x::AbstractArray{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    return ForwardDiff.value.(x)
end

"""
    _construct_dual_result(result::AbstractArray{V}, partials::AbstractArray{<:ForwardDiff.Partials}) where V

Construct a Dual number array from values and partials.
"""
function _construct_dual_result(result::AbstractArray{V}, 
                                partials::AbstractArray{<:ForwardDiff.Partials{N,V}}) where {V,N}
    return ForwardDiff.Dual{ForwardDiff.Tag{Nothing,V},V,N}.(result, partials)
end

# ==========================================
# Core Transform Rules
# ==========================================

"""
ForwardDiff rule for synthesize (spectral → spatial).

For f(x) = synthesize(cfg, x), we have:
∂f/∂x = synthesize(cfg, ∂x)

This is because the synthesis operation is linear in the coefficients.
"""
function synthesize(cfg::SHTnsKit.SHTnsConfig{T}, 
                   sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Extract values and partials
    values = ForwardDiff.value.(sh_coeffs)
    partials = ForwardDiff.partials.(sh_coeffs)
    
    # Apply synthesis to values
    spatial_values = synthesize(cfg, values)
    
    # Apply synthesis to each partial derivative component
    n_partials = length(partials[1])
    spatial_partials = Matrix{V}(undef, size(spatial_values)..., n_partials)
    
    for i in 1:n_partials
        partial_coeffs = [p[i] for p in partials]
        spatial_partials[:, :, i] = synthesize(cfg, partial_coeffs)
    end
    
    # Construct dual result
    result = Matrix{ForwardDiff.Dual{Tag,V,N}}(undef, size(spatial_values))
    for idx in eachindex(spatial_values)
        partial_vec = ForwardDiff.Partials{N,V}(tuple([spatial_partials[idx, i] for i in 1:n_partials]...))
        result[idx] = ForwardDiff.Dual{Tag,V,N}(spatial_values[idx], partial_vec)
    end
    
    return result
end

"""
ForwardDiff rule for analyze (spatial → spectral).

For f(x) = analyze(cfg, x), we have:
∂f/∂x = analyze(cfg, ∂x)

This is because the analysis operation is also linear.
"""
function analyze(cfg::SHTnsKit.SHTnsConfig{T},
                spatial_data::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Extract values and partials
    values = ForwardDiff.value.(spatial_data)
    partials = ForwardDiff.partials.(spatial_data)
    
    # Apply analysis to values
    spectral_values = analyze(cfg, values)
    
    # Apply analysis to each partial derivative component
    n_partials = length(partials[1,1])
    spectral_partials = Matrix{V}(undef, length(spectral_values), n_partials)
    
    for i in 1:n_partials
        partial_spatial = [p[i] for p in partials]
        spectral_partials[:, i] = analyze(cfg, partial_spatial)
    end
    
    # Construct dual result
    result = Vector{ForwardDiff.Dual{Tag,V,N}}(undef, length(spectral_values))
    for idx in eachindex(spectral_values)
        partial_vec = ForwardDiff.Partials{N,V}(tuple([spectral_partials[idx, i] for i in 1:n_partials]...))
        result[idx] = ForwardDiff.Dual{Tag,V,N}(spectral_values[idx], partial_vec)
    end
    
    return result
end

# ==========================================
# In-place Transform Rules
# ==========================================

"""
ForwardDiff rule for in-place synthesis.
"""
function sh_to_spat!(cfg::SHTnsKit.SHTnsConfig{T},
                    sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}},
                    spatial_data::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Use the out-of-place version and copy results
    result = synthesize(cfg, sh_coeffs)
    spatial_data .= result
    return spatial_data
end

"""
ForwardDiff rule for in-place analysis.
"""
function spat_to_sh!(cfg::SHTnsKit.SHTnsConfig{T},
                    spatial_data::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}},
                    sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Use the out-of-place version and copy results
    result = analyze(cfg, spatial_data)
    sh_coeffs .= result
    return sh_coeffs
end

# ==========================================
# Vector Transform Rules
# ==========================================

"""
ForwardDiff rule for vector synthesis.
"""
function synthesize_vector(cfg::SHTnsKit.SHTnsConfig{T},
                          sph_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}},
                          tor_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Extract values
    sph_values = ForwardDiff.value.(sph_coeffs)
    tor_values = ForwardDiff.value.(tor_coeffs)
    sph_partials = ForwardDiff.partials.(sph_coeffs)
    tor_partials = ForwardDiff.partials.(tor_coeffs)
    
    # Apply vector synthesis to values
    u_theta_values, u_phi_values = synthesize_vector(cfg, sph_values, tor_values)
    
    # Apply to each partial component
    n_partials = length(sph_partials[1])
    u_theta_partials = Array{V,3}(undef, size(u_theta_values)..., n_partials)
    u_phi_partials = Array{V,3}(undef, size(u_phi_values)..., n_partials)
    
    for i in 1:n_partials
        sph_partial = [p[i] for p in sph_partials]
        tor_partial = [p[i] for p in tor_partials]
        u_theta_partials[:, :, i], u_phi_partials[:, :, i] = 
            synthesize_vector(cfg, sph_partial, tor_partial)
    end
    
    # Construct dual results
    u_theta_dual = Matrix{ForwardDiff.Dual{Tag,V,N}}(undef, size(u_theta_values))
    u_phi_dual = Matrix{ForwardDiff.Dual{Tag,V,N}}(undef, size(u_phi_values))
    
    for idx in eachindex(u_theta_values)
        # Theta component
        theta_partial_vec = ForwardDiff.Partials{N,V}(tuple([u_theta_partials[idx, i] for i in 1:n_partials]...))
        u_theta_dual[idx] = ForwardDiff.Dual{Tag,V,N}(u_theta_values[idx], theta_partial_vec)
        
        # Phi component  
        phi_partial_vec = ForwardDiff.Partials{N,V}(tuple([u_phi_partials[idx, i] for i in 1:n_partials]...))
        u_phi_dual[idx] = ForwardDiff.Dual{Tag,V,N}(u_phi_values[idx], phi_partial_vec)
    end
    
    return u_theta_dual, u_phi_dual
end

"""
ForwardDiff rule for vector analysis.
"""
function analyze_vector(cfg::SHTnsKit.SHTnsConfig{T},
                       u_theta::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}},
                       u_phi::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Extract values and partials
    u_theta_values = ForwardDiff.value.(u_theta)
    u_phi_values = ForwardDiff.value.(u_phi)
    u_theta_partials = ForwardDiff.partials.(u_theta)
    u_phi_partials = ForwardDiff.partials.(u_phi)
    
    # Apply vector analysis to values
    sph_values, tor_values = analyze_vector(cfg, u_theta_values, u_phi_values)
    
    # Apply to each partial component
    n_partials = length(u_theta_partials[1,1])
    sph_partials = Matrix{V}(undef, length(sph_values), n_partials)
    tor_partials = Matrix{V}(undef, length(tor_values), n_partials)
    
    for i in 1:n_partials
        u_theta_partial = [p[i] for p in u_theta_partials]
        u_phi_partial = [p[i] for p in u_phi_partials]
        sph_partials[:, i], tor_partials[:, i] = 
            analyze_vector(cfg, u_theta_partial, u_phi_partial)
    end
    
    # Construct dual results
    sph_dual = Vector{ForwardDiff.Dual{Tag,V,N}}(undef, length(sph_values))
    tor_dual = Vector{ForwardDiff.Dual{Tag,V,N}}(undef, length(tor_values))
    
    for idx in eachindex(sph_values)
        sph_partial_vec = ForwardDiff.Partials{N,V}(tuple([sph_partials[idx, i] for i in 1:n_partials]...))
        sph_dual[idx] = ForwardDiff.Dual{Tag,V,N}(sph_values[idx], sph_partial_vec)
        
        tor_partial_vec = ForwardDiff.Partials{N,V}(tuple([tor_partials[idx, i] for i in 1:n_partials]...))
        tor_dual[idx] = ForwardDiff.Dual{Tag,V,N}(tor_values[idx], tor_partial_vec)
    end
    
    return sph_dual, tor_dual
end

# ==========================================
# Point Evaluation Rules
# ==========================================

"""
ForwardDiff rule for point evaluation.
"""
function evaluate_at_point(cfg::SHTnsKit.SHTnsConfig{T},
                          sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}},
                          theta::Real, phi::Real) where {Tag,T,V,N}
    # Extract values and partials
    values = ForwardDiff.value.(sh_coeffs)
    partials = ForwardDiff.partials.(sh_coeffs)
    
    # Evaluate at point for values
    result_value = evaluate_at_point(cfg, values, T(theta), T(phi))
    
    # Evaluate at point for each partial
    n_partials = length(partials[1])
    result_partials = Vector{V}(undef, n_partials)
    
    for i in 1:n_partials
        partial_coeffs = [p[i] for p in partials]
        result_partials[i] = evaluate_at_point(cfg, partial_coeffs, T(theta), T(phi))
    end
    
    # Construct dual result
    partial_vec = ForwardDiff.Partials{N,V}(tuple(result_partials...))
    return ForwardDiff.Dual{Tag,V,N}(result_value, partial_vec)
end

# ==========================================
# Analysis Functions Rules
# ==========================================

"""
ForwardDiff rule for power spectrum computation.
"""
function power_spectrum(cfg::SHTnsKit.SHTnsConfig{T},
                       sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Power spectrum: P_l = Σ_m |c_{l,m}|² 
    # Derivative: ∂P_l/∂c_{l,m} = 2 * Re(c_{l,m}) for complex coefficients
    #            or ∂P_l/∂c_{l,m} = 2 * c_{l,m} for real coefficients
    
    values = ForwardDiff.value.(sh_coeffs)
    partials = ForwardDiff.partials.(sh_coeffs)
    
    # Compute power spectrum for values
    power_values = power_spectrum(cfg, values)
    
    # Compute derivatives
    n_partials = length(partials[1])
    power_partials = Matrix{V}(undef, length(power_values), n_partials)
    
    for i in 1:n_partials
        power_partials[:, i] = _power_spectrum_partial_derivative(cfg, values, partials, i)
    end
    
    # Construct dual result
    result = Vector{ForwardDiff.Dual{Tag,V,N}}(undef, length(power_values))
    for idx in eachindex(power_values)
        partial_vec = ForwardDiff.Partials{N,V}(tuple([power_partials[idx, i] for i in 1:n_partials]...))
        result[idx] = ForwardDiff.Dual{Tag,V,N}(power_values[idx], partial_vec)
    end
    
    return result
end

"""
Helper function to compute power spectrum partial derivatives.
"""
function _power_spectrum_partial_derivative(cfg::SHTnsKit.SHTnsConfig{T}, 
                                          values::AbstractVector{V},
                                          partials::AbstractVector{<:ForwardDiff.Partials},
                                          partial_idx::Int) where {T,V}
    lmax = SHTnsKit.get_lmax(cfg)
    power_derivs = zeros(V, lmax + 1)
    
    for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        coeff_val = values[coeff_idx]
        coeff_partial = partials[coeff_idx][partial_idx]
        
        # For power spectrum P_l = Σ_m |c_{l,m}|²
        # ∂P_l/∂c_{l,m} = 2 * c_{l,m} for all m (including m=0)
        # The factor of 2 comes from d/dx(x²) = 2x
        power_derivs[l + 1] += 2 * coeff_val * coeff_partial
    end
    
    return power_derivs
end

"""
ForwardDiff rule for total power computation.
"""
function total_power(cfg::SHTnsKit.SHTnsConfig{T},
                    sh_coeffs::AbstractVector{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Total power is sum of power spectrum
    power_spec = power_spectrum(cfg, sh_coeffs)
    return sum(power_spec)
end

# ==========================================
# Spatial Integration Rules
# ==========================================

"""
ForwardDiff rule for spatial integration.
"""
function spatial_integral(cfg::SHTnsKit.SHTnsConfig{T},
                         spatial_data::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Extract values and partials
    values = ForwardDiff.value.(spatial_data)
    partials = ForwardDiff.partials.(spatial_data)
    
    # Compute integral of values
    integral_value = spatial_integral(cfg, values)
    
    # Compute integral of each partial
    n_partials = length(partials[1,1])
    integral_partials = Vector{V}(undef, n_partials)
    
    for i in 1:n_partials
        partial_spatial = [p[i] for p in partials]
        integral_partials[i] = spatial_integral(cfg, partial_spatial)
    end
    
    # Construct dual result
    partial_vec = ForwardDiff.Partials{N,V}(tuple(integral_partials...))
    return ForwardDiff.Dual{Tag,V,N}(integral_value, partial_vec)
end

"""
ForwardDiff rule for spatial mean computation.
"""
function spatial_mean(cfg::SHTnsKit.SHTnsConfig{T},
                     spatial_data::AbstractMatrix{<:ForwardDiff.Dual{Tag,V,N}}) where {Tag,T,V,N}
    # Mean is integral / (4π)
    integral_result = spatial_integral(cfg, spatial_data)
    return integral_result / (4π)
end

# ==========================================
# Utility Functions
# ==========================================

"""
    gradient(f, cfg::SHTnsConfig, x::AbstractVector) -> Vector

Compute the gradient of function f with respect to spectral coefficients x.

# Example
```julia
cfg = create_gauss_config(8, 8)
x = rand(get_nlm(cfg))

# Gradient of total power with respect to coefficients
∇P = gradient(x -> total_power(cfg, x), cfg, x)

# Gradient of field value at specific point
θ, φ = π/4, π/2
∇f = gradient(x -> evaluate_at_point(cfg, x, θ, φ), cfg, x)
```
"""
function gradient(f::Function, cfg::SHTnsKit.SHTnsConfig, x::AbstractVector{T}) where T
    return ForwardDiff.gradient(f, x)
end

"""
    jacobian(f, cfg::SHTnsConfig, x::AbstractVector) -> Matrix

Compute the Jacobian matrix of vector function f with respect to spectral coefficients x.
"""
function jacobian(f::Function, cfg::SHTnsKit.SHTnsConfig, x::AbstractVector{T}) where T
    return ForwardDiff.jacobian(f, x)
end

"""
    hessian(f, cfg::SHTnsConfig, x::AbstractVector) -> Matrix

Compute the Hessian matrix of scalar function f with respect to spectral coefficients x.
"""
function hessian(f::Function, cfg::SHTnsKit.SHTnsConfig, x::AbstractVector{T}) where T
    return ForwardDiff.hessian(f, x)
end

end # module