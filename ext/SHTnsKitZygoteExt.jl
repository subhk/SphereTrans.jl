module SHTnsKitZygoteExt

using SHTnsKit
import Zygote
using Zygote: @adjoint, pullback
using ChainRulesCore
using ChainRulesCore: rrule, NoTangent, @not_implemented

"""
Zygote extension for SHTnsKit.jl

This extension enables reverse-mode automatic differentiation through 
spherical harmonic transforms using Zygote.jl. It provides adjoint rules
for the main transform functions leveraging the fact that spherical
harmonic transforms are linear operations.

Key insight: If F is a linear transformation, then:
- Forward: y = F(x)  
- Reverse: x̄ = F*(ȳ) where F* is the adjoint of F

For spherical harmonic transforms:
- synthesize* = analyze (up to normalization)
- analyze* = synthesize (up to normalization)
"""

# ============================================================================
# Reverse-mode AD rules for basic transforms
# ============================================================================

"""
Adjoint rule for synthesize function.
Since synthesize is the forward SHT, its adjoint is the backward SHT (analyze).
"""
function rrule(::typeof(synthesize), 
               cfg::SHTnsConfig, 
               sh::AbstractVector{<:Real})
    
    # Forward pass
    spatial = synthesize(cfg, sh)
    
    # Pullback function
    function synthesize_pullback(spatial_bar)
        # The adjoint of synthesis is analysis
        # ∂L/∂sh = analyze(cfg, ∂L/∂spatial)
        sh_bar = if spatial_bar === nothing
            NoTangent()
        else
            analyze(cfg, spatial_bar)
        end
        
        return NoTangent(), NoTangent(), sh_bar
    end
    
    return spatial, synthesize_pullback
end

"""
Adjoint rule for analyze function.
Since analyze is the backward SHT, its adjoint is the forward SHT (synthesize).
"""
function rrule(::typeof(analyze), 
               cfg::SHTnsConfig, 
               spatial::AbstractMatrix{<:Real})
    
    # Forward pass
    sh = analyze(cfg, spatial)
    
    # Pullback function
    function analyze_pullback(sh_bar)
        # The adjoint of analysis is synthesis
        # ∂L/∂spatial = synthesize(cfg, ∂L/∂sh)
        spatial_bar = if sh_bar === nothing
            NoTangent()
        else
            synthesize(cfg, sh_bar)
        end
        
        return NoTangent(), NoTangent(), spatial_bar
    end
    
    return sh, analyze_pullback
end

# ============================================================================
# In-place variants
# ============================================================================

"""
Adjoint rule for in-place synthesize!
Note: in-place operations need special care in reverse-mode AD
"""
function rrule(::typeof(synthesize!), 
               cfg::SHTnsConfig,
               sh::AbstractVector{<:Real},
               spatial::AbstractMatrix{<:Real})
    
    # Forward pass - store original spatial for potential restoration
    original_spatial = copy(spatial)
    synthesize!(cfg, sh, spatial)
    
    function synthesize!_pullback(spatial_bar)
        # The gradient w.r.t. sh is analyze(spatial_bar)  
        sh_bar = if spatial_bar === nothing
            NoTangent()
        else
            analyze(cfg, spatial_bar)
        end
        
        # Note: The gradient w.r.t. the pre-allocated spatial array is complex
        # In most cases, this should be NoTangent() since spatial is output
        spatial_bar_out = spatial_bar
        
        return NoTangent(), NoTangent(), sh_bar, spatial_bar_out
    end
    
    return spatial, synthesize!_pullback
end

"""
Adjoint rule for in-place analyze!
"""
function rrule(::typeof(analyze!), 
               cfg::SHTnsConfig,
               spatial::AbstractMatrix{<:Real},
               sh::AbstractVector{<:Real})
    
    # Forward pass
    original_sh = copy(sh)
    analyze!(cfg, spatial, sh)
    
    function analyze!_pullback(sh_bar)
        # The gradient w.r.t. spatial is synthesize(sh_bar)
        spatial_bar = if sh_bar === nothing
            NoTangent()
        else
            synthesize(cfg, sh_bar)
        end
        
        sh_bar_out = sh_bar
        
        return NoTangent(), NoTangent(), spatial_bar, sh_bar_out
    end
    
    return sh, analyze!_pullback
end

# ============================================================================
# Complex field transforms
# ============================================================================

"""
Adjoint rule for complex field synthesis.
"""
function rrule(::typeof(synthesize_complex), 
               cfg::SHTnsConfig,
               sh::AbstractVector{<:Complex})
    
    # Forward pass
    spatial = synthesize_complex(cfg, sh)
    
    function synthesize_complex_pullback(spatial_bar)
        sh_bar = if spatial_bar === nothing
            NoTangent()
        else
            # For complex transforms, the adjoint is still the complex analyze
            analyze_complex(cfg, spatial_bar)
        end
        
        return NoTangent(), NoTangent(), sh_bar
    end
    
    return spatial, synthesize_complex_pullback
end

"""
Adjoint rule for complex field analysis.
"""
function rrule(::typeof(analyze_complex), 
               cfg::SHTnsConfig,
               spatial::AbstractMatrix{<:Complex})
    
    # Forward pass
    sh = analyze_complex(cfg, spatial)
    
    function analyze_complex_pullback(sh_bar)
        spatial_bar = if sh_bar === nothing
            NoTangent()
        else
            synthesize_complex(cfg, sh_bar)
        end
        
        return NoTangent(), NoTangent(), spatial_bar
    end
    
    return sh, analyze_complex_pullback
end

# ============================================================================
# Vector field transforms
# ============================================================================

"""
Adjoint rule for vector field synthesis.
"""
function rrule(::typeof(synthesize_vector), 
               cfg::SHTnsConfig,
               S_lm::AbstractVector{<:Real},
               T_lm::AbstractVector{<:Real})
    
    # Forward pass
    Vt, Vp = synthesize_vector(cfg, S_lm, T_lm)
    
    function synthesize_vector_pullback((Vt_bar, Vp_bar))
        if Vt_bar === nothing || Vp_bar === nothing
            return NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # The adjoint of vector synthesis is vector analysis
        S_bar, T_bar = analyze_vector(cfg, Vt_bar, Vp_bar)
        
        return NoTangent(), NoTangent(), S_bar, T_bar
    end
    
    return (Vt, Vp), synthesize_vector_pullback
end

"""
Adjoint rule for vector field analysis.
"""
function rrule(::typeof(analyze_vector), 
               cfg::SHTnsConfig,
               Vt::AbstractMatrix{<:Real},
               Vp::AbstractMatrix{<:Real})
    
    # Forward pass
    S_lm, T_lm = analyze_vector(cfg, Vt, Vp)
    
    function analyze_vector_pullback((S_bar, T_bar))
        if S_bar === nothing || T_bar === nothing
            return NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # The adjoint of vector analysis is vector synthesis
        Vt_bar, Vp_bar = synthesize_vector(cfg, S_bar, T_bar)
        
        return NoTangent(), NoTangent(), Vt_bar, Vp_bar
    end
    
    return (S_lm, T_lm), analyze_vector_pullback
end

# ============================================================================
# Gradient and curl operations
# ============================================================================

"""
Adjoint rule for gradient computation.
The adjoint of gradient is negative divergence.
"""
function rrule(::typeof(compute_gradient), 
               cfg::SHTnsConfig,
               scalar_lm::AbstractVector{<:Real})
    
    # Forward pass
    ∇θ, ∇φ = compute_gradient(cfg, scalar_lm)
    
    function compute_gradient_pullback((∇θ_bar, ∇φ_bar))
        if ∇θ_bar === nothing || ∇φ_bar === nothing
            return NoTangent(), NoTangent(), NoTangent()
        end
        
        # The adjoint of gradient is the divergence operator
        # For gradient fields, the adjoint corresponds to the spheroidal component
        # In spherical harmonics: ∇f is purely spheroidal, so its adjoint
        # extracts the divergence information from the spheroidal coefficients
        
        # Decompose vector field into spheroidal and toroidal components  
        S_bar, T_bar = analyze_vector(cfg, ∇θ_bar, ∇φ_bar)
        
        # The spheroidal component gives us the divergence = adjoint of gradient
        scalar_bar = S_bar
        
        # Sanity check: T_bar should be small for true gradient fields
        # (Optional: could add a warning if norm(T_bar) is large)
        
        return NoTangent(), NoTangent(), scalar_bar
    end
    
    return (∇θ, ∇φ), compute_gradient_pullback
end

"""
Adjoint rule for curl computation.
The adjoint of curl is also curl (for 2D case).
"""
function rrule(::typeof(compute_curl), 
               cfg::SHTnsConfig,
               toroidal_lm::AbstractVector{<:Real})
    
    # Forward pass
    curlθ, curlφ = compute_curl(cfg, toroidal_lm)
    
    function compute_curl_pullback((curlθ_bar, curlφ_bar))
        if curlθ_bar === nothing || curlφ_bar === nothing
            return NoTangent(), NoTangent(), NoTangent()
        end
        
        # For curl, the adjoint is related to the curl operation itself
        # This is a simplification - the exact adjoint depends on the specific
        # implementation of curl in spherical coordinates
        _, T_bar = analyze_vector(cfg, curlθ_bar, curlφ_bar)
        toroidal_bar = T_bar
        
        return NoTangent(), NoTangent(), toroidal_bar
    end
    
    return (curlθ, curlφ), compute_curl_pullback
end

# ============================================================================
# Power spectrum (non-linear operation)
# ============================================================================

"""
Adjoint rule for power spectrum computation.
This is more complex because power spectrum involves |aₗᵐ|² operations.
"""
function rrule(::typeof(power_spectrum), 
               cfg::SHTnsConfig,
               sh::AbstractVector{<:Real})
    
    # Forward pass
    power = power_spectrum(cfg, sh)
    
    function power_spectrum_pullback(power_bar)
        if power_bar === nothing
            return NoTangent(), NoTangent(), NoTangent()
        end
        
        # For power spectrum P(l) = Σₘ |aₗᵐ|², we have:
        # ∂P(l)/∂aₗᵐ = 2 aₗᵐ
        
        lmax = get_lmax(cfg)
        sh_bar = zeros(eltype(sh), length(sh))
        
        for idx in 1:length(sh)
            l, m = get_lm_from_index(cfg, idx)  # Need to implement this
            sh_bar[idx] = 2 * sh[idx] * power_bar[l + 1]
        end
        
        return NoTangent(), NoTangent(), sh_bar
    end
    
    return power, power_spectrum_pullback
end

# ============================================================================
# Field rotation operations
# ============================================================================

"""
Adjoint rule for field rotation.
Rotation matrices are unitary, so the adjoint is the transpose (inverse rotation).
"""
function rrule(::typeof(rotate_field), 
               cfg::SHTnsConfig,
               sh::AbstractVector{<:Real},
               α::Real, β::Real, γ::Real)
    
    # Forward pass
    sh_rotated = rotate_field(cfg, sh, α, β, γ)
    
    function rotate_field_pullback(sh_rotated_bar)
        if sh_rotated_bar === nothing
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # The adjoint of rotation by (α,β,γ) is rotation by (-γ,-β,-α)
        sh_bar = rotate_field(cfg, sh_rotated_bar, -γ, -β, -α)
        
        # Gradients w.r.t. rotation angles would require more complex computation
        # For now, we return NoTangent for the angle parameters
        α_bar = NoTangent()
        β_bar = NoTangent()  
        γ_bar = NoTangent()
        
        return NoTangent(), NoTangent(), sh_bar, α_bar, β_bar, γ_bar
    end
    
    return sh_rotated, rotate_field_pullback
end

"""
Adjoint rule for spatial field rotation.
"""
function rrule(::typeof(rotate_spatial_field), 
               cfg::SHTnsConfig,
               spatial::AbstractMatrix{<:Real},
               α::Real, β::Real, γ::Real)
    
    # Forward pass
    spatial_rotated = rotate_spatial_field(cfg, spatial, α, β, γ)
    
    function rotate_spatial_field_pullback(spatial_rotated_bar)
        if spatial_rotated_bar === nothing
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # The adjoint is the inverse rotation
        spatial_bar = rotate_spatial_field(cfg, spatial_rotated_bar, -γ, -β, -α)
        
        # Angle gradients
        α_bar = NoTangent()
        β_bar = NoTangent()
        γ_bar = NoTangent()
        
        return NoTangent(), NoTangent(), spatial_bar, α_bar, β_bar, γ_bar
    end
    
    return spatial_rotated, rotate_spatial_field_pullback
end

# ============================================================================
# Allocation functions (these don't need gradients)
# ============================================================================

function rrule(::typeof(allocate_spectral), cfg::SHTnsConfig)
    result = allocate_spectral(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(allocate_spatial), cfg::SHTnsConfig)
    result = allocate_spatial(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(allocate_complex_spectral), cfg::SHTnsConfig)
    result = allocate_complex_spectral(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(allocate_complex_spatial), cfg::SHTnsConfig)
    result = allocate_complex_spatial(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

# ============================================================================
# Configuration and query functions (no gradients needed)
# ============================================================================

function rrule(::typeof(get_lmax), cfg::SHTnsConfig)
    result = get_lmax(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(get_mmax), cfg::SHTnsConfig)
    result = get_mmax(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(get_nlat), cfg::SHTnsConfig)
    result = get_nlat(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(get_nphi), cfg::SHTnsConfig)
    result = get_nphi(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(get_nlm), cfg::SHTnsConfig)
    result = get_nlm(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end

function rrule(::typeof(get_coordinates), cfg::SHTnsConfig)
    result = get_coordinates(cfg)
    return result, _ -> (NoTangent(), NoTangent())
end


end # module
