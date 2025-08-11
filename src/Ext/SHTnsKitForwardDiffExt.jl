module SHTnsKitForwardDiffExt

using SHTnsKit
import ForwardDiff
using ForwardDiff: Dual, value, partials

"""
ForwardDiff extension for SHTnsKit.jl

This extension enables forward-mode automatic differentiation through 
spherical harmonic transforms using ForwardDiff.jl. It provides differentiable
versions of the main transform functions by implementing appropriate rules
for handling Dual numbers.

The implementation leverages the linearity of spherical harmonic transforms
to efficiently compute derivatives.
"""

# ============================================================================
# Forward-mode AD rules for basic transforms
# ============================================================================

"""
Forward-mode AD rule for synthesize function.
Handles ForwardDiff.Dual inputs by applying the transform to both 
value and derivative parts separately (since SHT is linear).
"""
function SHTnsKit.synthesize(cfg::SHTnsKit.SHTnsConfig, 
                            sh::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values and partials
    sh_values = value.(sh)
    sh_partials = partials.(sh)
    
    # Apply transform to values
    spatial_values = SHTnsKit.synthesize(cfg, sh_values)
    
    # Apply transform to each partial derivative component
    # This works because spherical harmonic transforms are linear
    spatial_partials = ntuple(N) do i
        partials_i = map(p -> p[i], sh_partials)
        SHTnsKit.synthesize(cfg, partials_i)
    end
    
    # Reconstruct dual numbers in spatial domain
    spatial_duals = map(eachindex(spatial_values)) do idx
        Dual{T}(spatial_values[idx], 
               ntuple(i -> spatial_partials[i][idx], N))
    end
    
    return reshape(spatial_duals, size(spatial_values))
end

"""
Forward-mode AD rule for analyze function.
"""
function SHTnsKit.analyze(cfg::SHTnsKit.SHTnsConfig, 
                         spatial::AbstractMatrix{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values and partials
    spatial_values = value.(spatial)
    spatial_partials = partials.(spatial)
    
    # Apply transform to values
    sh_values = SHTnsKit.analyze(cfg, spatial_values)
    
    # Apply transform to each partial derivative component
    sh_partials = ntuple(N) do i
        partials_i = map(p -> p[i], spatial_partials)
        SHTnsKit.analyze(cfg, partials_i)
    end
    
    # Reconstruct dual numbers in spectral domain
    sh_duals = map(eachindex(sh_values)) do idx
        Dual{T}(sh_values[idx],
               ntuple(i -> sh_partials[i][idx], N))
    end
    
    return sh_duals
end

# ============================================================================
# In-place variants (more memory efficient)
# ============================================================================

"""
Forward-mode AD rule for in-place synthesize!
"""
function SHTnsKit.synthesize!(cfg::SHTnsKit.SHTnsConfig,
                             sh::AbstractVector{<:Dual{T,V,N}},
                             spatial::AbstractMatrix{<:Dual{T,V,N}}) where {T,V,N}
    # Work with temporary value arrays to avoid modifying input
    sh_values = value.(sh)
    spatial_values = similar(spatial, V)
    
    # Forward transform for values
    SHTnsKit.synthesize!(cfg, sh_values, spatial_values)
    
    # Forward transform for each partial component
    for i in 1:N
        sh_partials_i = map(p -> p[i], partials.(sh))
        spatial_partials_i = similar(spatial, V)
        SHTnsKit.synthesize!(cfg, sh_partials_i, spatial_partials_i)
        
        # Update the spatial array with dual numbers
        for idx in eachindex(spatial)
            if i == 1
                spatial[idx] = Dual{T}(spatial_values[idx], 
                                     ntuple(j -> j == 1 ? spatial_partials_i[idx] : zero(V), N))
            else
                # Add this partial to existing dual number
                old_partials = partials(spatial[idx])
                new_partials = ntuple(j -> j == i ? spatial_partials_i[idx] : old_partials[j], N)
                spatial[idx] = Dual{T}(value(spatial[idx]), new_partials)
            end
        end
    end
    
    return spatial
end

"""
Forward-mode AD rule for in-place analyze!
"""
function SHTnsKit.analyze!(cfg::SHTnsKit.SHTnsConfig,
                          spatial::AbstractMatrix{<:Dual{T,V,N}},
                          sh::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # Work with temporary value arrays
    spatial_values = value.(spatial)
    sh_values = similar(sh, V)
    
    # Backward transform for values
    SHTnsKit.analyze!(cfg, spatial_values, sh_values)
    
    # Backward transform for each partial component
    for i in 1:N
        spatial_partials_i = map(p -> p[i], partials.(spatial))
        sh_partials_i = similar(sh, V)
        SHTnsKit.analyze!(cfg, spatial_partials_i, sh_partials_i)
        
        # Update the spectral array with dual numbers
        for idx in eachindex(sh)
            if i == 1
                sh[idx] = Dual{T}(sh_values[idx],
                                ntuple(j -> j == 1 ? sh_partials_i[idx] : zero(V), N))
            else
                # Add this partial to existing dual number
                old_partials = partials(sh[idx])
                new_partials = ntuple(j -> j == i ? sh_partials_i[idx] : old_partials[j], N)
                sh[idx] = Dual{T}(value(sh[idx]), new_partials)
            end
        end
    end
    
    return sh
end

# ============================================================================
# Complex field transforms
# ============================================================================

"""
Forward-mode AD rule for complex field synthesis.
"""
function SHTnsKit.synthesize_complex(cfg::SHTnsKit.SHTnsConfig,
                                    sh::AbstractVector{<:Complex{<:Dual{T,V,N}}}) where {T,V,N}
    # Split into real and imaginary parts
    sh_real = real.(sh)
    sh_imag = imag.(sh)
    
    # Apply AD rules to each part
    spatial_real = SHTnsKit.synthesize(cfg, sh_real)
    spatial_imag = SHTnsKit.synthesize(cfg, sh_imag)
    
    # Recombine into complex result
    return complex.(spatial_real, spatial_imag)
end

"""
Forward-mode AD rule for complex field analysis.
"""
function SHTnsKit.analyze_complex(cfg::SHTnsKit.SHTnsConfig,
                                 spatial::AbstractMatrix{<:Complex{<:Dual{T,V,N}}}) where {T,V,N}
    # Split into real and imaginary parts
    spatial_real = real.(spatial)
    spatial_imag = imag.(spatial)
    
    # Apply AD rules to each part
    sh_real = SHTnsKit.analyze(cfg, spatial_real)
    sh_imag = SHTnsKit.analyze(cfg, spatial_imag)
    
    # Recombine into complex result
    return complex.(sh_real, sh_imag)
end

# ============================================================================
# Vector field transforms
# ============================================================================

"""
Forward-mode AD rule for vector field synthesis.
"""
function SHTnsKit.synthesize_vector(cfg::SHTnsKit.SHTnsConfig,
                                   S_lm::AbstractVector{<:Dual{T,V,N}},
                                   T_lm::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values
    S_values = value.(S_lm)
    T_values = value.(T_lm)
    S_partials = partials.(S_lm)
    T_partials = partials.(T_lm)
    
    # Apply vector transform to values
    Vt_values, Vp_values = SHTnsKit.synthesize_vector(cfg, S_values, T_values)
    
    # Apply vector transform to each partial component
    Vt_partials = ntuple(N) do i
        S_partials_i = map(p -> p[i], S_partials)
        T_partials_i = map(p -> p[i], T_partials)
        Vt_i, _ = SHTnsKit.synthesize_vector(cfg, S_partials_i, T_partials_i)
        Vt_i
    end
    
    Vp_partials = ntuple(N) do i
        S_partials_i = map(p -> p[i], S_partials)
        T_partials_i = map(p -> p[i], T_partials)
        _, Vp_i = SHTnsKit.synthesize_vector(cfg, S_partials_i, T_partials_i)
        Vp_i
    end
    
    # Reconstruct dual numbers
    Vt_duals = map(eachindex(Vt_values)) do idx
        Dual{T}(Vt_values[idx], 
               ntuple(i -> Vt_partials[i][idx], N))
    end
    
    Vp_duals = map(eachindex(Vp_values)) do idx
        Dual{T}(Vp_values[idx],
               ntuple(i -> Vp_partials[i][idx], N))
    end
    
    return reshape(Vt_duals, size(Vt_values)), reshape(Vp_duals, size(Vp_values))
end

"""
Forward-mode AD rule for vector field analysis.
"""
function SHTnsKit.analyze_vector(cfg::SHTnsKit.SHTnsConfig,
                                Vt::AbstractMatrix{<:Dual{T,V,N}},
                                Vp::AbstractMatrix{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values
    Vt_values = value.(Vt)
    Vp_values = value.(Vp)
    Vt_partials = partials.(Vt)
    Vp_partials = partials.(Vp)
    
    # Apply vector transform to values
    S_values, T_values = SHTnsKit.analyze_vector(cfg, Vt_values, Vp_values)
    
    # Apply vector transform to each partial component
    S_partials = ntuple(N) do i
        Vt_partials_i = map(p -> p[i], Vt_partials)
        Vp_partials_i = map(p -> p[i], Vp_partials)
        S_i, _ = SHTnsKit.analyze_vector(cfg, Vt_partials_i, Vp_partials_i)
        S_i
    end
    
    T_partials = ntuple(N) do i
        Vt_partials_i = map(p -> p[i], Vt_partials)
        Vp_partials_i = map(p -> p[i], Vp_partials)
        _, T_i = SHTnsKit.analyze_vector(cfg, Vt_partials_i, Vp_partials_i)
        T_i
    end
    
    # Reconstruct dual numbers
    S_duals = map(eachindex(S_values)) do idx
        Dual{T}(S_values[idx],
               ntuple(i -> S_partials[i][idx], N))
    end
    
    T_duals = map(eachindex(T_values)) do idx
        Dual{T}(T_values[idx],
               ntuple(i -> T_partials[i][idx], N))
    end
    
    return S_duals, T_duals
end

# ============================================================================
# Utility functions for gradient and curl operations
# ============================================================================

"""
Forward-mode AD rule for gradient computation.
"""
function SHTnsKit.compute_gradient(cfg::SHTnsKit.SHTnsConfig,
                                  scalar_lm::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values and partials
    scalar_values = value.(scalar_lm)
    scalar_partials = partials.(scalar_lm)
    
    # Apply gradient to values
    ∇θ_values, ∇φ_values = SHTnsKit.compute_gradient(cfg, scalar_values)
    
    # Apply gradient to each partial component
    ∇θ_partials = ntuple(N) do i
        scalar_partials_i = map(p -> p[i], scalar_partials)
        ∇θ_i, _ = SHTnsKit.compute_gradient(cfg, scalar_partials_i)
        ∇θ_i
    end
    
    ∇φ_partials = ntuple(N) do i
        scalar_partials_i = map(p -> p[i], scalar_partials)
        _, ∇φ_i = SHTnsKit.compute_gradient(cfg, scalar_partials_i)
        ∇φ_i
    end
    
    # Reconstruct dual numbers
    ∇θ_duals = map(eachindex(∇θ_values)) do idx
        Dual{T}(∇θ_values[idx],
               ntuple(i -> ∇θ_partials[i][idx], N))
    end
    
    ∇φ_duals = map(eachindex(∇φ_values)) do idx
        Dual{T}(∇φ_values[idx],
               ntuple(i -> ∇φ_partials[i][idx], N))
    end
    
    return reshape(∇θ_duals, size(∇θ_values)), reshape(∇φ_duals, size(∇φ_values))
end

"""
Forward-mode AD rule for curl computation.
"""
function SHTnsKit.compute_curl(cfg::SHTnsKit.SHTnsConfig,
                              toroidal_lm::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # Extract values and partials
    toroidal_values = value.(toroidal_lm)
    toroidal_partials = partials.(toroidal_lm)
    
    # Apply curl to values
    curlθ_values, curlφ_values = SHTnsKit.compute_curl(cfg, toroidal_values)
    
    # Apply curl to each partial component
    curlθ_partials = ntuple(N) do i
        toroidal_partials_i = map(p -> p[i], toroidal_partials)
        curlθ_i, _ = SHTnsKit.compute_curl(cfg, toroidal_partials_i)
        curlθ_i
    end
    
    curlφ_partials = ntuple(N) do i
        toroidal_partials_i = map(p -> p[i], toroidal_partials)
        _, curlφ_i = SHTnsKit.compute_curl(cfg, toroidal_partials_i)
        curlφ_i
    end
    
    # Reconstruct dual numbers
    curlθ_duals = map(eachindex(curlθ_values)) do idx
        Dual{T}(curlθ_values[idx],
               ntuple(i -> curlθ_partials[i][idx], N))
    end
    
    curlφ_duals = map(eachindex(curlφ_values)) do idx
        Dual{T}(curlφ_values[idx],
               ntuple(i -> curlφ_partials[i][idx], N))
    end
    
    return reshape(curlθ_duals, size(curlθ_values)), reshape(curlφ_duals, size(curlφ_values))
end

# ============================================================================
# Power spectrum (special handling needed due to norm operations)
# ============================================================================

"""
Forward-mode AD rule for power spectrum computation.
This is more complex because it involves |aₗᵐ|² operations.
"""
function SHTnsKit.power_spectrum(cfg::SHTnsKit.SHTnsConfig,
                                sh::AbstractVector{<:Dual{T,V,N}}) where {T,V,N}
    # For power spectrum P(l) = Σₘ |aₗᵐ|², we need to handle the square carefully
    # d/dx |f(x)|² = 2 Re(f̄(x) f'(x)) for complex f, or 2 f(x) f'(x) for real f
    
    sh_values = value.(sh)
    sh_partials = partials.(sh)
    
    # Compute power spectrum of values
    power_values = SHTnsKit.power_spectrum(cfg, sh_values)
    
    # Compute derivatives of power spectrum
    # For each mode l,m: d/dx |aₗᵐ|² = 2 aₗᵐ * daₗᵐ/dx
    lmax = SHTnsKit.get_lmax(cfg)
    power_partials = ntuple(N) do i
        power_derivs = zeros(V, lmax + 1)
        
        for idx in 1:length(sh)
            l, m = SHTnsKit.get_lm_from_index(cfg, idx)  # This function needs to be implemented
            power_derivs[l + 1] += 2 * sh_values[idx] * sh_partials[idx][i]
        end
        
        power_derivs
    end
    
    # Reconstruct dual numbers for power spectrum
    power_duals = map(eachindex(power_values)) do idx
        Dual{T}(power_values[idx],
               ntuple(i -> power_partials[i][idx], N))
    end
    
    return power_duals
end

end # module