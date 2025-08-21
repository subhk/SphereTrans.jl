"""
Complex-valued spherical harmonic transforms.
Handles transforms of complex scalar fields on the sphere.
"""

"""
    cplx_sh_to_spat!(cfg::SHTnsConfig{T}, 
                    sh_coeffs::AbstractVector{Complex{T}},
                    spatial_data::AbstractMatrix{Complex{T}}) where T

Transform complex spherical harmonic coefficients to complex spatial field.
Synthesis for complex-valued fields: c_lm → f(θ,φ)

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Input complex spherical harmonic coefficients (length nlm)
- `spatial_data`: Output complex spatial field (nlat × nphi, pre-allocated)
"""
function cplx_sh_to_spat!(cfg::SHTnsConfig{T},
                         sh_coeffs::AbstractVector{Complex{T}},
                         spatial_data::AbstractMatrix{Complex{T}}) where T
    validate_config(cfg)
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch")
    
    lock(cfg.lock) do
        _cplx_sh_to_spat_impl!(cfg, sh_coeffs, spatial_data)
    end
    
    return spatial_data
end

"""
    cplx_spat_to_sh!(cfg::SHTnsConfig{T},
                    spatial_data::AbstractMatrix{Complex{T}},
                    sh_coeffs::AbstractVector{Complex{T}}) where T

Transform complex spatial field to complex spherical harmonic coefficients.
Analysis for complex-valued fields: f(θ,φ) → c_lm

# Arguments  
- `cfg`: SHTns configuration
- `spatial_data`: Input complex spatial field (nlat × nphi)
- `sh_coeffs`: Output complex spherical harmonic coefficients (length nlm, pre-allocated)
"""
function cplx_spat_to_sh!(cfg::SHTnsConfig{T},
                         spatial_data::AbstractMatrix{Complex{T}},
                         sh_coeffs::AbstractVector{Complex{T}}) where T
    validate_config(cfg)
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch")
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    lock(cfg.lock) do
        _cplx_spat_to_sh_impl!(cfg, spatial_data, sh_coeffs)
    end
    
    return sh_coeffs
end

# Non-mutating versions

"""
    cplx_sh_to_spat(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T

Transform complex spherical harmonic coefficients to spatial field (allocating).
"""
function cplx_sh_to_spat(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    spatial_data = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
    return cplx_sh_to_spat!(cfg, sh_coeffs, spatial_data)
end

"""
    cplx_spat_to_sh(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}}) where T

Transform complex spatial field to spherical harmonic coefficients (allocating).
"""
function cplx_spat_to_sh(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}}) where T
    sh_coeffs = Vector{Complex{T}}(undef, cfg.nlm)
    return cplx_spat_to_sh!(cfg, spatial_data, sh_coeffs)
end

# Implementation functions

"""
    _cplx_sh_to_spat_impl!(cfg::SHTnsConfig{T},
                          sh_coeffs::AbstractVector{Complex{T}},
                          spatial_data::AbstractMatrix{Complex{T}}) where T

Internal implementation of complex spherical harmonic synthesis.
Uses full complex FFTs in the azimuthal direction.
"""
function _cplx_sh_to_spat_impl!(cfg::SHTnsConfig{T},
                               sh_coeffs::AbstractVector{Complex{T}},
                               spatial_data::AbstractMatrix{Complex{T}}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Allocate working array for complex Fourier coefficients
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi)
    fill!(fourier_coeffs, zero(Complex{T}))
    
    # For each azimuthal mode m (now including negative m)
    for m in -cfg.mmax:cfg.mmax
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue
        
        # Fourier coefficient index (handle negative frequencies)
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        
        # Compute mode coefficients for all latitudes
        mode_coeffs = Vector{Complex{T}}(undef, nlat)
        fill!(mode_coeffs, zero(Complex{T}))
        
        # Sum over l for this m: Σ_l c_{l,m} P_l^|m|(cos θ) e^{im φ}
        for i in 1:nlat
            value = zero(Complex{T})
            
            for (coeff_idx, (l, m_coeff)) in enumerate(cfg.lm_indices)
                if m_coeff == abs(m) && l >= abs(m)
                    plm_val = cfg.plm_cache[i, coeff_idx]
                    
                    if m >= 0
                        # Positive m: use coefficient directly
                        value += sh_coeffs[coeff_idx] * plm_val
                    else
                        # Negative m: use conjugate symmetry
                        # c_{l,-m} = (-1)^m * conj(c_{l,m})
                        phase_factor = m % 2 == 0 ? 1 : -1
                        value += phase_factor * conj(sh_coeffs[coeff_idx]) * plm_val
                    end
                end
            end
            
            mode_coeffs[i] = value
        end
        
        # Insert into Fourier array
        for i in 1:nlat
            fourier_coeffs[i, m_idx] = mode_coeffs[i]
        end
    end
    
    # Transform from Fourier coefficients to spatial domain
    for i in 1:nlat
        azimuthal_fft_complex_backward!(cfg, view(fourier_coeffs, i, :), view(spatial_data, i, :))
    end
    
    return nothing
end

"""
    _cplx_spat_to_sh_impl!(cfg::SHTnsConfig{T},
                          spatial_data::AbstractMatrix{Complex{T}},
                          sh_coeffs::AbstractVector{Complex{T}}) where T

Internal implementation of complex spherical harmonic analysis.
"""
function _cplx_spat_to_sh_impl!(cfg::SHTnsConfig{T},
                               spatial_data::AbstractMatrix{Complex{T}},
                               sh_coeffs::AbstractVector{Complex{T}}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Transform spatial data to complex Fourier coefficients
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi)
    for i in 1:nlat
        azimuthal_fft_complex_forward!(cfg, view(spatial_data, i, :), view(fourier_coeffs, i, :))
    end
    
    # For each (l,m) coefficient
    fill!(sh_coeffs, zero(Complex{T}))
    
    for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        # Handle both positive and negative m (but store only |m|)
        if m <= nphi ÷ 2
            # Get Fourier mode data
            mode_data = Vector{Complex{T}}(undef, nlat)
            
            # Extract positive m mode
            m_idx = m + 1
            for i in 1:nlat
                mode_data[i] = fourier_coeffs[i, m_idx]
            end
            
            # Integrate over latitude using quadrature
            integral = zero(Complex{T})
            for i in 1:nlat
                plm_val = cfg.plm_cache[i, coeff_idx]
                weight = cfg.gauss_weights[i]
                integral += mode_data[i] * plm_val * weight
            end
            
            # Store result with proper normalization
            normalization = _get_complex_normalization(cfg.norm, l, m)
            sh_coeffs[coeff_idx] = integral * normalization
        end
    end
    
    return nothing
end

"""
    _get_complex_normalization(norm::SHTnsNorm, l::Int, m::Int) -> T

Get normalization factor for complex spherical harmonics.
Different conventions exist for complex vs real spherical harmonics.
"""
function _get_complex_normalization(norm::SHTnsNorm, l::Int, m::Int)
    if norm == SHT_ORTHONORMAL
        # 4π normalization for complex harmonics
        return 1.0
    elseif norm == SHT_FOURPI
        return 1.0 / (4π)
    elseif norm == SHT_SCHMIDT
        # Schmidt normalization
        if m == 0
            return 1.0
        else
            return 1.0 / sqrt(2.0)
        end
    else
        return 1.0
    end
end

# Utility functions for complex transforms

"""
    allocate_complex_spectral(cfg::SHTnsConfig{T}) -> Vector{Complex{T}}

Allocate array for complex spherical harmonic coefficients.
"""
function allocate_complex_spectral(cfg::SHTnsConfig{T}) where T
    return Vector{Complex{T}}(undef, cfg.nlm)
end

"""
    allocate_complex_spatial(cfg::SHTnsConfig{T}) -> Matrix{Complex{T}}

Allocate array for complex spatial field.
"""
function allocate_complex_spatial(cfg::SHTnsConfig{T}) where T
    return Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
end

"""
    synthesize_complex(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T

Convenience function for complex synthesis (allocating version).
"""
function synthesize_complex(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    return cplx_sh_to_spat(cfg, sh_coeffs)
end

"""
    analyze_complex(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}}) where T

Convenience function for complex analysis (allocating version).
"""
function analyze_complex(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}}) where T
    return cplx_spat_to_sh(cfg, spatial_data)
end

"""
    create_complex_test_field(cfg::SHTnsConfig{T}, l::Int, m::Int) -> Matrix{Complex{T}}

Create a test complex field consisting of a single spherical harmonic Y_l^m.
Useful for testing and validation.

# Arguments
- `cfg`: SHTns configuration
- `l`: Spherical harmonic degree
- `m`: Spherical harmonic order

# Returns
- Complex spatial field containing Y_l^m(θ,φ)
"""
function create_complex_test_field(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    0 <= l <= cfg.lmax || error("l must be in range [0, lmax]")
    abs(m) <= min(l, cfg.mmax) || error("m must be in range [-min(l,mmax), min(l,mmax)]")
    
    # Create coefficients with single mode
    sh_coeffs = zeros(Complex{T}, cfg.nlm)
    
    # Find the coefficient index for (l, |m|)
    for (idx, (l_idx, m_idx)) in enumerate(cfg.lm_indices)
        if l_idx == l && m_idx == abs(m)
            sh_coeffs[idx] = one(Complex{T})
            break
        end
    end
    
    # Synthesize to spatial domain
    return cplx_sh_to_spat(cfg, sh_coeffs)
end