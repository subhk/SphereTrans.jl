"""
Complex-valued spherical harmonic transforms (canonical).
Handles transforms of complex scalar fields on the sphere, storing full
coefficient sets for m ∈ [-min(l,mmax), ..., 0, ..., +min(l,mmax)].
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
    length(sh_coeffs) == _cplx_nlm(cfg) || error("sh_coeffs length must equal complex nlm")
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
    length(sh_coeffs) == _cplx_nlm(cfg) || error("sh_coeffs length must equal complex nlm")
    
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
    sh_coeffs = Vector{Complex{T}}(undef, _cplx_nlm(cfg))
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

    idx_list = _cplx_lm_indices(cfg)

    # For each azimuthal mode m (including negative)
    for m in -cfg.mmax:cfg.mmax
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue

        m_idx = m >= 0 ? m + 1 : nphi + m + 1

        # Compute mode coefficients for all latitudes
        for i in 1:nlat
            value = zero(Complex{T})
            for (coeff_idx, (l2, m2)) in enumerate(idx_list)
                if l2 >= abs(m) && m2 == m
                    # plm_cache stores only |m| rows; map to |m|
                    # Find cfg.lm_indices index for (l, |m|)
                    for (k, (ll, mm)) in enumerate(cfg.lm_indices)
                        if ll == l2 && mm == abs(m)
                            plm_val = cfg.plm_cache[i, k]
                            value += sh_coeffs[coeff_idx] * plm_val
                            break
                        end
                    end
                end
            end
            fourier_coeffs[i, m_idx] = value
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

    # For each (l,m) coefficient over full m range
    fill!(sh_coeffs, zero(Complex{T}))
    idx_list = _cplx_lm_indices(cfg)
    for (coeff_idx, (l, m)) in enumerate(idx_list)
        abs(m) <= nphi ÷ 2 || continue
        # Extract Fourier mode m
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        # Integrate over latitude using quadrature
        integral = zero(Complex{T})
        for i in 1:nlat
            # Map to |m| for plm
            # Find cfg.lm_indices index for (l, |m|)
            for (k, (ll, mm)) in enumerate(cfg.lm_indices)
                if ll == l && mm == abs(m)
                    plm_val = cfg.plm_cache[i, k]
                    weight = cfg.gauss_weights[i]
                    integral += fourier_coeffs[i, m_idx] * plm_val * weight
                    break
                end
            end
        end
        normalization = _get_complex_normalization(cfg.norm, l, abs(m))
        sh_coeffs[coeff_idx] = integral * normalization
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
    return Vector{Complex{T}}(undef, _cplx_nlm(cfg))
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
    cplx_spectral_derivative_phi(cfg, sh_coeffs)

Compute spectral φ-derivative for complex coefficients: (∂/∂φ) maps c_{l,m} -> i*m*c_{l,m}.
Returns a new coefficient vector of the same length.
"""
function cplx_spectral_derivative_phi(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    length(sh_coeffs) == _cplx_nlm(cfg) || error("length mismatch for complex coefficients")
    out = similar(sh_coeffs)
    for (idx, (l, m)) in enumerate(_cplx_lm_indices(cfg))
        out[idx] = Complex{T}(0, m) * sh_coeffs[idx]
    end
    return out
end

"""
    cplx_spectral_laplacian(cfg, sh_coeffs)

Apply surface Laplacian (Δ_S) in spectral domain for complex coefficients.
For unit sphere: Δ_S Y_l^m = -l(l+1) Y_l^m.
"""
function cplx_spectral_laplacian(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    length(sh_coeffs) == _cplx_nlm(cfg) || error("length mismatch for complex coefficients")
    out = similar(sh_coeffs)
    for (idx, (l, m)) in enumerate(_cplx_lm_indices(cfg))
        out[idx] = -T(l*(l+1)) * sh_coeffs[idx]
    end
    return out
end

"""
    cplx_spatial_derivatives(cfg, sh_coeffs)

Compute spatial derivatives (∂θ f, ∂φ f) from complex spectral coefficients.
Uses analytical θ-derivatives (via associated Legendre recurrences) and exact φ FFT factors.
Returns two matrices (nlat × nphi): (dθ, dφ).
"""
function cplx_spatial_derivatives(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    length(sh_coeffs) == _cplx_nlm(cfg) || error("length mismatch for complex coefficients")

    # Build Fourier arrays for dθ and f
    dtheta_fourier = Matrix{Complex{T}}(undef, nlat, nphi)
    fill!(dtheta_fourier, zero(Complex{T}))
    fourier_f = Matrix{Complex{T}}(undef, nlat, nphi)
    fill!(fourier_f, zero(Complex{T}))

    idx_list = _cplx_lm_indices(cfg)
    # Accumulate over m
    for m in -cfg.mmax:cfg.mmax
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        for i in 1:nlat
            val_f = zero(Complex{T})
            val_dt = zero(Complex{T})
            theta = cfg.theta_grid[i]
            for (idx, (l, mm)) in enumerate(idx_list)
                if mm == m && l >= abs(m)
                    # Index for (l, |m|) in plm cache
                    k_plm = _find_plm_index(cfg, l, abs(m))
                    plm = cfg.plm_cache[i, k_plm]
                    dplm = _plm_dtheta(cfg, l, m, theta, i)
                    c = sh_coeffs[idx]
                    val_f += c * plm
                    val_dt += c * dplm
                end
            end
            dtheta_fourier[i, m_idx] = val_dt
            fourier_f[i, m_idx] = val_f
        end
    end

    # Build dφ via multiplying Fourier by i*m and inverse FFT
    dphi_fourier = similar(fourier_f)
    for m in 0:(nphi-1)
        # Map to signed m for factor; but fourier_f uses full C2C indexing
        signed_m = m <= nphi÷2 ? m : m - nphi
        factor = Complex{T}(0, signed_m)
        @inbounds dphi_fourier[:, m+1] .= factor .* fourier_f[:, m+1]
    end

    # Inverse FFTs to spatial
    dtheta_spatial = Matrix{Complex{T}}(undef, nlat, nphi)
    dphi_spatial = Matrix{Complex{T}}(undef, nlat, nphi)
    for i in 1:nlat
        azimuthal_fft_complex_backward!(cfg, view(dtheta_fourier, i, :), view(dtheta_spatial, i, :))
        azimuthal_fft_complex_backward!(cfg, view(dphi_fourier, i, :), view(dphi_spatial, i, :))
    end
    return dtheta_spatial, dphi_spatial
end

# Local helpers for derivative evaluation
function _find_plm_index(cfg::SHTnsConfig, l::Int, m::Int)
    @inbounds for (k, (ll, mm)) in enumerate(cfg.lm_indices)
        if ll == l && mm == m
            return k
        end
    end
    error("plm index not found for (l=$l, m=$m)")
end

function _plm_dtheta(cfg::SHTnsConfig{T}, l::Int, m::Int, theta::T, lat_idx::Int) where T
    if l == 0
        return zero(T)
    end
    k_lm = _find_plm_index(cfg, l, abs(m))
    k_lm1 = l-1 >= 0 ? _find_plm_index(cfg, l-1, abs(m)) : 0
    Plm = cfg.plm_cache[lat_idx, k_lm]
    Plm1 = k_lm1 == 0 ? zero(T) : cfg.plm_cache[lat_idx, k_lm1]
    x = cos(theta)
    s = sin(theta)
    if abs(s) < T(1e-12)
        return zero(T)
    end
    return (l * x * Plm - (l + abs(m)) * Plm1) / s
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
    -min(l, cfg.mmax) <= m <= min(l, cfg.mmax) || error("m out of range for l and mmax")
    
    # Create coefficients with single mode
    sh_coeffs = zeros(Complex{T}, _cplx_nlm(cfg))
    
    # Find the coefficient index for (l, m)
    for (idx, (ll, mm)) in enumerate(_cplx_lm_indices(cfg))
        if ll == l && mm == m
            sh_coeffs[idx] = one(Complex{T})
            break
        end
    end
    
    # Synthesize to spatial domain
    return cplx_sh_to_spat(cfg, sh_coeffs)
end

# Internal helpers for complex coefficient indexing
function _cplx_lm_indices(cfg::SHTnsConfig)
    idx = Tuple{Int,Int}[]
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # negative m down to -mres in steps of -mres
        for m in -maxm:-cfg.mres:-cfg.mres
            push!(idx, (l, m))
        end
        # m=0
        push!(idx, (l, 0))
        # positive m in steps of mres
        for m in cfg.mres:cfg.mres:maxm
            push!(idx, (l, m))
        end
    end
    return idx
end

function _cplx_nlm(cfg::SHTnsConfig)
    # Count full m for each l with mres
    total = 0
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        if cfg.mres == 1
            total += 2*maxm + 1
        else
            # negative: m = -mres, -2mres, ... >= -maxm
            cnt_neg = maxm ÷ cfg.mres
            cnt_pos = cnt_neg
            total += cnt_neg + 1 + cnt_pos
        end
    end
    return total
end
