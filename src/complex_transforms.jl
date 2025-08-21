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

    # For each azimuthal mode m (including negative)
    for m in -cfg.mmax:cfg.mmax
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue

        m_idx = m >= 0 ? m + 1 : nphi + m + 1

        # Compute mode coefficients for all latitudes
        for i in 1:nlat
            value = zero(Complex{T})
            # Sum over pregrouped entries for this m
            for (coeff_idx, k, l2) in get(_cplx_mode_groups(cfg), m, NTuple{3,Int}[])
                l2 >= abs(m) || continue
                plm_val = cfg.plm_cache[i, k]
                value += sh_coeffs[coeff_idx] * plm_val
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
    groups = _cplx_mode_groups(cfg)
    for (m, entries) in groups
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        # Precompute latitudinal weighted transform for this m
        for (coeff_idx, k, l) in entries
            l >= abs(m) || continue
            integral = zero(Complex{T})
            for i in 1:nlat
                plm_val = cfg.plm_cache[i, k]
                weight = cfg.gauss_weights[i]
                integral += fourier_coeffs[i, m_idx] * plm_val * weight
            end
            norm = _get_complex_normalization(cfg.norm, l, abs(m))
            sh_coeffs[coeff_idx] = integral * norm
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

    groups = _cplx_mode_groups(cfg)
    # Accumulate over m
    for m in -cfg.mmax:cfg.mmax
        abs(m) <= nphi ÷ 2 || continue
        (m == 0 || abs(m) % cfg.mres == 0) || continue
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        for i in 1:nlat
            val_f = zero(Complex{T})
            val_dt = zero(Complex{T})
            theta = cfg.theta_grid[i]
            for (coeff_idx, k, l) in get(groups, m, NTuple{3,Int}[])
                l >= abs(m) || continue
                plm = cfg.plm_cache[i, k]
                dplm = _plm_dtheta(cfg, l, m, theta, i)
                c = sh_coeffs[coeff_idx]
                val_f += c * plm
                val_dt += c * dplm
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
    return SHTnsKit.find_plm_index(cfg, l, m)
end

"""
    cplx_spectral_gradient_spatial(cfg, sh_coeffs)

Compute the surface gradient of a complex scalar field in spatial components (θ, φ components).
Returns (gθ, gφ) with size (nlat × nphi), where gθ = ∂θ f and gφ = (1/sinθ) ∂φ f.
"""
function cplx_spectral_gradient_spatial(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}) where T
    dθ, dφ = cplx_spatial_derivatives(cfg, sh_coeffs)
    nlat, nphi = cfg.nlat, cfg.nphi
    gθ = dθ
    gφ = Matrix{Complex{T}}(undef, nlat, nphi)
    @inbounds for i in 1:nlat
        s = sin(cfg.theta_grid[i])
        invs = s > 1e-12 ? (one(T)/s) : zero(T)
        for j in 1:nphi
            gφ[i, j] = invs * dφ[i, j]
        end
    end
    return gθ, gφ
end

"""
    cplx_divergence_spectral(cfg, sph_coeffs, tor_coeffs)

Compute spectral divergence of a tangential vector field decomposed as spheroidal/toroidal.
In spectral domain: div_lm = -l(l+1) * S_lm; toroidal part is divergence-free.
Returns a vector of complex spectral coefficients matching `_cplx_nlm(cfg)`.
"""
function cplx_divergence_spectral(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{Complex{T}}, tor_coeffs::AbstractVector{Complex{T}}) where T
    length(sph_coeffs) == _cplx_nlm(cfg) || error("length mismatch for sph_coeffs")
    length(tor_coeffs) == _cplx_nlm(cfg) || error("length mismatch for tor_coeffs")
    out = similar(sph_coeffs)
    for (idx, (l, m)) in enumerate(_cplx_lm_indices(cfg))
        out[idx] = -T(l*(l+1)) * sph_coeffs[idx]
    end
    return out
end

"""
    cplx_vorticity_spectral(cfg, sph_coeffs, tor_coeffs)

Compute spectral vertical vorticity (radial curl) of a tangential vector field.
In spectral domain: vort_lm = -l(l+1) * T_lm; spheroidal part is curl-free.
Returns a vector of complex spectral coefficients matching `_cplx_nlm(cfg)`.
"""
function cplx_vorticity_spectral(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{Complex{T}}, tor_coeffs::AbstractVector{Complex{T}}) where T
    length(sph_coeffs) == _cplx_nlm(cfg) || error("length mismatch for sph_coeffs")
    length(tor_coeffs) == _cplx_nlm(cfg) || error("length mismatch for tor_coeffs")
    out = similar(tor_coeffs)
    for (idx, (l, m)) in enumerate(_cplx_lm_indices(cfg))
        out[idx] = -T(l*(l+1)) * tor_coeffs[idx]
    end
    return out
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
    cplx_vector_from_spheroidal(cfg, S_coeffs)

Construct tangential vector field from spheroidal potential S (complex spectral coefficients).
Returns (uθ, uφ) where uθ = ∂θ S and uφ = (1/sinθ) ∂φ S, both complex spatial matrices.
"""
function cplx_vector_from_spheroidal(cfg::SHTnsConfig{T}, S_coeffs::AbstractVector{Complex{T}}) where T
    gθ, gφ = cplx_spectral_gradient_spatial(cfg, S_coeffs)
    return gθ, gφ
end

"""
    cplx_vector_from_toroidal(cfg, T_coeffs)

Construct tangential vector field from toroidal potential T (complex spectral coefficients).
Returns (uθ, uφ) where uθ = (1/sinθ) ∂φ T and uφ = -∂θ T, both complex spatial matrices.
"""
function cplx_vector_from_toroidal(cfg::SHTnsConfig{T}, T_coeffs::AbstractVector{Complex{T}}) where T
    dθ, dφ = cplx_spatial_derivatives(cfg, T_coeffs)
    nlat, nphi = cfg.nlat, cfg.nphi
    uθ = Matrix{Complex{T}}(undef, nlat, nphi)
    uφ = Matrix{Complex{T}}(undef, nlat, nphi)
    @inbounds for i in 1:nlat
        s = sin(cfg.theta_grid[i])
        invs = s > 1e-12 ? (one(T)/s) : zero(T)
        for j in 1:nphi
            uθ[i, j] = invs * dφ[i, j]
            uφ[i, j] = -dθ[i, j]
        end
    end
    return uθ, uφ
end

"""
    cplx_vector_from_potentials(cfg, S_coeffs, T_coeffs)

Construct tangential vector field from spheroidal (S) and toroidal (T) potentials.
Returns (uθ, uφ) as complex spatial matrices.
"""
function cplx_vector_from_potentials(cfg::SHTnsConfig{T}, S_coeffs::AbstractVector{Complex{T}}, T_coeffs::AbstractVector{Complex{T}}) where T
    uθS, uφS = cplx_vector_from_spheroidal(cfg, S_coeffs)
    uθT, uφT = cplx_vector_from_toroidal(cfg, T_coeffs)
    return uθS .+ uθT, uφS .+ uφT
end

"""
    cplx_divergence_spatial_from_potentials(cfg, S_coeffs, T_coeffs)

Compute spatial divergence as complex field from spheroidal/toroidal potentials using spectral identity:
div_lm = -l(l+1) S_lm; synthesize to spatial.
"""
function cplx_divergence_spatial_from_potentials(cfg::SHTnsConfig{T}, S_coeffs::AbstractVector{Complex{T}}, T_coeffs::AbstractVector{Complex{T}}) where T
    div_spec = cplx_divergence_spectral(cfg, S_coeffs, T_coeffs)
    return cplx_sh_to_spat(cfg, div_spec)
end

"""
    cplx_sphtor_to_spat!(cfg, S_coeffs, T_coeffs, u_theta, u_phi)

Synthesize complex tangential vector field from spheroidal (S) and toroidal (T) potentials.
Writes results into u_theta and u_phi (complex spatial matrices).
"""
function cplx_sphtor_to_spat!(cfg::SHTnsConfig{T},
                              S_coeffs::AbstractVector{Complex{T}},
                              T_coeffs::AbstractVector{Complex{T}},
                              u_theta::AbstractMatrix{Complex{T}},
                              u_phi::AbstractMatrix{Complex{T}}) where T
    size(u_theta) == (cfg.nlat, cfg.nphi) || error("u_theta size mismatch")
    size(u_phi) == (cfg.nlat, cfg.nphi) || error("u_phi size mismatch")
    uθ, uφ = cplx_vector_from_potentials(cfg, S_coeffs, T_coeffs)
    u_theta .= uθ
    u_phi  .= uφ
    return u_theta, u_phi
end

"""
    cplx_synthesize_vector(cfg, S_coeffs, T_coeffs)

Allocating version of complex vector synthesis from potentials; returns (u_theta, u_phi).
"""
function cplx_synthesize_vector(cfg::SHTnsConfig{T},
                                S_coeffs::AbstractVector{Complex{T}},
                                T_coeffs::AbstractVector{Complex{T}}) where T
    u_theta = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
    u_phi   = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
    return cplx_sphtor_to_spat!(cfg, S_coeffs, T_coeffs, u_theta, u_phi)
end

"""
    cplx_spat_to_sphtor!(cfg, u_theta, u_phi, S_coeffs, T_coeffs)

Analyze complex tangential vector field into spheroidal and toroidal complex spectral coefficients.
"""
function cplx_spat_to_sphtor!(cfg::SHTnsConfig{T},
                              u_theta::AbstractMatrix{Complex{T}},
                              u_phi::AbstractMatrix{Complex{T}},
                              S_coeffs::AbstractVector{Complex{T}},
                              T_coeffs::AbstractVector{Complex{T}}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    size(u_theta) == (nlat, nphi) || error("u_theta size mismatch")
    size(u_phi) == (nlat, nphi) || error("u_phi size mismatch")
    length(S_coeffs) == _cplx_nlm(cfg) || error("S_coeffs length mismatch")
    length(T_coeffs) == _cplx_nlm(cfg) || error("T_coeffs length mismatch")

    # Fourier transform u_theta and u_phi along longitude
    theta_fourier = Matrix{Complex{T}}(undef, nlat, nphi)
    phi_fourier   = Matrix{Complex{T}}(undef, nlat, nphi)
    for i in 1:nlat
        azimuthal_fft_complex_forward!(cfg, view(u_theta, i, :), view(theta_fourier, i, :))
        azimuthal_fft_complex_forward!(cfg, view(u_phi,   i, :), view(phi_fourier,   i, :))
    end

    fill!(S_coeffs, zero(Complex{T}))
    fill!(T_coeffs, zero(Complex{T}))
    idx_list = _cplx_lm_indices(cfg)

    for (idx, (l, m)) in enumerate(idx_list)
        l >= 1 || continue
        abs(m) <= nphi ÷ 2 || continue
        m_idx = m >= 0 ? m + 1 : nphi + m + 1
        sph_int = zero(Complex{T})
        tor_int = zero(Complex{T})
        for i in 1:nlat
            θ = cfg.theta_grid[i]
            sθ = sin(θ)
            invs = sθ > 1e-12 ? (one(T)/sθ) : zero(T)
            w = cfg.gauss_weights[i]
            k = SHTnsKit.find_plm_index(cfg, l, abs(m))
            Plm = cfg.plm_cache[i, k]
            # d/dθ P_l^{|m|}(cosθ)
            dPlm = _plm_dtheta(cfg, l, m, θ, i)
            uθm = theta_fourier[i, m_idx]
            uφm = phi_fourier[i, m_idx]
            # Projections
            sph_int += (uθm * dPlm + uφm * (Complex{T}(0, m) * Plm * invs)) * w
            tor_int += (uθm * (Complex{T}(0, m) * Plm * invs) - uφm * dPlm) * w
        end
        norm = T(1) / (l * (l + 1))
        S_coeffs[idx] = sph_int * norm
        T_coeffs[idx] = tor_int * norm
    end
    return S_coeffs, T_coeffs
end

"""
    cplx_analyze_vector(cfg, u_theta, u_phi)

Allocating version of complex vector analysis; returns (S_coeffs, T_coeffs).
"""
function cplx_analyze_vector(cfg::SHTnsConfig{T},
                             u_theta::AbstractMatrix{Complex{T}},
                             u_phi::AbstractMatrix{Complex{T}}) where T
    S_coeffs = Vector{Complex{T}}(undef, _cplx_nlm(cfg))
    T_coeffs = Vector{Complex{T}}(undef, _cplx_nlm(cfg))
    return cplx_spat_to_sphtor!(cfg, u_theta, u_phi, S_coeffs, T_coeffs)
end

"""
    cplx_vorticity_spatial_from_potentials(cfg, S_coeffs, T_coeffs)

Compute spatial vertical vorticity (radial curl) as complex field from potentials using spectral identity:
vort_lm = -l(l+1) T_lm; synthesize to spatial.
"""
function cplx_vorticity_spatial_from_potentials(cfg::SHTnsConfig{T}, S_coeffs::AbstractVector{Complex{T}}, T_coeffs::AbstractVector{Complex{T}}) where T
    vort_spec = cplx_vorticity_spectral(cfg, S_coeffs, T_coeffs)
    return cplx_sh_to_spat(cfg, vort_spec)
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

# Cache of per-m mode groups: stored as an array of length (2*mmax+1),
# index = m + mmax + 1, each entry: Vector{(complex_idx, plm_k, l)}
const _cplx_groups_cache = IdDict{SHTnsConfig, Vector{Vector{NTuple{3,Int}}}}()

function _cplx_mode_groups_arr(cfg::SHTnsConfig)
    groups = get!(_cplx_groups_cache, cfg) do
        mmax = cfg.mmax
        arr = [NTuple{3,Int}[] for _ in 1:(2*mmax + 1)]
        idx_list = _cplx_lm_indices(cfg)
        for (idx, (l, m)) in enumerate(idx_list)
            k = SHTnsKit.find_plm_index(cfg, l, abs(m))
            push!(arr[m + mmax + 1], (idx, k, l))
        end
        arr
    end
    return groups
end
