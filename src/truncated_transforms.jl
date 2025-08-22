"""
Truncated transforms at given degree l.
These functions perform transforms with a maximum degree ltr ≤ lmax,
which can save computational time when full resolution is not needed.
"""

"""
    spat_to_sh_l(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, qlm::AbstractVector{Complex{T}}, ltr::Int) where T

Scalar analysis with truncation at degree ltr.

# Arguments
- `cfg`: SHTns configuration
- `vr`: Input spatial field (nlat × nphi)  
- `qlm`: Output SH coefficients (pre-allocated)
- `ltr`: Maximum degree for analysis (ltr ≤ cfg.lmax)

Only coefficients with l ≤ ltr will be computed. This can provide significant
computational savings when the full resolution is not needed.

Equivalent to the C library function `spat_to_SH_l()`.
"""
function spat_to_sh_l(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, qlm::AbstractVector{Complex{T}}, ltr::Int) where T
    validate_config(cfg)
    size(vr) == (cfg.nlat, cfg.nphi) || error("vr size must be (nlat, nphi)")  
    length(qlm) == cfg.nlm || error("qlm length must equal nlm")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    # Zero out all coefficients first
    fill!(qlm, zero(Complex{T}))
    
    # Only compute coefficients for l ≤ ltr
    lock(cfg.lock) do
        _spat_to_sh_l_impl!(cfg, vr, qlm, ltr)
    end
    
    return qlm
end

"""
    sh_to_spat_l(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}, ltr::Int) where T

Scalar synthesis with truncation at degree ltr.

# Arguments
- `cfg`: SHTns configuration  
- `qlm`: Input SH coefficients
- `vr`: Output spatial field (pre-allocated, nlat × nphi)
- `ltr`: Maximum degree for synthesis (ltr ≤ cfg.lmax)

Only coefficients with l ≤ ltr will be used in the synthesis.

Equivalent to the C library function `SH_to_spat_l()`.
"""
function sh_to_spat_l(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}, ltr::Int) where T
    validate_config(cfg)
    length(qlm) == cfg.nlm || error("qlm length must equal nlm")
    size(vr) == (cfg.nlat, cfg.nphi) || error("vr size must be (nlat, nphi)")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    # Zero out spatial field
    fill!(vr, zero(T))
    
    lock(cfg.lock) do  
        _sh_to_spat_l_impl!(cfg, qlm, vr, ltr)
    end
    
    return vr
end

"""
    sphtor_to_spat_l(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                     vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T

Vector synthesis with truncation at degree ltr.

Equivalent to the C library function `SHsphtor_to_spat_l()`.
"""
function sphtor_to_spat_l(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                          vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T
    validate_config(cfg)
    length(slm) == cfg.nlm || error("slm length must equal nlm")
    length(tlm) == cfg.nlm || error("tlm length must equal nlm")
    size(vt) == (cfg.nlat, cfg.nphi) || error("vt size must be (nlat, nphi)")
    size(vp) == (cfg.nlat, cfg.nphi) || error("vp size must be (nlat, nphi)")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    fill!(vt, zero(T))
    fill!(vp, zero(T))
    
    lock(cfg.lock) do
        _sphtor_to_spat_l_impl!(cfg, slm, tlm, vt, vp, ltr)
    end
    
    return (vt, vp)
end

"""
    spat_to_sphtor_l(cfg::SHTnsConfig{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                     slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T

Vector analysis with truncation at degree ltr.

Equivalent to the C library function `spat_to_SHsphtor_l()`.
"""
function spat_to_sphtor_l(cfg::SHTnsConfig{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                          slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T
    validate_config(cfg)
    size(vt) == (cfg.nlat, cfg.nphi) || error("vt size must be (nlat, nphi)")
    size(vp) == (cfg.nlat, cfg.nphi) || error("vp size must be (nlat, nphi)")
    length(slm) == cfg.nlm || error("slm length must equal nlm")
    length(tlm) == cfg.nlm || error("tlm length must equal nlm")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    fill!(slm, zero(Complex{T}))
    fill!(tlm, zero(Complex{T}))
    
    lock(cfg.lock) do
        _spat_to_sphtor_l_impl!(cfg, vt, vp, slm, tlm, ltr)
    end
    
    return (slm, tlm)
end

"""
    sh_to_grad_spat_l(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}}, 
                      gt::AbstractMatrix{T}, gp::AbstractMatrix{T}, ltr::Int) where T

Compute spatial gradient with truncation at degree ltr.
This is an alias for `sph_to_spat_l()` applied to gradient calculation.

Equivalent to the C library function `SH_to_grad_spat_l()`.
"""
function sh_to_grad_spat_l(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}},
                          gt::AbstractMatrix{T}, gp::AbstractMatrix{T}, ltr::Int) where T
    # The gradient operation is equivalent to the spheroidal vector synthesis
    return sphtor_to_spat_l(cfg, slm, zero(slm), gt, gp, ltr)
end

"""
    spat_to_shqst_l(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                    qlm::AbstractVector{Complex{T}}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T

3D vector analysis with truncation at degree ltr.

Equivalent to the C library function `spat_to_SHqst_l()`.
"""
function spat_to_shqst_l(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                         qlm::AbstractVector{Complex{T}}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T
    validate_config(cfg)
    size(vr) == (cfg.nlat, cfg.nphi) || error("vr size must be (nlat, nphi)")
    size(vt) == (cfg.nlat, cfg.nphi) || error("vt size must be (nlat, nphi)") 
    size(vp) == (cfg.nlat, cfg.nphi) || error("vp size must be (nlat, nphi)")
    length(qlm) == cfg.nlm || error("qlm length must equal nlm")
    length(slm) == cfg.nlm || error("slm length must equal nlm")
    length(tlm) == cfg.nlm || error("tlm length must equal nlm")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    # Clear output arrays
    fill!(qlm, zero(Complex{T}))
    fill!(slm, zero(Complex{T}))
    fill!(tlm, zero(Complex{T}))
    
    lock(cfg.lock) do
        # Analyze radial component
        _spat_to_sh_l_impl!(cfg, vr, qlm, ltr)
        
        # Analyze horizontal vector components  
        _spat_to_sphtor_l_impl!(cfg, vt, vp, slm, tlm, ltr)
    end
    
    return (qlm, slm, tlm)
end

"""
    shqst_to_spat_l(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                    vr::AbstractMatrix{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T

3D vector synthesis with truncation at degree ltr.

Equivalent to the C library function `SHqst_to_spat_l()`.
"""
function shqst_to_spat_l(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                         vr::AbstractMatrix{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T
    validate_config(cfg)
    length(qlm) == cfg.nlm || error("qlm length must equal nlm")
    length(slm) == cfg.nlm || error("slm length must equal nlm")
    length(tlm) == cfg.nlm || error("tlm length must equal nlm")
    size(vr) == (cfg.nlat, cfg.nphi) || error("vr size must be (nlat, nphi)")
    size(vt) == (cfg.nlat, cfg.nphi) || error("vt size must be (nlat, nphi)")
    size(vp) == (cfg.nlat, cfg.nphi) || error("vp size must be (nlat, nphi)")
    0 <= ltr <= cfg.lmax || error("ltr must be in [0, lmax]")
    
    # Clear output arrays
    fill!(vr, zero(T))
    fill!(vt, zero(T))
    fill!(vp, zero(T))
    
    lock(cfg.lock) do
        # Synthesize radial component
        _sh_to_spat_l_impl!(cfg, qlm, vr, ltr)
        
        # Synthesize horizontal vector components
        _sphtor_to_spat_l_impl!(cfg, slm, tlm, vt, vp, ltr)
    end
    
    return (vr, vt, vp)
end

# Implementation functions

"""
    _spat_to_sh_l_impl!(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, qlm::AbstractVector{Complex{T}}, ltr::Int) where T

Internal implementation of truncated scalar analysis.
"""
function _spat_to_sh_l_impl!(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, qlm::AbstractVector{Complex{T}}, ltr::Int) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Allocate work arrays
    nphi_modes = nphi ÷ 2 + 1
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    
    # Forward FFT in φ direction for each θ
    fft_plan = get!(cfg.fft_plans, :forward) do
        plan_rfft(zeros(T, nphi))
    end
    
    for j in 1:nlat
        fourier_coeffs[j, :] = fft_plan * view(vr, j, :)
    end
    
    # Legendre analysis for each m, but only up to ltr
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l <= ltr  # Only compute coefficients for l ≤ ltr
            m_abs = abs(m)
            if m_abs < nphi_modes
                m_idx = m_abs + 1
                
                # Get Fourier coefficients for this m
                fourier_mode = view(fourier_coeffs, :, m_idx)
                
                # Legendre transform
                coeff = zero(Complex{T})
                for j in 1:nlat
                    # Get Legendre polynomial value from cache or compute
                    plm_val = _get_cached_plm_value(cfg, j, l, m_abs)
                    weight = _get_integration_weight(cfg, j)
                    
                    if m == 0
                        coeff += real(fourier_mode[j]) * plm_val * weight
                    elseif m > 0  
                        coeff += fourier_mode[j] * plm_val * weight
                    else  # m < 0, use conjugate symmetry
                        coeff += conj(fourier_mode[j]) * plm_val * weight * (-1)^m_abs
                    end
                end
                
                # Apply normalization
                qlm[idx] = coeff * _get_analysis_normalization_factor(cfg, l, m)
            end
        end
        # Coefficients for l > ltr remain zero (already filled with zeros)
    end
end

"""
    _sh_to_spat_l_impl!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}, ltr::Int) where T

Internal implementation of truncated scalar synthesis.
"""
function _sh_to_spat_l_impl!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}, ltr::Int) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Allocate work arrays
    nphi_modes = nphi ÷ 2 + 1
    fourier_coeffs = zeros(Complex{T}, nlat, nphi_modes)
    
    # Legendre synthesis for each m, but only for l ≤ ltr
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l <= ltr  # Only use coefficients for l ≤ ltr
            m_abs = abs(m)
            if m_abs < nphi_modes
                m_idx = m_abs + 1
                coeff = qlm[idx]
                
                for j in 1:nlat
                    plm_val = _get_cached_plm_value(cfg, j, l, m_abs)
                    
                    if m == 0
                        fourier_coeffs[j, m_idx] += real(coeff) * plm_val
                    elseif m > 0
                        fourier_coeffs[j, m_idx] += coeff * plm_val
                    else  # m < 0, handled by conjugate symmetry in RFFT
                        # Contribution will be handled automatically by RFFT
                    end
                end
            end
        end
    end
    
    # Inverse FFT in φ direction  
    ifft_plan = get!(cfg.fft_plans, :backward) do
        plan_irfft(zeros(Complex{T}, nphi_modes), nphi)
    end
    
    for j in 1:nlat
        vr[j, :] = ifft_plan * view(fourier_coeffs, j, :)
    end
end

"""
    _sphtor_to_spat_l_impl!(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                           vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T

Internal implementation of truncated vector synthesis.
"""
function _sphtor_to_spat_l_impl!(cfg::SHTnsConfig{T}, slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}},
                                vt::AbstractMatrix{T}, vp::AbstractMatrix{T}, ltr::Int) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Work arrays for Fourier coefficients
    sph_fourier_t = zeros(Complex{T}, nlat, nphi_modes)  # θ component from spheroidal
    sph_fourier_p = zeros(Complex{T}, nlat, nphi_modes)  # φ component from spheroidal  
    tor_fourier_t = zeros(Complex{T}, nlat, nphi_modes)  # θ component from toroidal
    tor_fourier_p = zeros(Complex{T}, nlat, nphi_modes)  # φ component from toroidal
    
    # Process each coefficient, but only for l ≤ ltr
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l <= ltr && l > 0  # Vector fields need l > 0
            m_abs = abs(m)
            if m_abs < nphi_modes
                m_idx = m_abs + 1
                s_coeff = slm[idx]
                t_coeff = tlm[idx]
                
                for j in 1:nlat
                    plm_val = _get_cached_plm_value(cfg, j, l, m_abs)
                    dplm_val = _get_cached_dplm_value(cfg, j, l, m_abs)
                    sint = sin(cfg.theta_grid[j])
                    
                    # Spheroidal contributions
                    # vθ += ∂S/∂θ, vφ += (1/sin θ) * im * S  
                    if m == 0
                        sph_fourier_t[j, m_idx] += real(s_coeff) * dplm_val
                        # No φ component for m=0
                    elseif m > 0
                        sph_fourier_t[j, m_idx] += s_coeff * dplm_val
                        if sint > 1e-12
                            sph_fourier_p[j, m_idx] += s_coeff * (Complex{T}(0, m) / sint) * plm_val
                        end
                    end
                    
                    # Toroidal contributions  
                    # vθ += -(1/sin θ) * im * T, vφ += ∂T/∂θ
                    if m == 0
                        tor_fourier_p[j, m_idx] += real(t_coeff) * dplm_val
                        # No θ component for m=0
                    elseif m > 0
                        tor_fourier_p[j, m_idx] += t_coeff * dplm_val
                        if sint > 1e-12
                            tor_fourier_t[j, m_idx] -= t_coeff * (Complex{T}(0, m) / sint) * plm_val
                        end
                    end
                end
            end
        end
    end
    
    # Inverse FFTs
    ifft_plan = get!(cfg.fft_plans, :backward) do
        plan_irfft(zeros(Complex{T}, nphi_modes), nphi)
    end
    
    work_array = zeros(T, nphi)
    
    for j in 1:nlat
        # θ component
        combined_t = sph_fourier_t[j, :] + tor_fourier_t[j, :]
        vt[j, :] = ifft_plan * combined_t
        
        # φ component  
        combined_p = sph_fourier_p[j, :] + tor_fourier_p[j, :]
        vp[j, :] = ifft_plan * combined_p
    end
end

"""
    _spat_to_sphtor_l_impl!(cfg::SHTnsConfig{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                           slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T

Internal implementation of truncated vector analysis.
"""
function _spat_to_sphtor_l_impl!(cfg::SHTnsConfig{T}, vt::AbstractMatrix{T}, vp::AbstractMatrix{T},
                                slm::AbstractVector{Complex{T}}, tlm::AbstractVector{Complex{T}}, ltr::Int) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Forward FFTs
    fft_plan = get!(cfg.fft_plans, :forward) do
        plan_rfft(zeros(T, nphi))
    end
    
    fourier_vt = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    fourier_vp = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    
    for j in 1:nlat
        fourier_vt[j, :] = fft_plan * view(vt, j, :)
        fourier_vp[j, :] = fft_plan * view(vp, j, :)
    end
    
    # Legendre analysis for each coefficient, but only for l ≤ ltr
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l <= ltr && l > 0
            m_abs = abs(m)
            if m_abs < nphi_modes
                m_idx = m_abs + 1
                
                sph_integral = zero(Complex{T})
                tor_integral = zero(Complex{T})
                
                for j in 1:nlat
                    plm_val = _get_cached_plm_value(cfg, j, l, m_abs)
                    dplm_val = _get_cached_dplm_value(cfg, j, l, m_abs)  
                    weight = _get_integration_weight(cfg, j)
                    sint = sin(cfg.theta_grid[j])
                    
                    vt_mode = fourier_vt[j, m_idx]
                    vp_mode = fourier_vp[j, m_idx]
                    
                    # Spheroidal integral: ∫(vθ * ∂P/∂θ + vφ * im*P/sin θ) dΩ
                    sph_contrib = vt_mode * dplm_val * weight
                    if sint > 1e-12
                        sph_contrib += vp_mode * (Complex{T}(0, m) / sint) * plm_val * weight
                    end
                    sph_integral += sph_contrib
                    
                    # Toroidal integral: ∫(-vθ * im*P/sin θ + vφ * ∂P/∂θ) dΩ
                    tor_contrib = vp_mode * dplm_val * weight
                    if sint > 1e-12  
                        tor_contrib -= vt_mode * (Complex{T}(0, m) / sint) * plm_val * weight
                    end
                    tor_integral += tor_contrib
                end
                
                # Apply normalization and l(l+1) factor
                norm_factor = _get_vector_analysis_normalization_factor(cfg, l, m)
                if l > 0
                    slm[idx] = sph_integral * norm_factor / T(l * (l + 1))
                    tlm[idx] = tor_integral * norm_factor / T(l * (l + 1))  
                end
            end
        end
    end
end

# Helper functions (simplified versions - should be more sophisticated in practice)

function _get_cached_plm_value(cfg::SHTnsConfig{T}, j::Int, l::Int, m::Int) where T
    # This should access the cached Legendre polynomial values
    # For now, return a placeholder - in practice this would be optimized
    cost = cos(cfg.theta_grid[j])
    sint = sin(cfg.theta_grid[j])
    return _compute_single_legendre_basic(l, m, cost, sint)  # From single_m_transforms.jl
end

function _get_cached_dplm_value(cfg::SHTnsConfig{T}, j::Int, l::Int, m::Int) where T
    # This should access cached derivatives
    # Placeholder implementation
    cost = cos(cfg.theta_grid[j])
    sint = sin(cfg.theta_grid[j])
    return _compute_legendre_derivative_basic(l, m, cost, sint)
end

function _get_integration_weight(cfg::SHTnsConfig{T}, j::Int) where T
    if cfg.grid_type == SHT_GAUSS
        return cfg.gauss_weights[j]
    else
        # Regular grid weight
        return π / (cfg.nlat - 1)  # Simplified
    end
end

function _get_analysis_normalization_factor(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    """Truncated transform analysis normalization factor."""
    if cfg.norm == SHT_ORTHONORMAL
        # Orthonormal: standard integration factor
        factor = T(4π)
        if m > 0
            factor *= T(2)  # Real coefficient handling
        end
        return factor
    elseif cfg.norm == SHT_FOURPI
        return T(4π)
    elseif cfg.norm == SHT_SCHMIDT
        # Schmidt semi-normalization
        factor = T(4π)
        if m > 0
            factor /= sqrt(T(2))
        end
        return factor
    else # SHT_REAL_NORM
        return T(4π)
    end
end

function _get_vector_analysis_normalization_factor(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    """Vector field analysis normalization for truncated transforms."""
    # Include l(l+1) scaling for vector fields
    l_scaling = (l == 0) ? T(1) : T(l * (l + 1))
    
    base_factor = _get_analysis_normalization_factor(cfg, l, m)
    
    # Apply vector field correction
    return base_factor / sqrt(l_scaling)
end

function _compute_legendre_derivative_basic(l::Int, m::Int, cost::T, sint::T) where T
    # Simplified derivative computation - should use optimized version
    if l == 0
        return zero(T)
    elseif l == 1 && m == 0
        return -sint
    else
        # Use recurrence relation - simplified version
        return zero(T)  # Placeholder
    end
end