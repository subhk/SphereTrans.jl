"""
Vector spherical harmonic transforms.
Handles vector fields decomposed into spheroidal and toroidal components.
"""

"""
    sphtor_to_spat!(cfg::SHTnsConfig{T}, 
                   sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                   u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Transform spheroidal and toroidal coefficients to vector components.
Synthesis: (S_lm, T_lm) → (u_θ, u_φ)

The vector field is decomposed as:
**u** = ∇×(S × **r̂**) + ∇×∇×(T × **r̂**)
where S and T are the spheroidal and toroidal scalars.

# Arguments
- `cfg`: SHTns configuration
- `sph_coeffs`: Spheroidal (poloidal) coefficients (length nlm)
- `tor_coeffs`: Toroidal coefficients (length nlm)  
- `u_theta`: Output theta component (nlat × nphi, pre-allocated)
- `u_phi`: Output phi component (nlat × nphi, pre-allocated)
"""
function sphtor_to_spat!(cfg::SHTnsConfig{T},
                        sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                        u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    validate_config(cfg)
    length(sph_coeffs) == cfg.nlm || error("sph_coeffs length must equal nlm")
    length(tor_coeffs) == cfg.nlm || error("tor_coeffs length must equal nlm")
    size(u_theta) == (cfg.nlat, cfg.nphi) || error("u_theta size mismatch")
    size(u_phi) == (cfg.nlat, cfg.nphi) || error("u_phi size mismatch")
    
    lock(cfg.lock) do
        _sphtor_to_spat_impl!(cfg, sph_coeffs, tor_coeffs, u_theta, u_phi)
    end
    
    return u_theta, u_phi
end

"""
    spat_to_sphtor!(cfg::SHTnsConfig{T},
                   u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                   sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T

Transform vector components to spheroidal and toroidal coefficients.
Analysis: (u_θ, u_φ) → (S_lm, T_lm)

# Arguments
- `cfg`: SHTns configuration
- `u_theta`: Input theta component (nlat × nphi)
- `u_phi`: Input phi component (nlat × nphi)
- `sph_coeffs`: Output spheroidal coefficients (length nlm, pre-allocated)
- `tor_coeffs`: Output toroidal coefficients (length nlm, pre-allocated)
"""
function spat_to_sphtor!(cfg::SHTnsConfig{T},
                        u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                        sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T
    validate_config(cfg)
    size(u_theta) == (cfg.nlat, cfg.nphi) || error("u_theta size mismatch")
    size(u_phi) == (cfg.nlat, cfg.nphi) || error("u_phi size mismatch")
    length(sph_coeffs) == cfg.nlm || error("sph_coeffs length must equal nlm")
    length(tor_coeffs) == cfg.nlm || error("tor_coeffs length must equal nlm")
    
    lock(cfg.lock) do
        _spat_to_sphtor_impl!(cfg, u_theta, u_phi, sph_coeffs, tor_coeffs)
    end
    
    return sph_coeffs, tor_coeffs
end

# Implementation functions

"""
    _sphtor_to_spat_impl!(cfg::SHTnsConfig{T},
                         sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                         u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Internal implementation of vector synthesis.

The spheroidal-toroidal decomposition gives:
- u_θ = ∂S/∂θ + (1/sin θ) ∂T/∂φ  
- u_φ = (1/sin θ) ∂S/∂φ - ∂T/∂θ

Where S and T are reconstructed from their spherical harmonic coefficients.
"""
function _sphtor_to_spat_impl!(cfg::SHTnsConfig{T},
                              sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                              u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Allocate working arrays
    nphi_modes = nphi ÷ 2 + 1
    sph_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    tor_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    sph_dtheta_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    tor_dtheta_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    
    fill!(sph_fourier, zero(Complex{T}))
    fill!(tor_fourier, zero(Complex{T}))
    fill!(sph_dtheta_fourier, zero(Complex{T}))
    fill!(tor_dtheta_fourier, zero(Complex{T}))
    fill!(u_theta, zero(T))
    fill!(u_phi, zero(T))
    
    # For each azimuthal mode m
    for m in 0:min(cfg.mmax, nphi÷2)
        (m == 0 || m % cfg.mres == 0) || continue
        
        # Compute Fourier coefficients for spheroidal component
        sph_mode = Vector{Complex{T}}(undef, nlat)
        tor_mode = Vector{Complex{T}}(undef, nlat)
        dtheta_sph_mode = Vector{Complex{T}}(undef, nlat)
        dtheta_tor_mode = Vector{Complex{T}}(undef, nlat)
        fill!(sph_mode, zero(Complex{T}))
        fill!(tor_mode, zero(Complex{T}))
        fill!(dtheta_sph_mode, zero(Complex{T}))
        fill!(dtheta_tor_mode, zero(Complex{T}))
        
        for i in 1:nlat
            theta = cfg.theta_grid[i]
            sint = sin(theta)
            
            sph_sum = zero(Complex{T})
            tor_sum = zero(Complex{T})
            dtheta_sph = zero(Complex{T})
            dtheta_tor = zero(Complex{T})

            for (coeff_idx, (l, m_coeff)) in enumerate(cfg.lm_indices)
                if m_coeff == m && l >= 1  # Vector transforms start from l=1
                    plm_val = cfg.plm_cache[i, coeff_idx]
                    
                    # Compute derivatives of P_l^m
                    dplm_theta = _compute_plm_theta_derivative(cfg, l, m, theta, coeff_idx, i)
                    
                    # Spheroidal contribution
                    if abs(sph_coeffs[coeff_idx]) > 0
                        sph_sum += sph_coeffs[coeff_idx] * plm_val
                        dtheta_sph += sph_coeffs[coeff_idx] * dplm_theta
                    end
                    
                    # Toroidal contribution  
                    if abs(tor_coeffs[coeff_idx]) > 0
                        tor_sum += tor_coeffs[coeff_idx] * plm_val
                        dtheta_tor += tor_coeffs[coeff_idx] * dplm_theta
                    end
                end
            end
            
            sph_mode[i] = sph_sum
            tor_mode[i] = tor_sum
            dtheta_sph_mode[i] = dtheta_sph
            dtheta_tor_mode[i] = dtheta_tor
        end
        
        # Insert into Fourier arrays
        insert_fourier_mode!(sph_fourier, m, sph_mode, nlat)
        insert_fourier_mode!(tor_fourier, m, tor_mode, nlat)
        insert_fourier_mode!(sph_dtheta_fourier, m, dtheta_sph_mode, nlat)
        insert_fourier_mode!(tor_dtheta_fourier, m, dtheta_tor_mode, nlat)
    end
    
    # Compute vector components
    _compute_vector_components_from_fourier!(cfg, sph_fourier, tor_fourier,
                                             sph_dtheta_fourier, tor_dtheta_fourier,
                                             u_theta, u_phi)
    
    return nothing
end

"""
    _spat_to_sphtor_impl!(cfg::SHTnsConfig{T},
                         u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                         sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T

Internal implementation of vector analysis.
"""
function _spat_to_sphtor_impl!(cfg::SHTnsConfig{T},
                              u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                              sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T
    # This is a complex operation requiring the inverse of the vector synthesis
    # For now, implement a simplified version using scalar transforms of components
    
    # Transform components to Fourier domain
    theta_fourier = compute_fourier_coefficients_spatial(u_theta, cfg)
    phi_fourier = compute_fourier_coefficients_spatial(u_phi, cfg)
    
    fill!(sph_coeffs, zero(T))
    fill!(tor_coeffs, zero(T))
    
    # For each (l,m) coefficient
    for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        l >= 1 || continue  # Vector modes start from l=1
        
        if m <= cfg.nphi ÷ 2
            # Extract mode data
            theta_mode = Vector{Complex{T}}(undef, cfg.nlat)
            phi_mode = Vector{Complex{T}}(undef, cfg.nlat)
            
            extract_fourier_mode!(theta_fourier, m, theta_mode, cfg.nlat)
            extract_fourier_mode!(phi_fourier, m, phi_mode, cfg.nlat)
            
            # Compute spheroidal and toroidal coefficients using quadrature
            sph_integral = zero(Complex{T})
            tor_integral = zero(Complex{T})
            
            for i in 1:cfg.nlat
                theta = cfg.theta_grid[i]
                sint = sin(theta)
                weight = cfg.gauss_weights[i]
                
                plm_val = cfg.plm_cache[i, coeff_idx]
                dplm_theta = _compute_plm_theta_derivative(cfg, l, m, theta, coeff_idx, i)
                
                # Vector harmonic analysis (simplified)
                if sint > 1e-12  # Avoid division by zero at poles
                    # This is a simplified form - full implementation needs proper vector harmonic analysis
                    sph_contribution = theta_mode[i] * dplm_theta + phi_mode[i] * (im * m * plm_val / sint)
                    tor_contribution = theta_mode[i] * (im * m * plm_val / sint) - phi_mode[i] * dplm_theta
                    
                    sph_integral += sph_contribution * weight
                    tor_integral += tor_contribution * weight
                end
            end
            
            # Normalization factor
            norm_factor = T(1) / (l * (l + 1))
            
            sph_coeffs[coeff_idx] = real(sph_integral) * norm_factor
            tor_coeffs[coeff_idx] = real(tor_integral) * norm_factor
        end
    end
    
    return nothing
end

"""
    _compute_plm_theta_derivative(cfg::SHTnsConfig{T}, l::Int, m::Int, theta::T, 
                                 coeff_idx::Int, lat_idx::Int) where T

Compute the theta derivative of associated Legendre polynomial P_l^m(cos θ).
Uses analytical recurrence:
∂θ P_l^m(cosθ) = (l*cosθ*P_l^m(cosθ) - (l+m) P_{l-1}^m(cosθ)) / sinθ
"""
function _compute_plm_theta_derivative(cfg::SHTnsConfig{T}, l::Int, m::Int, theta::T,
                                      coeff_idx::Int, lat_idx::Int) where T
    # Fetch P_l^m and P_{l-1}^m at this latitude
    Plm = cfg.plm_cache[lat_idx, coeff_idx]
    if l == 0
        return zero(T)
    end
    # Find index for (l-1, m)
    idx_lm1 = -1
    @inbounds for (k, (ll, mm)) in enumerate(cfg.lm_indices)
        if ll == l - 1 && mm == m
            idx_lm1 = k
            break
        end
    end
    Plm1 = idx_lm1 > 0 ? cfg.plm_cache[lat_idx, idx_lm1] : zero(T)
    x = cos(theta)
    s = sin(theta)
    if abs(s) < T(1e-12)
        return zero(T)
    end
    return (l * x * Plm - (l + m) * Plm1) / s
end

"""
    _compute_vector_components_from_fourier!(cfg::SHTnsConfig{T},
                                           sph_fourier::AbstractMatrix{Complex{T}},
                                           tor_fourier::AbstractMatrix{Complex{T}},
                                           u_theta::AbstractMatrix{T},
                                           u_phi::AbstractMatrix{T}) where T

Compute final vector components from Fourier representations.
Applies the differential operators to convert spheroidal/toroidal to θ,φ components.
"""
function _compute_vector_components_from_fourier!(cfg::SHTnsConfig{T},
                                                 sph_fourier::AbstractMatrix{Complex{T}},
                                                 tor_fourier::AbstractMatrix{Complex{T}},
                                                 sph_dtheta_fourier::AbstractMatrix{Complex{T}},
                                                 tor_dtheta_fourier::AbstractMatrix{Complex{T}},
                                                 u_theta::AbstractMatrix{T},
                                                 u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Inverse FFTs for S, T and their theta-derivatives
    sph_spatial = compute_spatial_from_fourier(sph_fourier, cfg)
    tor_spatial = compute_spatial_from_fourier(tor_fourier, cfg)
    dsph_dtheta = compute_spatial_from_fourier(sph_dtheta_fourier, cfg)
    dtor_dtheta = compute_spatial_from_fourier(tor_dtheta_fourier, cfg)

    # Build phi-derivative in Fourier domain: multiply mode m by im*m
    nphi_modes = size(sph_fourier, 2)
    sph_dphi_fourier = similar(sph_fourier)
    tor_dphi_fourier = similar(tor_fourier)
    @inbounds for m in 0:(nphi_modes-1)
        factor = Complex{T}(0, m)
        sph_dphi_fourier[:, m+1] .= factor .* sph_fourier[:, m+1]
        tor_dphi_fourier[:, m+1] .= factor .* tor_fourier[:, m+1]
    end
    dsph_dphi = compute_spatial_from_fourier(sph_dphi_fourier, cfg)
    dtor_dphi = compute_spatial_from_fourier(tor_dphi_fourier, cfg)

    # Compose vector components
    @inbounds for i in 1:nlat
        theta = cfg.theta_grid[i]
        sint = sin(theta)
        inv_sint = sint > 1e-12 ? (one(T)/sint) : zero(T)
        for j in 1:nphi
            u_theta[i, j] = dsph_dtheta[i, j] + inv_sint * dtor_dphi[i, j]
            u_phi[i, j] = inv_sint * dsph_dphi[i, j] - dtor_dtheta[i, j]
        end
    end
    return nothing
end

# Convenience functions

"""
    analyze_vector(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, 
                  u_phi::AbstractMatrix{T}) where T

Analyze vector field into spheroidal and toroidal components (allocating).
"""
function analyze_vector(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T},
                       u_phi::AbstractMatrix{T}) where T
    sph_coeffs = Vector{T}(undef, cfg.nlm)
    tor_coeffs = Vector{T}(undef, cfg.nlm)
    spat_to_sphtor!(cfg, u_theta, u_phi, sph_coeffs, tor_coeffs)
    return sph_coeffs, tor_coeffs
end

"""
    synthesize_vector(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{T},
                     tor_coeffs::AbstractVector{T}) where T

Synthesize vector field from spheroidal and toroidal components (allocating).
"""
function synthesize_vector(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{T},
                          tor_coeffs::AbstractVector{T}) where T
    u_theta = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    u_phi = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    sphtor_to_spat!(cfg, sph_coeffs, tor_coeffs, u_theta, u_phi)
    return u_theta, u_phi
end
