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
    
    fill!(sph_fourier, zero(Complex{T}))
    fill!(tor_fourier, zero(Complex{T}))
    fill!(u_theta, zero(T))
    fill!(u_phi, zero(T))
    
    # For each azimuthal mode m
    for m in 0:min(cfg.mmax, nphi÷2)
        (m == 0 || m % cfg.mres == 0) || continue
        
        # Compute Fourier coefficients for spheroidal component
        sph_mode = Vector{Complex{T}}(undef, nlat)
        tor_mode = Vector{Complex{T}}(undef, nlat)
        fill!(sph_mode, zero(Complex{T}))
        fill!(tor_mode, zero(Complex{T}))
        
        for i in 1:nlat
            theta = cfg.theta_grid[i]
            sint = sin(theta)
            
            sph_sum = zero(Complex{T})
            tor_sum = zero(Complex{T})
            
            for (coeff_idx, (l, m_coeff)) in enumerate(cfg.lm_indices)
                if m_coeff == m && l >= 1  # Vector transforms start from l=1
                    plm_val = cfg.plm_cache[i, coeff_idx]
                    
                    # Compute derivatives of P_l^m
                    dplm_theta = _compute_plm_theta_derivative(cfg, l, m, theta, coeff_idx, i)
                    
                    # Spheroidal contribution
                    if abs(sph_coeffs[coeff_idx]) > 0
                        sph_sum += sph_coeffs[coeff_idx] * plm_val
                    end
                    
                    # Toroidal contribution  
                    if abs(tor_coeffs[coeff_idx]) > 0
                        tor_sum += tor_coeffs[coeff_idx] * plm_val
                    end
                end
            end
            
            sph_mode[i] = sph_sum
            tor_mode[i] = tor_sum
        end
        
        # Insert into Fourier arrays
        insert_fourier_mode!(sph_fourier, m, sph_mode, nlat)
        insert_fourier_mode!(tor_fourier, m, tor_mode, nlat)
    end
    
    # Compute vector components
    _compute_vector_components_from_fourier!(cfg, sph_fourier, tor_fourier, u_theta, u_phi)
    
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
Uses finite differences for now - could be optimized with analytical derivatives.
"""
function _compute_plm_theta_derivative(cfg::SHTnsConfig{T}, l::Int, m::Int, theta::T,
                                      coeff_idx::Int, lat_idx::Int) where T
    # Simple finite difference approximation
    dtheta = T(1e-8)
    
    if lat_idx > 1 && lat_idx < cfg.nlat
        # Central difference
        theta_plus = cfg.theta_grid[lat_idx + 1]
        theta_minus = cfg.theta_grid[lat_idx - 1]
        
        plm_plus = cfg.plm_cache[lat_idx + 1, coeff_idx]
        plm_minus = cfg.plm_cache[lat_idx - 1, coeff_idx]
        
        return (plm_plus - plm_minus) / (theta_plus - theta_minus)
    else
        # Forward/backward difference at boundaries
        if lat_idx == 1
            plm_curr = cfg.plm_cache[lat_idx, coeff_idx]
            plm_next = cfg.plm_cache[lat_idx + 1, coeff_idx]
            theta_curr = cfg.theta_grid[lat_idx]
            theta_next = cfg.theta_grid[lat_idx + 1]
            return (plm_next - plm_curr) / (theta_next - theta_curr)
        else
            plm_curr = cfg.plm_cache[lat_idx, coeff_idx]
            plm_prev = cfg.plm_cache[lat_idx - 1, coeff_idx]
            theta_curr = cfg.theta_grid[lat_idx]
            theta_prev = cfg.theta_grid[lat_idx - 1]
            return (plm_curr - plm_prev) / (theta_curr - theta_prev)
        end
    end
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
                                                 u_theta::AbstractMatrix{T},
                                                 u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Working arrays for spatial fields
    sph_spatial = compute_spatial_from_fourier(sph_fourier, cfg)
    tor_spatial = compute_spatial_from_fourier(tor_fourier, cfg)
    
    # Apply differential operators
    # This is simplified - full implementation needs proper derivatives
    for i in 1:nlat
        theta = cfg.theta_grid[i]
        sint = sin(theta)
        
        for j in 1:nphi
            # Simplified vector component computation
            # Full implementation needs proper gradient calculations
            
            # Theta derivative (finite difference)
            if i > 1 && i < nlat
                dsph_dtheta = (sph_spatial[i+1, j] - sph_spatial[i-1, j]) / 
                             (cfg.theta_grid[i+1] - cfg.theta_grid[i-1])
                dtor_dtheta = (tor_spatial[i+1, j] - tor_spatial[i-1, j]) / 
                             (cfg.theta_grid[i+1] - cfg.theta_grid[i-1])
            else
                dsph_dtheta = zero(T)
                dtor_dtheta = zero(T)
            end
            
            # Phi derivative (spectral)
            phi_idx = j <= nphi÷2 ? j : nphi - j + 1
            dphi = 2π / nphi
            dsph_dphi = zero(T)  # Simplified
            dtor_dphi = zero(T)  # Simplified
            
            # Vector components
            if sint > 1e-12
                u_theta[i, j] = dsph_dtheta + dtor_dphi / sint
                u_phi[i, j] = dsph_dphi / sint - dtor_dtheta
            else
                u_theta[i, j] = zero(T)
                u_phi[i, j] = zero(T)
            end
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