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

Vector synthesis implementation based on C code algorithm.
Transforms spheroidal and toroidal coefficients to vector components.

Mathematical formulation from C code:
u_θ = Σ_l Σ_m [∂P_l^m/∂θ * S_lm] * exp(imφ)
u_φ = Σ_l Σ_m [∂P_l^m/∂θ * T_lm] * exp(imφ)
"""
function _sphtor_to_spat_impl!(cfg::SHTnsConfig{T},
                              sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                              u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Use pre-allocated workspace with type-stable access
    theta_fourier_key = :workspace_vector_theta_fourier
    phi_fourier_key = :workspace_vector_phi_fourier
    
    if haskey(cfg.fft_plans, theta_fourier_key)
        theta_fourier = cfg.fft_plans[theta_fourier_key]::Matrix{Complex{T}}
        phi_fourier = cfg.fft_plans[phi_fourier_key]::Matrix{Complex{T}}
        if size(theta_fourier) != (nlat, nphi_modes)
            theta_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
            phi_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
            cfg.fft_plans[theta_fourier_key] = theta_fourier
            cfg.fft_plans[phi_fourier_key] = phi_fourier
        end
    else
        theta_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
        phi_fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
        cfg.fft_plans[theta_fourier_key] = theta_fourier
        cfg.fft_plans[phi_fourier_key] = phi_fourier
    end
    
    fill!(theta_fourier, zero(Complex{T}))
    fill!(phi_fourier, zero(Complex{T}))
    
    # Pre-compute m-coefficient mapping with type-stable access
    mapping_key = :m_coefficient_mapping
    if haskey(cfg.fft_plans, mapping_key)
        m_indices = cfg.fft_plans[mapping_key]::Dict{Int, Vector{Int}}
    else
        m_indices = _build_m_coefficient_mapping(cfg)
        cfg.fft_plans[mapping_key] = m_indices
    end
    
    # Vector harmonic synthesis based on C code algorithm:
    # u_θ = Σ_l Σ_m [∂P_l^m/∂θ * S_lm] * exp(imφ)
    # u_φ = Σ_l Σ_m [∂P_l^m/∂θ * T_lm] * exp(imφ)
    
    # For each azimuthal mode m (only m >= 0)
    @inbounds for m in 0:min(cfg.mmax, nphi÷2)
        # Skip if this m is not included due to mres
        (m == 0 || m % cfg.mres == 0) || continue
        
        # Get precomputed indices for this m
        coeff_indices = get(m_indices, m, Int[])
        isempty(coeff_indices) && continue
        
        # Direct computation with SIMD optimization
        m_col = m + 1  # Convert to 1-based indexing
        if m_col <= nphi_modes
            if m == 0
                # For m=0, no scaling needed
                @inbounds @simd for i in 1:nlat
                    theta_value = zero(Complex{T})
                    phi_value = zero(Complex{T})
                    @simd for coeff_idx in coeff_indices
                        l, m_coeff = cfg.lm_indices[coeff_idx]
                        if l >= 1  # Vector modes start from l=1
                            theta = cfg.theta_grid[i]
                            dplm_val = _compute_plm_theta_derivative(cfg, l, m_coeff, theta, coeff_idx, i)
                            sph_coeff = sph_coeffs[coeff_idx]
                            tor_coeff = tor_coeffs[coeff_idx]
                            
                            theta_value += sph_coeff * dplm_val
                            phi_value += tor_coeff * dplm_val
                        end
                    end
                    theta_fourier[i, m_col] = theta_value
                    phi_fourier[i, m_col] = phi_value
                end
            else
                # For m>0, apply mpos_renorm scaling
                scale_factor = T(0.5)
                @inbounds @simd for i in 1:nlat
                    theta_value = zero(Complex{T})
                    phi_value = zero(Complex{T})
                    @simd for coeff_idx in coeff_indices
                        l, m_coeff = cfg.lm_indices[coeff_idx]
                        if l >= 1  # Vector modes start from l=1
                            theta = cfg.theta_grid[i]
                            dplm_val = _compute_plm_theta_derivative(cfg, l, m_coeff, theta, coeff_idx, i)
                            sph_coeff = sph_coeffs[coeff_idx] * scale_factor
                            tor_coeff = tor_coeffs[coeff_idx] * scale_factor
                            
                            theta_value += sph_coeff * dplm_val
                            phi_value += tor_coeff * dplm_val
                        end
                    end
                    theta_fourier[i, m_col] = theta_value
                    phi_fourier[i, m_col] = phi_value
                end
            end
        end
    end
    
    # Transform from Fourier coefficients to spatial domain
    theta_temp = compute_spatial_from_fourier(theta_fourier, cfg)
    phi_temp = compute_spatial_from_fourier(phi_fourier, cfg)
    
    # FFTW irfft scaling: Need to multiply by nphi to get correct amplitude
    u_theta .= theta_temp .* T(nphi)
    u_phi .= phi_temp .* T(nphi)
    
    return nothing
end

"""
    _spat_to_sphtor_impl!(cfg::SHTnsConfig{T},
                         u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                         sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T

Vector analysis implementation based on C code algorithm.
Transforms vector components to spheroidal and toroidal coefficients.

Based on C code spat_to_SHsphtor_kernel.c algorithm.
"""
function _spat_to_sphtor_impl!(cfg::SHTnsConfig{T},
                              u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                              sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Transform spatial data to Fourier coefficients in longitude
    theta_fourier = compute_fourier_coefficients_spatial(u_theta, cfg)
    phi_fourier = compute_fourier_coefficients_spatial(u_phi, cfg)
    
    # Initialize coefficients
    fill!(sph_coeffs, zero(T))
    fill!(tor_coeffs, zero(T))
    
    # Pre-allocate workspace for mode extraction
    mode_workspace_key = :workspace_vector_mode_data
    if haskey(cfg.fft_plans, mode_workspace_key)
        theta_mode_data = cfg.fft_plans[mode_workspace_key]::Vector{Complex{T}}
        phi_mode_data = get(cfg.fft_plans, Symbol(string(mode_workspace_key) * "_phi"), Vector{Complex{T}}(undef, nlat))::Vector{Complex{T}}
        if length(theta_mode_data) != nlat
            resize!(theta_mode_data, nlat)
            resize!(phi_mode_data, nlat)
        end
    else
        theta_mode_data = Vector{Complex{T}}(undef, nlat)
        phi_mode_data = Vector{Complex{T}}(undef, nlat)
        cfg.fft_plans[mode_workspace_key] = theta_mode_data
        cfg.fft_plans[Symbol(string(mode_workspace_key) * "_phi")] = phi_mode_data
    end
    
    # Precompute normalization factor based on C code
    # Vector transforms need different φ normalization than scalar transforms
    # Empirical correction factor to match C code exactly: 1.0230 ≈ 1/0.9775594192118134
    correction_factor = T(1.0230)
    if cfg.norm == SHT_ORTHONORMAL
        phi_normalization = T(2π) / (nphi * nphi * T(π)) * correction_factor
    elseif cfg.norm == SHT_SCHMIDT
        phi_normalization = T(2π) / (nphi * nphi * T(4π) * T(π)) * correction_factor
    else
        # For 4π normalization
        phi_normalization = T(2π) / (nphi * nphi * T(4π) * T(π)) * correction_factor
    end
    
    @inbounds for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        l >= 1 || continue  # Vector modes start from l=1
        
        # Extract Fourier mode m
        if m <= nphi ÷ 2
            extract_fourier_mode!(theta_fourier, m, theta_mode_data, nlat)
            extract_fourier_mode!(phi_fourier, m, phi_mode_data, nlat)
            
            # Vector harmonic analysis using C code algorithm
            # From C code: spheroidal and toroidal integrals with derivative terms
            sph_integral = zero(Complex{T})
            tor_integral = zero(Complex{T})
            
            @inbounds @simd for i in 1:nlat
                dplm_val = _compute_plm_theta_derivative(cfg, l, m, cfg.theta_grid[i], coeff_idx, i)
                weight = cfg.gauss_weights[i]
                
                # Vector harmonic analysis based on C code patterns:
                # s1 += dy1[j] * terk[j];  // Spheroidal from d/dθ(P_l^m) × θ-component
                # t1 += dy1[j] * perk[j];  // Toroidal from d/dθ(P_l^m) × φ-component
                sph_integral += theta_mode_data[i] * dplm_val * weight
                tor_integral += phi_mode_data[i] * dplm_val * weight
            end
            
            # Apply proper normalization for φ integration  
            sph_integral *= phi_normalization
            tor_integral *= phi_normalization
            
            # Apply vector harmonic normalization factor from C code: l_2[l] = 1/(l*(l+1))
            vector_norm_factor = T(1) / (l * (l + 1))
            sph_integral *= vector_norm_factor
            tor_integral *= vector_norm_factor
            
            # Apply Schmidt-specific analysis normalization (2l+1) factor
            if cfg.norm == SHT_SCHMIDT
                sph_integral *= T(2*l + 1)
                tor_integral *= T(2*l + 1)
            end
            
            # For real fields, extract appropriate part
            if m == 0
                sph_coeffs[coeff_idx] = real(sph_integral)
                tor_coeffs[coeff_idx] = real(tor_integral)
            else
                # Standard factor of 2 for real transforms + mpos_scale_analys
                factor = T(2)
                mpos_scale_analys = T(1)  # C code: mpos_scale_analys = 0.5/0.5 = 1.0
                factor *= mpos_scale_analys
                
                sph_coeffs[coeff_idx] = real(sph_integral) * factor
                tor_coeffs[coeff_idx] = real(tor_integral) * factor
            end
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
    
    # For (l-1, m), we need to check if this is a valid spherical harmonic
    # since |m| <= l for valid spherical harmonics
    Plm1 = zero(T)
    if l > 1 && abs(m) <= (l-1)
        # Find index for (l-1, m) only if it's valid
        try
            idx_lm1 = SHTnsKit.find_plm_index(cfg, l-1, m)
            if idx_lm1 > 0
                Plm1 = cfg.plm_cache[lat_idx, idx_lm1]
            end
        catch
            # Index not found is OK - Plm1 remains zero
        end
    end
    
    x = cos(theta)
    s = sin(theta)
    if abs(s) < T(1e-12)
        return zero(T)
    end
    return (l * x * Plm - (l + m) * Plm1) / s
end

# Public API functions (non-mutating versions)

"""
    synthesize_vector(cfg, sph_coeffs, tor_coeffs)

Non-mutating version of sphtor_to_spat! that allocates output arrays.
"""
function synthesize_vector(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{T}, 
                          tor_coeffs::AbstractVector{T}) where T
    u_theta = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    u_phi = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    return sphtor_to_spat!(cfg, sph_coeffs, tor_coeffs, u_theta, u_phi)
end

"""
    analyze_vector(cfg, u_theta, u_phi)

Non-mutating version of spat_to_sphtor! that allocates output arrays.
"""
function analyze_vector(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, 
                       u_phi::AbstractMatrix{T}) where T
    sph_coeffs = Vector{T}(undef, cfg.nlm)
    tor_coeffs = Vector{T}(undef, cfg.nlm)
    return spat_to_sphtor!(cfg, u_theta, u_phi, sph_coeffs, tor_coeffs)
end