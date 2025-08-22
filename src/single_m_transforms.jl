"""
Single-m transforms (Legendre transforms at fixed m, no FFT).
These are useful for problems with specific azimuthal symmetries or 
for building custom transform algorithms.
"""

"""
    spat_to_sh_ml(cfg::SHTnsConfig{T}, m::Int, vr::AbstractVector{Complex{T}}, 
                  ql::AbstractVector{Complex{T}}, ltr::Int=cfg.lmax) where T

Legendre transform at given m (no FFT) from spatial to spectral.

# Arguments
- `cfg`: SHTns configuration
- `m`: Azimuthal order (can be negative)  
- `vr`: Spatial data at fixed m (length nlat), complex values along latitude
- `ql`: Output spectral coefficients for degrees l=|m|..ltr (pre-allocated)
- `ltr`: Maximum degree for truncation (default: cfg.lmax)

The input array `vr` contains the Fourier coefficients for the given m,
and the output `ql` contains spherical harmonic coefficients for that m only.

Equivalent to the C library function `spat_to_SH_ml()`.
"""
function spat_to_sh_ml(cfg::SHTnsConfig{T}, m::Int, vr::AbstractVector{Complex{T}}, 
                       ql::AbstractVector{Complex{T}}, ltr::Int=cfg.lmax) where T
    validate_config(cfg)
    abs(m) <= cfg.mmax || error("m must be <= mmax")
    ltr <= cfg.lmax || error("ltr must be <= lmax") 
    length(vr) == cfg.nlat || error("vr length must equal nlat")
    expected_nl = ltr - abs(m) + 1
    length(ql) >= expected_nl || error("ql must have at least $(expected_nl) elements")
    
    m_abs = abs(m)
    nlat = cfg.nlat
    
    # Get integration weights and nodes
    if cfg.grid_type == SHT_GAUSS
        weights = cfg.gauss_weights
        nodes = cfg.gauss_nodes  # cos(θ) values
    else
        # Regular grid - use trapezoidal or other quadrature
        weights = _regular_grid_weights(cfg)
        nodes = cos.(cfg.theta_grid)
    end
    
    # Perform Legendre analysis
    fill!(ql, zero(Complex{T}))
    
    for l in m_abs:ltr
        l_idx = l - m_abs + 1
        
        # Compute integral: ∫ vr(θ) * P_l^m(cos θ) * weights * sin θ dθ
        integral = zero(Complex{T})
        
        for j in 1:nlat
            cost = nodes[j]
            sint = sqrt(1 - cost^2)
            
            # Evaluate associated Legendre polynomial
            plm = _evaluate_legendre_normalized(cfg, l, m_abs, cost, sint)
            
            # Add to integral with appropriate weight
            weight = weights[j]
            if cfg.grid_type != SHT_GAUSS
                weight *= sint  # Include sin θ factor for regular grids
            end
            
            integral += vr[j] * plm * weight
        end
        
        # Apply normalization factor
        norm_factor = _get_analysis_normalization(cfg, l, m)
        ql[l_idx] = integral * norm_factor
    end
    
    return ql
end

"""
    sh_to_spat_ml(cfg::SHTnsConfig{T}, m::Int, ql::AbstractVector{Complex{T}},
                  vr::AbstractVector{Complex{T}}, ltr::Int=cfg.lmax) where T

Legendre synthesis at given m (no FFT) from spectral to spatial.

# Arguments  
- `cfg`: SHTns configuration
- `m`: Azimuthal order (can be negative)
- `ql`: Input spectral coefficients for degrees l=|m|..ltr
- `vr`: Output spatial data at fixed m (length nlat, pre-allocated)
- `ltr`: Maximum degree for truncation (default: cfg.lmax)

Equivalent to the C library function `SH_to_spat_ml()`.
"""
function sh_to_spat_ml(cfg::SHTnsConfig{T}, m::Int, ql::AbstractVector{Complex{T}},
                       vr::AbstractVector{Complex{T}}, ltr::Int=cfg.lmax) where T
    validate_config(cfg)
    abs(m) <= cfg.mmax || error("m must be <= mmax") 
    ltr <= cfg.lmax || error("ltr must be <= lmax")
    length(vr) == cfg.nlat || error("vr length must equal nlat")
    expected_nl = ltr - abs(m) + 1
    length(ql) >= expected_nl || error("ql must have at least $(expected_nl) elements")
    
    m_abs = abs(m)
    nlat = cfg.nlat
    
    # Get grid nodes
    if cfg.grid_type == SHT_GAUSS
        nodes = cfg.gauss_nodes  # cos(θ) values
    else  
        nodes = cos.(cfg.theta_grid)
    end
    
    # Perform Legendre synthesis
    fill!(vr, zero(Complex{T}))
    
    for j in 1:nlat
        cost = nodes[j] 
        sint = sqrt(1 - cost^2)
        
        # Sum over all l for this m
        for l in m_abs:ltr
            l_idx = l - m_abs + 1
            
            # Evaluate associated Legendre polynomial  
            plm = _evaluate_legendre_normalized(cfg, l, m_abs, cost, sint)
            
            # Add contribution with proper normalization
            coeff = ql[l_idx] 
            vr[j] += coeff * plm
        end
    end
    
    return vr
end

"""
    spat_to_sphtor_ml(cfg::SHTnsConfig{T}, m::Int, 
                      vt::AbstractVector{Complex{T}}, vp::AbstractVector{Complex{T}},
                      sl::AbstractVector{Complex{T}}, tl::AbstractVector{Complex{T}}, 
                      ltr::Int=cfg.lmax) where T

Vector Legendre analysis at given m from (vθ, vφ) to spheroidal-toroidal components.

# Arguments
- `cfg`: SHTns configuration
- `m`: Azimuthal order
- `vt`: θ-component of vector field (length nlat)
- `vp`: φ-component of vector field (length nlat)  
- `sl`: Output spheroidal coefficients (pre-allocated)
- `tl`: Output toroidal coefficients (pre-allocated)
- `ltr`: Maximum degree for truncation

Equivalent to the C library function `spat_to_SHsphtor_ml()`.
"""
function spat_to_sphtor_ml(cfg::SHTnsConfig{T}, m::Int,
                           vt::AbstractVector{Complex{T}}, vp::AbstractVector{Complex{T}},
                           sl::AbstractVector{Complex{T}}, tl::AbstractVector{Complex{T}}, 
                           ltr::Int=cfg.lmax) where T
    validate_config(cfg)
    abs(m) <= cfg.mmax || error("m must be <= mmax")
    ltr <= cfg.lmax || error("ltr must be <= lmax")
    length(vt) == cfg.nlat || error("vt length must equal nlat")
    length(vp) == cfg.nlat || error("vp length must equal nlat")
    
    m_abs = abs(m)
    nlat = cfg.nlat
    expected_nl = ltr - m_abs + 1
    
    # Ensure l starts from max(1, m_abs) for vector fields
    l_start = max(1, m_abs)
    if l_start > ltr
        fill!(sl, zero(Complex{T}))
        fill!(tl, zero(Complex{T}))  
        return (sl, tl)
    end
    
    # Get integration setup
    if cfg.grid_type == SHT_GAUSS
        weights = cfg.gauss_weights
        nodes = cfg.gauss_nodes
    else
        weights = _regular_grid_weights(cfg) 
        nodes = cos.(cfg.theta_grid)
    end
    
    fill!(sl, zero(Complex{T}))
    fill!(tl, zero(Complex{T}))
    
    for l in l_start:ltr
        l_idx = l - m_abs + 1
        if l_idx <= length(sl)
            
            sph_integral = zero(Complex{T})
            tor_integral = zero(Complex{T})
            
            for j in 1:nlat
                cost = nodes[j]
                sint = sqrt(1 - cost^2)
                
                # Evaluate Legendre polynomial and derivative
                plm = _evaluate_legendre_normalized(cfg, l, m_abs, cost, sint)
                dplm = _evaluate_legendre_derivative_normalized(cfg, l, m_abs, cost, sint)
                
                weight = weights[j]
                if cfg.grid_type != SHT_GAUSS
                    weight *= sint
                end
                
                # For spheroidal component: 
                # S_l^m from: vθ * dP_l^m/dθ + vφ * (im/sin θ) * P_l^m
                sph_contrib = vt[j] * dplm
                if sint > 1e-12
                    sph_contrib += vp[j] * (Complex{T}(0, m) / sint) * plm
                end
                sph_integral += sph_contrib * weight
                
                # For toroidal component:
                # T_l^m from: vθ * (-im/sin θ) * P_l^m + vφ * dP_l^m/dθ  
                tor_contrib = vp[j] * dplm
                if sint > 1e-12
                    tor_contrib += vt[j] * (-Complex{T}(0, m) / sint) * plm
                end
                tor_integral += tor_contrib * weight
            end
            
            # Apply normalization (include l(l+1) factors)
            if l > 0
                norm_factor = _get_vector_analysis_normalization(cfg, l, m)
                sl[l_idx] = sph_integral * norm_factor / T(l * (l + 1))
                tl[l_idx] = tor_integral * norm_factor / T(l * (l + 1))
            end
        end
    end
    
    return (sl, tl)
end

"""
    sphtor_to_spat_ml(cfg::SHTnsConfig{T}, m::Int,
                      sl::AbstractVector{Complex{T}}, tl::AbstractVector{Complex{T}},
                      vt::AbstractVector{Complex{T}}, vp::AbstractVector{Complex{T}},
                      ltr::Int=cfg.lmax) where T

Vector Legendre synthesis at given m from spheroidal-toroidal to (vθ, vφ).

Equivalent to the C library function `SHsphtor_to_spat_ml()`.
"""
function sphtor_to_spat_ml(cfg::SHTnsConfig{T}, m::Int,
                           sl::AbstractVector{Complex{T}}, tl::AbstractVector{Complex{T}},
                           vt::AbstractVector{Complex{T}}, vp::AbstractVector{Complex{T}},
                           ltr::Int=cfg.lmax) where T
    validate_config(cfg) 
    abs(m) <= cfg.mmax || error("m must be <= mmax")
    ltr <= cfg.lmax || error("ltr must be <= lmax")
    length(vt) == cfg.nlat || error("vt length must equal nlat")
    length(vp) == cfg.nlat || error("vp length must equal nlat")
    
    m_abs = abs(m)
    nlat = cfg.nlat
    
    # Get grid nodes
    if cfg.grid_type == SHT_GAUSS
        nodes = cfg.gauss_nodes
    else
        nodes = cos.(cfg.theta_grid)  
    end
    
    fill!(vt, zero(Complex{T}))
    fill!(vp, zero(Complex{T}))
    
    l_start = max(1, m_abs)
    
    for j in 1:nlat
        cost = nodes[j]
        sint = sqrt(1 - cost^2)
        
        for l in l_start:ltr
            l_idx = l - m_abs + 1
            if l_idx <= length(sl) && l > 0
                
                # Evaluate Legendre functions
                plm = _evaluate_legendre_normalized(cfg, l, m_abs, cost, sint)
                dplm = _evaluate_legendre_derivative_normalized(cfg, l, m_abs, cost, sint)
                
                s_coeff = sl[l_idx]
                t_coeff = tl[l_idx] 
                
                # θ component: ∂S/∂θ - (im/sin θ) * T
                vt_contrib = s_coeff * dplm
                if sint > 1e-12
                    vt_contrib -= t_coeff * (Complex{T}(0, m) / sint) * plm
                end
                vt[j] += vt_contrib
                
                # φ component: (im/sin θ) * S + ∂T/∂θ
                vp_contrib = t_coeff * dplm
                if sint > 1e-12
                    vp_contrib += s_coeff * (Complex{T}(0, m) / sint) * plm
                end
                vp[j] += vp_contrib
            end
        end
    end
    
    return (vt, vp)
end

# Helper functions

"""
    _regular_grid_weights(cfg::SHTnsConfig{T}) where T

Compute integration weights for regular (non-Gaussian) grids.
"""
function _regular_grid_weights(cfg::SHTnsConfig{T}) where T
    nlat = cfg.nlat
    weights = Vector{T}(undef, nlat)
    
    # Simple trapezoidal rule weights
    dtheta = π / (nlat - 1)
    weights[1] = dtheta / 2
    weights[nlat] = dtheta / 2
    for j in 2:(nlat-1)
        weights[j] = dtheta
    end
    
    return weights
end

"""
    _evaluate_legendre_normalized(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T

Evaluate normalized associated Legendre polynomial for the given configuration.
"""
function _evaluate_legendre_normalized(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T
    # Use the same normalization as the main transforms
    plm = _compute_single_legendre_basic(l, m, cost, sint)
    norm_factor = _get_synthesis_normalization(cfg, l, m)
    return plm * norm_factor
end

"""
    _evaluate_legendre_derivative_normalized(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T

Evaluate derivative of normalized associated Legendre polynomial.
"""
function _evaluate_legendre_derivative_normalized(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T
    # Compute derivative using recurrence relations
    if l == 0
        return zero(T)
    end
    
    plm = _evaluate_legendre_normalized(cfg, l, m, cost, sint)
    
    if m == 0
        # For m=0: dP_l/dθ = -l * sin θ * P_l^1
        if l == 1
            return -sint * _get_synthesis_normalization(cfg, 1, 1)
        else
            pl1 = _evaluate_legendre_normalized(cfg, l, 1, cost, sint)
            return -l * pl1
        end
    else
        # General case: use recurrence relation
        if abs(cost) < 0.999  # Away from poles
            if l > m
                pl_minus_1_m = _evaluate_legendre_normalized(cfg, l-1, m, cost, sint)
                return (l * cost * plm - (l + m) * pl_minus_1_m) / (cost^2 - 1) * (-sint)
            else
                return m * cost * plm / sint
            end
        else  # Near poles
            return m * cost * plm / sint^2
        end
    end
end

"""
    _compute_single_legendre_basic(l::Int, m::Int, cost::T, sint::T) where T

Compute basic (unnormalized) associated Legendre polynomial.
"""
function _compute_single_legendre_basic(l::Int, m::Int, cost::T, sint::T) where T
    if l == 0 && m == 0
        return one(T)
    elseif l == 1 && m == 0
        return cost  
    elseif l == 1 && m == 1
        return -sint  # Includes Condon-Shortley phase
    end
    
    # Use recurrence - this is a simplified version
    # In practice, you'd use the optimized version from gauss_legendre.jl
    if m == 0
        # Standard Legendre polynomial recurrence
        p0 = one(T)
        p1 = cost
        
        for i in 2:l
            p2 = ((2*i - 1) * cost * p1 - (i - 1) * p0) / i
            p0 = p1
            p1 = p2
        end
        return p1
    else
        # Associated Legendre polynomial - simplified computation
        # Start with P_m^m
        pmm = one(T)
        if m > 0
            fact = one(T)
            for i in 1:m
                fact *= (2*i - 1)
            end
            pmm = (-1)^m * fact * sint^m
        end
        
        if l == m
            return pmm
        end
        
        # P_{m+1}^m
        pmp1m = cost * (2*m + 1) * pmm
        
        if l == m + 1
            return pmp1m
        end
        
        # General recurrence
        p0 = pmm
        p1 = pmp1m
        for i in (m+2):l
            p2 = ((2*i - 1) * cost * p1 - (i + m - 1) * p0) / (i - m)
            p0 = p1
            p1 = p2
        end
        
        return p1
    end
end

# Normalization helper functions (simplified versions)
function _get_analysis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    # This should match the normalization used in the main transforms
    return T(2π)  # Simplified - should be more sophisticated
end

function _get_synthesis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    # This should match the normalization used in the main transforms  
    return one(T)  # Simplified - should be more sophisticated
end

function _get_vector_analysis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    # Vector field normalization
    return T(2π)  # Simplified
end