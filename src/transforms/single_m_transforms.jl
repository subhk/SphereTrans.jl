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
    """
    Compute associated Legendre polynomial P_l^m(cos θ) using stable recurrence relations.
    
    Uses the standard three-term recurrence relations:
    - P_l^0 recurrence: (2l-1)x P_{l-1}^0 - (l-1) P_{l-2}^0 = l P_l^0
    - P_m^m initialization: P_m^m = (-1)^m (2m-1)!! sin^m(θ)  
    - P_l^m recurrence: (2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m = (l-m) P_l^m
    """
    
    # Input validation
    @assert l >= 0 "l must be non-negative"
    @assert abs(m) <= l "m must satisfy |m| <= l"
    
    # Handle absolute value of m (symmetry: P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m)
    abs_m = abs(m)
    
    # Base cases for efficiency
    if l == 0
        return (abs_m == 0) ? one(T) : zero(T)
    elseif l == 1
        if abs_m == 0
            return cost
        elseif abs_m == 1
            return -sint  # Includes Condon-Shortley phase (-1)^m
        else
            return zero(T)
        end
    end
    
    # For m = 0, use standard Legendre polynomial recurrence
    if abs_m == 0
        p_prev2 = one(T)        # P_0^0 = 1
        p_prev1 = cost          # P_1^0 = cos(θ)
        
        @inbounds for n in 2:l
            # Recurrence: n P_n^0 = (2n-1) cos(θ) P_{n-1}^0 - (n-1) P_{n-2}^0
            p_curr = (T(2*n - 1) * cost * p_prev1 - T(n - 1) * p_prev2) / T(n)
            p_prev2 = p_prev1
            p_prev1 = p_curr
        end
        
        return p_prev1
    end
    
    # For m > 0, use associated Legendre recurrence
    
    # Step 1: Compute P_m^m using double factorial formula
    # P_m^m = (-1)^m (2m-1)!! sin^m(θ)
    pmm = one(T)
    
    # Compute (2m-1)!! = 1×3×5×...×(2m-1) efficiently
    @inbounds for i in 1:abs_m
        pmm *= T(2*i - 1)
    end
    
    # Apply (-1)^m factor and sin^m(θ)
    pmm *= ((-1)^abs_m) * (sint^abs_m)
    
    if l == abs_m
        return pmm
    end
    
    # Step 2: Compute P_{m+1}^m using first recurrence
    # P_{m+1}^m = cos(θ) (2m+1) P_m^m
    pmp1m = cost * T(2*abs_m + 1) * pmm
    
    if l == abs_m + 1
        return pmp1m
    end
    
    # Step 3: Use general recurrence for P_l^m with l > m+1
    # (l-m) P_l^m = (2l-1) cos(θ) P_{l-1}^m - (l+m-1) P_{l-2}^m
    p_prev2 = pmm      # P_m^m
    p_prev1 = pmp1m    # P_{m+1}^m
    
    @inbounds for n in (abs_m + 2):l
        # Apply recurrence relation
        numerator = T(2*n - 1) * cost * p_prev1 - T(n + abs_m - 1) * p_prev2
        p_curr = numerator / T(n - abs_m)
        
        # Shift for next iteration
        p_prev2 = p_prev1
        p_prev1 = p_curr
    end
    
    return p_prev1
end

# Normalization helper functions
function _get_analysis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    """Analysis normalization factor for spherical harmonic coefficients."""
    if cfg.norm == SHT_ORTHONORMAL
        # Orthonormal: includes factor of 4π for integration
        factor = T(4π)
        # Add m-dependent factor for proper orthogonality
        if m > 0
            factor *= T(2)  # Real part normalization for m > 0
        end
        return factor
    elseif cfg.norm == SHT_FOURPI
        # 4π normalization convention
        return T(4π)
    elseif cfg.norm == SHT_SCHMIDT
        # Schmidt semi-normalized 
        factor = T(4π)
        if m > 0
            # Schmidt normalization removes sqrt(2) for m > 0
            factor /= sqrt(T(2))
        end
        return factor
    else # SHT_REAL_NORM
        # Real normalization (unit sphere integration)
        return T(4π)
    end
end

function _get_synthesis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    """Synthesis normalization factor for spatial reconstruction."""
    if cfg.norm == SHT_ORTHONORMAL
        # Orthonormal: direct coefficient usage
        return one(T)
    elseif cfg.norm == SHT_FOURPI
        # 4π: coefficients need to be rescaled
        factor = T(2*l + 1) / T(4π)
        if m > 0
            # Factorial correction for m > 0
            for k in (l-m+1):(l+m)
                factor /= T(k)
            end
            factor = sqrt(factor)
        end
        return factor
    elseif cfg.norm == SHT_SCHMIDT
        # Schmidt: includes degree-dependent factor
        factor = sqrt(T(2*l + 1))
        if m > 0
            factor *= sqrt(T(2))  # Schmidt includes sqrt(2) for m > 0
        end
        return factor
    else # SHT_REAL_NORM
        # Real normalization
        return sqrt(T(2*l + 1) / T(4π))
    end
end

function _get_vector_analysis_normalization(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    """Vector field analysis normalization accounting for gradient operations."""
    # Vector spherical harmonics need l-dependent scaling
    l_factor = (l == 0) ? T(1) : T(l * (l + 1))
    
    if cfg.norm == SHT_ORTHONORMAL
        # Orthonormal with gradient scaling
        factor = T(4π) / sqrt(l_factor)
        if m > 0
            factor *= T(2)  # Real part normalization
        end
        return factor
    elseif cfg.norm == SHT_FOURPI
        # 4π with vector scaling
        return T(4π) / sqrt(l_factor)
    elseif cfg.norm == SHT_SCHMIDT
        # Schmidt semi-normalized with vector correction
        factor = T(4π) / sqrt(l_factor)
        if m > 0
            factor /= sqrt(T(2))  # Schmidt m > 0 correction
        end
        return factor
    else # SHT_REAL_NORM
        # Real normalization with vector scaling
        return T(4π) / sqrt(l_factor)
    end
end