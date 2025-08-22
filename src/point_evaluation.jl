"""
Point evaluation functions for spherical harmonic representations.
These functions evaluate SH expansions at specific points without requiring a full grid.
"""

"""
    sh_to_point(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}, 
                cost::T, phi::T) where T

Evaluate a spherical harmonic expansion at a single point.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Spherical harmonic coefficients
- `cost`: cos(θ) where θ is colatitude
- `phi`: Longitude in radians

# Returns
- Scalar value at the specified point

This is equivalent to the C library function `SH_to_point()`.
"""
function sh_to_point(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}}, 
                     cost::T, phi::T) where T

    validate_config(cfg)
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    abs(cost) <= 1 || error("cost must be in [-1, 1]")
    
    result = zero(T)
    sint = sqrt(1 - cost*cost)
    
    # Evaluate at the point using Legendre polynomials
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if abs(m) <= cfg.mmax && l <= cfg.lmax
            plm_val = _evaluate_legendre_at_point(cfg, l, m, cost, sint)
            coeff = sh_coeffs[idx]
            
            if m == 0
                result += real(coeff) * plm_val
            elseif m > 0
                result += 2 * real(coeff * cis(m * phi)) * plm_val
            else  # m < 0
                result += 2 * imag(coeff * cis(m * phi)) * plm_val
            end
        end
    end
    
    return result
end

"""
    sh_to_point_cplx(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}},
                      cost::T, phi::T) where T

Evaluate a complex spherical harmonic expansion at a single point.

# Arguments  
- `cfg`: SHTns configuration
- `sh_coeffs`: Complex spherical harmonic coefficients (full complex expansion)
- `cost`: cos(θ) where θ is colatitude  
- `phi`: Longitude in radians

# Returns
- Complex value at the specified point

This is equivalent to the C library function `SH_to_point_cplx()`.
"""
function sh_to_point_cplx(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{Complex{T}},
                          cost::T, phi::T) where T
                          
    validate_config(cfg)
    
    result = zero(Complex{T})
    sint = sqrt(1 - cost*cost)
    
    # For complex coefficients, we need the full (l,m) range including negative m
    for l in 0:cfg.lmax
        for m in -min(l, cfg.mmax):min(l, cfg.mmax)
            # Calculate linear index for complex coefficient storage
            if m >= 0
                idx = l*(l+1) + m + 1  # +1 for Julia 1-based indexing
            else
                idx = l*(l+1) - m + 1
            end
            
            if idx <= length(sh_coeffs)
                plm_val = _evaluate_legendre_at_point(cfg, l, abs(m), cost, sint)
                coeff = sh_coeffs[idx]
                
                phase = cis(m * phi)
                if m < 0 && (abs(m) % 2) == 1
                    # Handle Condon-Shortley phase for negative m
                    phase *= -1
                end
                
                result += coeff * plm_val * phase
            end
        end
    end
    
    return result
end

"""
    shqst_to_point(cfg::SHTnsConfig{T}, 
                   q_coeffs::AbstractVector{Complex{T}},
                   s_coeffs::AbstractVector{Complex{T}}, 
                   t_coeffs::AbstractVector{Complex{T}},
                   cost::T, phi::T) where T

Evaluate radial-spheroidal-toroidal vector expansion at a single point.

# Arguments
- `cfg`: SHTns configuration  
- `q_coeffs`: Radial (scalar) coefficients
- `s_coeffs`: Spheroidal coefficients
- `t_coeffs`: Toroidal coefficients  
- `cost`: cos(θ) where θ is colatitude
- `phi`: Longitude in radians

# Returns
- `(vr, vt, vp)`: Vector components at the specified point

This is equivalent to the C library function `SHqst_to_point()`.
"""
function shqst_to_point(cfg::SHTnsConfig{T},
                       q_coeffs::AbstractVector{Complex{T}},
                       s_coeffs::AbstractVector{Complex{T}}, 
                       t_coeffs::AbstractVector{Complex{T}},
                       cost::T, phi::T) where T

    validate_config(cfg)
    length(q_coeffs) == cfg.nlm || error("q_coeffs length must equal nlm")
    length(s_coeffs) == cfg.nlm || error("s_coeffs length must equal nlm") 
    length(t_coeffs) == cfg.nlm || error("t_coeffs length must equal nlm")
    abs(cost) <= 1 || error("cost must be in [-1, 1]")
    
    vr = zero(T)
    vt = zero(T)  
    vp = zero(T)
    
    sint = sqrt(1 - cost*cost)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if abs(m) <= cfg.mmax && l <= cfg.lmax && l > 0
            plm_val = _evaluate_legendre_at_point(cfg, l, m, cost, sint)
            dplm_val = _evaluate_legendre_derivative_at_point(cfg, l, m, cost, sint)
            
            # Phase factor
            phase_m = cis(m * phi)
            
            # Radial component (from scalar potential)
            q_coeff = q_coeffs[idx]
            if m == 0
                vr += real(q_coeff) * plm_val
            elseif m > 0
                vr += 2 * real(q_coeff * phase_m) * plm_val
            else
                vr += 2 * imag(q_coeff * phase_m) * plm_val
            end
            
            # Horizontal components from spheroidal and toroidal potentials
            s_coeff = s_coeffs[idx]
            t_coeff = t_coeffs[idx]
            
            if l > 0
                # Spheroidal contribution to θ component: ∂S/∂θ
                sph_theta = dplm_val
                
                # Spheroidal contribution to φ component: (1/sin θ) ∂S/∂φ  
                sph_phi = (sint > 1e-12) ? (m * plm_val / sint) : zero(T)
                
                # Toroidal contribution to θ component: -(1/sin θ) ∂T/∂φ
                tor_theta = (sint > 1e-12) ? (-m * plm_val / sint) : zero(T)
                
                # Toroidal contribution to φ component: ∂T/∂θ
                tor_phi = dplm_val
                
                # Combine contributions
                if m == 0
                    vt += real(s_coeff) * sph_theta + real(t_coeff) * tor_theta
                    vp += real(s_coeff) * sph_phi + real(t_coeff) * tor_phi
                elseif m > 0
                    vt += 2 * real(s_coeff * phase_m) * sph_theta + 2 * real(t_coeff * phase_m) * tor_theta
                    vp += 2 * real(Complex{T}(0, 1) * s_coeff * phase_m) * sph_phi + 2 * real(Complex{T}(0, 1) * t_coeff * phase_m) * tor_phi
                else
                    vt += 2 * imag(s_coeff * phase_m) * sph_theta + 2 * imag(t_coeff * phase_m) * tor_theta
                    vp += 2 * imag(Complex{T}(0, 1) * s_coeff * phase_m) * sph_phi + 2 * imag(Complex{T}(0, 1) * t_coeff * phase_m) * tor_phi
                end
            end
        end
    end
    
    return (vr, vt, vp)
end

"""
    sh_to_grad_point(cfg::SHTnsConfig{T}, s_coeffs::AbstractVector{Complex{T}},
                     cost::T, phi::T) where T

Evaluate the gradient of a scalar field at a single point.

# Arguments
- `cfg`: SHTns configuration
- `s_coeffs`: Spherical harmonic coefficients of the scalar field
- `cost`: cos(θ) where θ is colatitude
- `phi`: Longitude in radians

# Returns  
- `(gr, gt, gp)`: Gradient components at the specified point

This computes ∇S = (∂S/∂r, (1/r)∂S/∂θ, (1/(r sin θ))∂S/∂φ).
For unit sphere (r=1): ∇S = (0, ∂S/∂θ, (1/sin θ)∂S/∂φ).

This is equivalent to the C library function `SH_to_grad_point()`.
"""
function sh_to_grad_point(cfg::SHTnsConfig{T}, s_coeffs::AbstractVector{Complex{T}},
                         cost::T, phi::T) where T
    validate_config(cfg)
    length(s_coeffs) == cfg.nlm || error("s_coeffs length must equal nlm")
    abs(cost) <= 1 || error("cost must be in [-1, 1]")
    
    gr = zero(T)  # Radial derivative (zero for surface field)
    gt = zero(T)  # θ component: ∂S/∂θ
    gp = zero(T)  # φ component: (1/sin θ)∂S/∂φ
    
    sint = sqrt(1 - cost*cost)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if abs(m) <= cfg.mmax && l <= cfg.lmax && l > 0
            plm_val = _evaluate_legendre_at_point(cfg, l, m, cost, sint)
            dplm_val = _evaluate_legendre_derivative_at_point(cfg, l, m, cost, sint)
            
            coeff = s_coeffs[idx]
            phase_m = cis(m * phi)
            
            # θ component: ∂S/∂θ  
            if m == 0
                gt += real(coeff) * dplm_val
            elseif m > 0
                gt += 2 * real(coeff * phase_m) * dplm_val
            else
                gt += 2 * imag(coeff * phase_m) * dplm_val
            end
            
            # φ component: (1/sin θ)∂S/∂φ
            if sint > 1e-12
                phi_contribution = m * plm_val / sint
                if m == 0
                    gp += 0  # No φ dependence for m=0
                elseif m > 0
                    gp += 2 * real(Complex{T}(0, 1) * coeff * phase_m) * phi_contribution
                else
                    gp += 2 * imag(Complex{T}(0, 1) * coeff * phase_m) * phi_contribution
                end
            end
        end
    end
    
    return (gr, gt, gp)
end

# Helper functions for point evaluation

"""
    _evaluate_legendre_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T

Evaluate associated Legendre polynomial P_l^m(cos θ) at a specific point.
"""
function _evaluate_legendre_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T
    m = abs(m)  # Only need |m|
    
    if l == 0 && m == 0
        return one(T)
    elseif l == 1 && m == 0
        return cost
    elseif l == 1 && m == 1
        return -sint  # Note: includes Condon-Shortley phase
    end
    
    # Use recurrence relation for higher degrees
    # This is a simplified version - in practice, you'd want the optimized version
    # from gauss_legendre.jl
    
    # Initialize for m=0 case
    if m == 0
        pmm = one(T)
        pmp1m = cost
        
        if l == 0
            return pmm
        elseif l == 1  
            return pmp1m
        end
        
        # Recurrence for l > 1, m = 0
        for ll in 2:l
            pll = ((2*ll - 1) * cost * pmp1m - (ll - 1) * pmm) / ll
            pmm = pmp1m
            pmp1m = pll
        end
        
        return pmp1m
    else
        # For m > 0, start with P_m^m
        pmm = one(T)
        fact = one(T)
        for i in 1:m
            fact *= (2*i - 1)
        end
        pmm = (-1)^m * fact * sint^m
        
        if l == m
            return pmm
        end
        
        # P_{m+1}^m
        pmp1m = cost * (2*m + 1) * pmm
        
        if l == m + 1
            return pmp1m
        end
        
        # Recurrence for l > m+1
        for ll in (m+2):l
            pll = ((2*ll - 1) * cost * pmp1m - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmp1m
            pmp1m = pll
        end
        
        return pmp1m
    end
end

"""
    _evaluate_legendre_derivative_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T

Evaluate derivative dP_l^m/dθ at a specific point.
"""
function _evaluate_legendre_derivative_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T
    m_abs = abs(m)
    
    if l == 0
        return zero(T)
    end
    
    # Use the relation: dP_l^m/dθ = -sint * dP_l^m/d(cos θ)
    # and dP_l^m/d(cos θ) can be computed using recurrence relations
    
    plm = _evaluate_legendre_at_point(cfg, l, m_abs, cost, sint)
    
    if m_abs == 0
        # For m=0: dP_l/d(cos θ) = l * [cos θ * P_l - P_{l-1}] / (cos²θ - 1)
        if l == 1
            return -sint
        else
            pl_minus_1 = _evaluate_legendre_at_point(cfg, l-1, 0, cost, sint)
            if abs(cost) < 0.99  # Avoid division by zero near poles
                dplm_dcost = l * (cost * plm - pl_minus_1) / (cost^2 - 1)
            else
                # Use alternative form near poles
                dplm_dcost = l * (l + 1) * cost * plm / 2
            end
            return -sint * dplm_dcost
        end
    else
        # For m ≠ 0, use the general recurrence relation
        if l > m_abs
            pl_minus_1_m = _evaluate_legendre_at_point(cfg, l-1, m_abs, cost, sint)
            dplm_dcost = (l * cost * plm - (l + m_abs) * pl_minus_1_m) / (cost^2 - 1)
        else
            # For l = |m|, use direct computation
            dplm_dcost = m_abs * cost * plm / sint^2
        end
        return -sint * dplm_dcost
    end
end