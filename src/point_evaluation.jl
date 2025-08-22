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
    """
    Evaluate associated Legendre polynomial P_l^m(cos θ) at a specific point using 
    optimized recurrence relations.
    
    This implementation uses:
    - Standard 3-term recurrence for P_l^0
    - Efficient double factorial computation for P_m^m 
    - Forward recurrence for P_l^m with l > m
    - Type-stable operations with @inbounds optimization
    """
    
    # Input validation
    @assert l >= 0 "l must be non-negative"
    @assert abs(m) <= l "m must satisfy |m| <= l"
    
    abs_m = abs(m)  # Work with |m| only
    
    # Fast base cases
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
        # P_0^0 = 1, P_1^0 = cos(θ)
        p_prev2 = one(T)
        p_prev1 = cost
        
        # Three-term recurrence: n P_n^0 = (2n-1) cos(θ) P_{n-1}^0 - (n-1) P_{n-2}^0
        @inbounds for n in 2:l
            p_curr = (T(2*n - 1) * cost * p_prev1 - T(n - 1) * p_prev2) / T(n)
            p_prev2 = p_prev1
            p_prev1 = p_curr
        end
        
        return p_prev1
    end
    
    # For m > 0, use associated Legendre recurrence
    
    # Step 1: Compute P_m^m = (-1)^m (2m-1)!! sin^m(θ)
    pmm = one(T)
    
    # Efficient computation of double factorial (2m-1)!! = 1×3×5×...×(2m-1)
    @inbounds for i in 1:abs_m
        pmm *= T(2*i - 1)
    end
    
    # Apply Condon-Shortley phase and sin^m factor
    pmm *= ((-1)^abs_m) * (sint^abs_m)
    
    if l == abs_m
        return pmm
    end
    
    # Step 2: Compute P_{m+1}^m = cos(θ) (2m+1) P_m^m
    pmp1m = cost * T(2*abs_m + 1) * pmm
    
    if l == abs_m + 1
        return pmp1m
    end
    
    # Step 3: Forward recurrence for P_l^m with l > m+1
    # (l-m) P_l^m = (2l-1) cos(θ) P_{l-1}^m - (l+m-1) P_{l-2}^m
    p_prev2 = pmm      # P_m^m
    p_prev1 = pmp1m    # P_{m+1}^m
    
    @inbounds for n in (abs_m + 2):l
        numerator = T(2*n - 1) * cost * p_prev1 - T(n + abs_m - 1) * p_prev2
        p_curr = numerator / T(n - abs_m)
        
        # Shift values for next iteration
        p_prev2 = p_prev1
        p_prev1 = p_curr
    end
    
    return p_prev1
end

"""
    _evaluate_legendre_derivative_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T

Evaluate derivative dP_l^m/dθ at a specific point.
"""
function _evaluate_legendre_derivative_at_point(cfg::SHTnsConfig{T}, l::Int, m::Int, cost::T, sint::T) where T
    """
    Evaluate derivative dP_l^m/dθ at a specific point using recurrence-based approach.
    
    Uses the standard derivative formula:
    dP_l^m/dθ = (1/sin θ) × [l cos(θ) P_l^m - (l+m) P_{l-1}^m]
    
    With special handling for poles and edge cases.
    """
    
    # Input validation
    @assert l >= 0 "l must be non-negative"
    @assert abs(m) <= l "m must satisfy |m| <= l"
    
    abs_m = abs(m)
    
    # Base cases
    if l == 0
        return zero(T)
    elseif l == 1
        if abs_m == 0
            return -sint  # d/dθ [cos(θ)] = -sin(θ)
        elseif abs_m == 1
            return -cost  # d/dθ [-sin(θ)] = -cos(θ)
        else
            return zero(T)
        end
    end
    
    # Get current P_l^m value
    plm_current = _evaluate_legendre_at_point(cfg, l, abs_m, cost, sint)
    
    # Handle special case where l = |m|
    if l == abs_m
        # For P_m^m, use: dP_m^m/dθ = m cos(θ)/sin(θ) × P_m^m
        if abs_m == 0
            return zero(T)
        else
            if abs(sint) < eps(T)
                # At poles, derivative becomes infinite
                return T(Inf) * sign(cost)
            else
                return T(abs_m) * cost / sint * plm_current
            end
        end
    end
    
    # General case: l > |m|
    # Use derivative formula: dP_l^m/dθ = (1/sin θ) × [l cos(θ) P_l^m - (l+m) P_{l-1}^m]
    
    # Get P_{l-1}^m value  
    plm_prev = _evaluate_legendre_at_point(cfg, l-1, abs_m, cost, sint)
    
    # Check for pole singularity
    if abs(sint) < eps(T)
        if abs_m == 0
            # For m=0 at poles, use polynomial derivative formula
            # dP_l/dx|_{x=±1} = ±l(l+1)/2
            return T(l * (l + 1) / 2) * sign(cost) * (-sint)
        else
            # For m>0 at poles, derivative diverges
            return T(Inf) * sign(cost)
        end
    else
        # Apply standard derivative formula
        numerator = T(l) * cost * plm_current - T(l + abs_m) * plm_prev
        return numerator / sint
    end
end