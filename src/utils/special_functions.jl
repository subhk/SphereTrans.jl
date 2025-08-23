"""
Special value functions and Legendre polynomial evaluation.
These correspond to the SHTns C library special functions.
"""

"""
    sh00_1(cfg::SHTnsConfig{T}) where T

Return the spherical harmonic representation of the constant function 1.
This is the (l=0, m=0) coefficient that represents unity on the sphere.

Equivalent to the C library function `sh00_1()`.
"""
function sh00_1(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    # The (0,0) coefficient for unity depends on the normalization
    if cfg.norm == SHT_ORTHONORMAL
        return 1/sqrt(4*π)
    elseif cfg.norm == SHT_FOURPI
        return 1/(2*sqrt(π))
    elseif cfg.norm == SHT_SCHMIDT
        return 1/(2*sqrt(π))
    elseif cfg.norm == SHT_REAL_NORM
        return T(0.5)
    else
        throw(ArgumentError("Unknown normalization: $(cfg.norm)"))
    end
end

"""
    sh10_ct(cfg::SHTnsConfig{T}) where T

Return the spherical harmonic representation of cos(θ).
This is the (l=1, m=0) coefficient.

Equivalent to the C library function `sh10_ct()`.
"""
function sh10_ct(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    # The (1,0) coefficient for cos(θ) depends on the normalization
    if cfg.norm == SHT_ORTHONORMAL
        return sqrt(3/(4*π))
    elseif cfg.norm == SHT_FOURPI
        return sqrt(3/π)
    elseif cfg.norm == SHT_SCHMIDT
        return sqrt(3/(4*π))
    elseif cfg.norm == SHT_REAL_NORM
        return sqrt(T(3)/2)
    else
        throw(ArgumentError("Unknown normalization: $(cfg.norm)"))
    end
end

"""
    sh11_st(cfg::SHTnsConfig{T}) where T

Return the spherical harmonic representation of sin(θ)cos(φ).
This is the (l=1, m=1) coefficient.

Equivalent to the C library function `sh11_st()`.
"""
function sh11_st(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    # The (1,1) coefficient for sin(θ)cos(φ) depends on the normalization
    if cfg.norm == SHT_ORTHONORMAL
        return -sqrt(3/(8*π))  # Note negative sign from Condon-Shortley phase
    elseif cfg.norm == SHT_FOURPI
        return -sqrt(3/(2*π))
    elseif cfg.norm == SHT_SCHMIDT
        return -sqrt(3/(8*π))
    elseif cfg.norm == SHT_REAL_NORM
        return -sqrt(T(3)/4)
    else
        throw(ArgumentError("Unknown normalization: $(cfg.norm)"))
    end
end

"""
    shlm_e1(cfg::SHTnsConfig{T}, l::Int, m::Int) where T

Return the (l,m) spherical harmonic coefficient corresponding to unit energy.
This is useful for initializing test cases with known energy content.

Equivalent to the C library function `shlm_e1()`.
"""
function shlm_e1(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    validate_config(cfg)
    0 <= l <= cfg.lmax || error("l must be in [0, lmax]")
    abs(m) <= min(l, cfg.mmax) || error("m must be in [-min(l, mmax), min(l, mmax)]")
    
    # Unit energy coefficient depends on normalization
    if cfg.norm == SHT_ORTHONORMAL
        # Orthonormal: energy = |coefficient|²
        return one(T)
    elseif cfg.norm == SHT_FOURPI
        # 4π normalization
        factor = (m == 0) ? T(1) : T(2)
        return 1/sqrt(factor)
    elseif cfg.norm == SHT_SCHMIDT
        # Schmidt normalization
        factor = (m == 0) ? T(1) : T(2)
        return 1/sqrt((2*l + 1) * factor)
    elseif cfg.norm == SHT_REAL_NORM
        # Real normalization
        return 1/sqrt(T(2*l + 1))
    else
        throw(ArgumentError("Unknown normalization: $(cfg.norm)"))
    end
end

"""
    gauss_weights(cfg::SHTnsConfig{T}) where T

Return the Gaussian quadrature weights for the current grid.
Returns an empty array if not using a Gaussian grid.

Equivalent to the C library function `shtns_gauss_wts()`.
"""
function gauss_weights(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    if cfg.grid_type == SHT_GAUSS
        return copy(cfg.gauss_weights)
    else
        return T[]
    end
end

"""
    legendre_sphPlm_array(cfg::SHTnsConfig{T}, lmax::Int, m::Int, x::T) where T

Compute values of Legendre polynomials normalized for spherical harmonics,
for a range of l=m..lmax, at given m and x=cos(θ).

# Arguments
- `cfg`: SHTns configuration
- `lmax`: Maximum degree to compute  
- `m`: Order (must be ≥ 0)
- `x`: Argument, x=cos(θ)

# Returns
- Vector of length (lmax-m+1) containing P_l^m(x) for l=m,m+1,...,lmax

Equivalent to the C library function `legendre_sphPlm_array()`.
"""
function legendre_sphPlm_array(cfg::SHTnsConfig{T}, lmax::Int, m::Int, x::T) where T
    validate_config(cfg)
    lmax >= 0 || error("lmax must be non-negative")
    m >= 0 || error("m must be non-negative")
    m <= lmax || error("m must be <= lmax")
    abs(x) <= 1 || error("x must be in [-1, 1]")
    
    n_values = lmax - m + 1
    yl = Vector{T}(undef, n_values)
    
    # Compute using stable recurrence
    _compute_legendre_array!(yl, lmax, m, x, cfg.norm)
    
    return yl
end

"""
    legendre_sphPlm_deriv_array(cfg::SHTnsConfig{T}, lmax::Int, m::Int, x::T, sint::T) where T

Compute values and derivatives of Legendre polynomials normalized for spherical harmonics.

# Arguments
- `cfg`: SHTns configuration
- `lmax`: Maximum degree to compute
- `m`: Order (must be ≥ 0)  
- `x`: Argument, x=cos(θ)
- `sint`: sin(θ) = sqrt(1-x²)

# Returns  
- `(yl, dyl)`: Tuple of vectors containing P_l^m(x) and dP_l^m/dx

Equivalent to the C library function `legendre_sphPlm_deriv_array()`.
"""
function legendre_sphPlm_deriv_array(cfg::SHTnsConfig{T}, lmax::Int, m::Int, x::T, sint::T) where T
    validate_config(cfg)
    lmax >= 0 || error("lmax must be non-negative")
    m >= 0 || error("m must be non-negative")
    m <= lmax || error("m must be <= lmax")
    abs(x) <= 1 || error("x must be in [-1, 1]")
    abs(sint - sqrt(1 - x^2)) < 1e-12 || error("sint must equal sqrt(1-x²)")
    
    n_values = lmax - m + 1
    yl = Vector{T}(undef, n_values)
    dyl = Vector{T}(undef, n_values)
    
    # Compute values
    _compute_legendre_array!(yl, lmax, m, x, cfg.norm)
    
    # Compute derivatives
    _compute_legendre_derivative_array!(dyl, yl, lmax, m, x, sint)
    
    return (yl, dyl)
end

# Internal implementation functions

"""
    _compute_legendre_array!(yl::Vector{T}, lmax::Int, m::Int, x::T, norm::SHTnsNorm) where T

Compute normalized associated Legendre polynomials using stable recurrence.
"""
function _compute_legendre_array!(yl::Vector{T}, lmax::Int, m::Int, x::T, norm::SHTnsNorm) where T
    # Handle the starting case P_m^m(x)
    sint = sqrt(1 - x^2)
    
    # Compute P_m^m using the standard formula
    pmm = one(T)
    if m > 0
        fact = one(T)
        for i in 1:m
            fact *= (2*i - 1)
        end
        pmm = (-1)^m * fact * sint^m
    end
    
    # Apply normalization factor
    pmm *= _get_normalization_factor(norm, m, m)
    yl[1] = pmm
    
    if lmax == m
        return
    end
    
    # Compute P_{m+1}^m
    pmp1m = x * (2*m + 1) * pmm
    pmp1m *= _get_normalization_factor(norm, m+1, m) / _get_normalization_factor(norm, m, m)
    yl[2] = pmp1m
    
    if lmax == m + 1
        return  
    end
    
    # Use three-term recurrence relation for l ≥ m+2
    for l in (m+2):lmax
        idx = l - m + 1
        
        # Recurrence coefficients
        a = T(2*l - 1) / T(l - m)
        b = T(l + m - 1) / T(l - m)
        
        pll = a * x * yl[idx-1] - b * yl[idx-2]
        pll *= _get_normalization_factor(norm, l, m) / _get_normalization_factor(norm, m+1, m)
        
        yl[idx] = pll
    end
end

"""
    _compute_legendre_derivative_array!(dyl::Vector{T}, yl::Vector{T}, lmax::Int, m::Int, x::T, sint::T) where T

Compute derivatives dP_l^m/dx from the values P_l^m.
"""
function _compute_legendre_derivative_array!(dyl::Vector{T}, yl::Vector{T}, lmax::Int, m::Int, x::T, sint::T) where T
    for l in m:lmax
        idx = l - m + 1
        
        if l == 0
            dyl[idx] = zero(T)
        elseif m == 0
            # For m=0, use the relation involving P_{l-1}
            if l == 1
                dyl[idx] = -sint  # Special case: dP_1/dx = -sin(θ) 
            else
                # Need P_{l-1}^0 - compute separately or use recurrence
                pl_minus_1 = _compute_single_legendre(l-1, 0, x, SHT_ORTHONORMAL)  # Use orthonormal for simplicity
                if abs(x) < 0.999  # Away from poles
                    dyl[idx] = l * (x * yl[idx] - pl_minus_1) / (x^2 - 1)
                else  # Near poles, use series expansion
                    dyl[idx] = l * (l + 1) * x * yl[idx] / 2
                end
            end
        else
            # For m ≠ 0, use the general recurrence
            if l > m
                # Need P_{l-1}^m - compute separately
                pl_minus_1_m = _compute_single_legendre(l-1, m, x, SHT_ORTHONORMAL)
                if abs(x) < 0.999
                    dyl[idx] = (l * x * yl[idx] - (l + m) * pl_minus_1_m) / (x^2 - 1)
                else
                    # Alternative form near poles
                    dyl[idx] = m * x * yl[idx] / sint^2
                end
            else  # l == m
                dyl[idx] = m * x * yl[idx] / sint^2
            end
        end
    end
end

"""
    _compute_single_legendre(l::Int, m::Int, x::T, norm::SHTnsNorm) where T

Compute a single Legendre polynomial value P_l^m(x).
"""
function _compute_single_legendre(l::Int, m::Int, x::T, norm::SHTnsNorm) where T
    m = abs(m)
    
    if l == 0 && m == 0
        return _get_normalization_factor(norm, 0, 0)
    elseif l == 1 && m == 0  
        return x * _get_normalization_factor(norm, 1, 0)
    elseif l == 1 && m == 1
        sint = sqrt(1 - x^2)
        return -sint * _get_normalization_factor(norm, 1, 1)
    end
    
    # For higher degrees, use the full computation
    yl = Vector{T}(undef, l - m + 1)
    _compute_legendre_array!(yl, l, m, x, norm)
    return yl[end]
end

"""
    _get_normalization_factor(norm::SHTnsNorm, l::Int, m::Int)

Get the normalization factor for P_l^m based on the chosen convention.
"""
function _get_normalization_factor(norm::SHTnsNorm, l::Int, m::Int)
    if norm == SHT_ORTHONORMAL
        # Orthonormal: sqrt((2l+1)(l-m)!/(4π(l+m)!))
        factor = (2*l + 1) / (4*π)
        if m > 0
            # Compute (l-m)!/(l+m)! = 1/((l-m+1)(l-m+2)...(l+m))
            for k in (l-m+1):(l+m)
                factor /= k
            end
        end
        return sqrt(factor)
    elseif norm == SHT_FOURPI  
        # 4π normalization
        factor = (2*l + 1)
        if m > 0
            for k in (l-m+1):(l+m)
                factor /= k
            end
        end
        return sqrt(factor)
    elseif norm == SHT_SCHMIDT
        # Schmidt semi-normalized
        factor = 1.0
        if m > 0
            for k in (l-m+1):(l+m)
                factor /= k  
            end
            factor *= 2  # Factor of 2 for m≠0
        end
        return sqrt(factor)
    elseif norm == SHT_REAL_NORM
        # Real normalization - no additional factors beyond the basic Legendre polynomial
        return 1.0
    else
        throw(ArgumentError("Unknown normalization: $(norm)"))
    end
end