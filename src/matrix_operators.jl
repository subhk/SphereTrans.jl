"""
Matrix operators for spherical harmonic transforms.
These implement spectral operators like multiplication by cos(θ) and sin(θ)d/dθ.
"""

"""
    mul_ct_matrix(cfg::SHTnsConfig{T}) where T

Compute the matrix required to multiply a spherical harmonic representation by cos(θ).

This matrix couples spherical harmonics of degrees l-1, l, and l+1 due to the
recurrence relation for cos(θ) * Y_l^m.

# Returns
- Matrix of size (nlm, nlm) for applying cos(θ) multiplication

Equivalent to the C library function `mul_ct_matrix()`.
"""
function mul_ct_matrix(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    n = cfg.nlm
    matrix = zeros(T, n, n)
    
    for (idx_out, (l_out, m_out)) in enumerate(cfg.lm_indices)
        for (idx_in, (l_in, m_in)) in enumerate(cfg.lm_indices)
            
            # cos(θ) couples only same m, and l differing by ±1
            if m_out == m_in
                coeff = zero(T)
                
                # Contribution from l_in = l_out + 1 term
                if l_in == l_out + 1 && l_in <= cfg.lmax
                    coeff += _costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                end
                
                # Contribution from l_in = l_out - 1 term  
                if l_in == l_out - 1 && l_in >= 0
                    coeff += _costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                end
                
                matrix[idx_out, idx_in] = coeff
            end
        end
    end
    
    return matrix
end

"""
    st_dt_matrix(cfg::SHTnsConfig{T}) where T

Compute the matrix required to apply sin(θ) * d/dθ to a spherical harmonic representation.

This operator is commonly used in fluid dynamics and is related to the curl operator.

# Returns
- Matrix of size (nlm, nlm) for applying sin(θ) * d/dθ

Equivalent to the C library function `st_dt_matrix()`.
"""
function st_dt_matrix(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    n = cfg.nlm
    matrix = zeros(T, n, n)
    
    for (idx_out, (l_out, m_out)) in enumerate(cfg.lm_indices)
        for (idx_in, (l_in, m_in)) in enumerate(cfg.lm_indices)
            
            # sin(θ) * d/dθ couples only same m, and l differing by ±1
            if m_out == m_in && l_in > 0
                coeff = zero(T)
                
                # Contribution from l_in = l_out + 1 term
                if l_in == l_out + 1 && l_in <= cfg.lmax
                    coeff += _sintdtheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                end
                
                # Contribution from l_in = l_out - 1 term
                if l_in == l_out - 1 && l_in >= 0  
                    coeff += _sintdtheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                end
                
                matrix[idx_out, idx_in] = coeff
            end
        end
    end
    
    return matrix
end

"""
    sh_mul_mx(cfg::SHTnsConfig{T}, matrix::AbstractMatrix{T}, 
              qlm_in::AbstractVector{Complex{T}}, qlm_out::AbstractVector{Complex{T}}) where T

Apply a matrix involving l+1 and l-1 coupling to a spherical harmonic representation.
This is a general matrix-vector multiplication for spectral space operators.

# Arguments
- `cfg`: SHTns configuration
- `matrix`: Coupling matrix (size nlm × nlm) 
- `qlm_in`: Input SH coefficients
- `qlm_out`: Output SH coefficients (pre-allocated, must be different from qlm_in)

Equivalent to the C library function `SH_mul_mx()`.
"""
function sh_mul_mx(cfg::SHTnsConfig{T}, matrix::AbstractMatrix{T},
                   qlm_in::AbstractVector{Complex{T}}, qlm_out::AbstractVector{Complex{T}}) where T
    validate_config(cfg)
    size(matrix) == (cfg.nlm, cfg.nlm) || error("Matrix size must be (nlm, nlm)")
    length(qlm_in) == cfg.nlm || error("qlm_in length must equal nlm")
    length(qlm_out) == cfg.nlm || error("qlm_out length must equal nlm") 
    qlm_in !== qlm_out || error("Input and output arrays must be different")
    
    # Apply matrix to real and imaginary parts separately
    for i in 1:cfg.nlm
        qlm_out[i] = zero(Complex{T})
        for j in 1:cfg.nlm
            qlm_out[i] += matrix[i, j] * qlm_in[j]
        end
    end
    
    return qlm_out
end

"""
    apply_costheta_operator(cfg::SHTnsConfig{T}, qlm_in::AbstractVector{Complex{T}}) where T

Apply the cos(θ) multiplication operator to spherical harmonic coefficients.

# Arguments  
- `cfg`: SHTns configuration
- `qlm_in`: Input SH coefficients

# Returns
- Output SH coefficients representing cos(θ) * f(θ,φ)

This is a convenience function that computes and applies the cos(θ) matrix.
"""
function apply_costheta_operator(cfg::SHTnsConfig{T}, qlm_in::AbstractVector{Complex{T}}) where T
    matrix = mul_ct_matrix(cfg)
    qlm_out = similar(qlm_in)
    return sh_mul_mx(cfg, matrix, qlm_in, qlm_out)
end

"""  
    apply_sintdtheta_operator(cfg::SHTnsConfig{T}, qlm_in::AbstractVector{Complex{T}}) where T

Apply the sin(θ) * d/dθ operator to spherical harmonic coefficients.

# Arguments
- `cfg`: SHTns configuration  
- `qlm_in`: Input SH coefficients

# Returns
- Output SH coefficients representing sin(θ) * ∂f/∂θ

This is a convenience function that computes and applies the sin(θ) * d/dθ matrix.
"""
function apply_sintdtheta_operator(cfg::SHTnsConfig{T}, qlm_in::AbstractVector{Complex{T}}) where T
    matrix = st_dt_matrix(cfg)
    qlm_out = similar(qlm_in)  
    return sh_mul_mx(cfg, matrix, qlm_in, qlm_out)
end

# Internal helper functions

"""
    _costheta_coupling_coefficient(cfg::SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T

Compute the coupling coefficient for cos(θ) multiplication between degrees l_out and l_in.
"""
function _costheta_coupling_coefficient(cfg::SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T
    m_abs = abs(m)
    
    # cos(θ) * Y_l^m = α(l,m) * Y_{l+1}^m + β(l,m) * Y_{l-1}^m
    # where α and β depend on the normalization
    
    if l_in == l_out + 1
        # Coefficient for Y_{l+1}^m term
        if cfg.norm == SHT_ORTHONORMAL
            num = (l_out + 1)^2 - m_abs^2
            den = (2*l_out + 1) * (2*l_out + 3)
            return sqrt(T(num) / T(den))
        else
            # For other normalizations, scale appropriately
            factor = _get_normalization_ratio(cfg.norm, l_out, l_in, m_abs)
            num = (l_out + 1)^2 - m_abs^2
            den = (2*l_out + 1) * (2*l_out + 3)
            return factor * sqrt(T(num) / T(den))
        end
        
    elseif l_in == l_out - 1 && l_out > 0
        # Coefficient for Y_{l-1}^m term
        if cfg.norm == SHT_ORTHONORMAL
            num = l_out^2 - m_abs^2
            den = (2*l_out - 1) * (2*l_out + 1)
            return sqrt(T(num) / T(den))
        else
            factor = _get_normalization_ratio(cfg.norm, l_out, l_in, m_abs)
            num = l_out^2 - m_abs^2  
            den = (2*l_out - 1) * (2*l_out + 1)
            return factor * sqrt(T(num) / T(den))
        end
    else
        return zero(T)
    end
end

"""
    _sintdtheta_coupling_coefficient(cfg::SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T

Compute the coupling coefficient for sin(θ) * d/dθ operator.
"""  
function _sintdtheta_coupling_coefficient(cfg::SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T
    m_abs = abs(m)
    
    # sin(θ) * d/dθ Y_l^m = γ(l,m) * Y_{l+1}^m + δ(l,m) * Y_{l-1}^m
    
    if l_in == l_out + 1
        # Coefficient for Y_{l+1}^m term  
        if cfg.norm == SHT_ORTHONORMAL
            num = (l_out + 1 + m_abs) * (l_out + 1 - m_abs)
            den = (2*l_out + 1) * (2*l_out + 3)
            return sqrt(T(num) / T(den))
        else
            factor = _get_normalization_ratio(cfg.norm, l_out, l_in, m_abs)
            num = (l_out + 1 + m_abs) * (l_out + 1 - m_abs)
            den = (2*l_out + 1) * (2*l_out + 3)
            return factor * sqrt(T(num) / T(den))
        end
        
    elseif l_in == l_out - 1 && l_out > 0
        # Coefficient for Y_{l-1}^m term
        if cfg.norm == SHT_ORTHONORMAL
            num = (l_out + m_abs) * (l_out - m_abs)
            den = (2*l_out - 1) * (2*l_out + 1)
            return -sqrt(T(num) / T(den))  # Note minus sign
        else
            factor = _get_normalization_ratio(cfg.norm, l_out, l_in, m_abs)
            num = (l_out + m_abs) * (l_out - m_abs)
            den = (2*l_out - 1) * (2*l_out + 1)
            return -factor * sqrt(T(num) / T(den))
        end
    else
        return zero(T)
    end
end

"""
    _get_normalization_ratio(norm::SHTnsNorm, l_out::Int, l_in::Int, m::Int)

Get the ratio of normalization factors between different degrees.
"""
function _get_normalization_ratio(norm::SHTnsNorm, l_out::Int, l_in::Int, m::Int)
    if norm == SHT_ORTHONORMAL
        return 1.0  # Already handled in orthonormal case
    elseif norm == SHT_FOURPI
        # 4π normalization
        return sqrt(T(2*l_out + 1) / T(2*l_in + 1))
    elseif norm == SHT_SCHMIDT
        # Schmidt normalization  
        return 1.0  # Schmidt maintains the same ratios as orthonormal
    elseif norm == SHT_REAL_NORM
        # Real normalization
        return sqrt(T(2*l_out + 1) / T(2*l_in + 1))
    else
        return 1.0
    end
end

"""
    laplacian_matrix(cfg::SHTnsConfig{T}) where T

Compute the matrix for the spherical Laplacian operator ∇².

For spherical harmonics Y_l^m, the Laplacian is:
∇²Y_l^m = -l(l+1) Y_l^m

This returns a diagonal matrix with eigenvalues -l(l+1).
"""
function laplacian_matrix(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    n = cfg.nlm
    matrix = zeros(T, n, n)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        matrix[idx, idx] = -T(l * (l + 1))
    end
    
    return matrix
end

"""
    apply_laplacian(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}) where T

Apply the spherical Laplacian operator to SH coefficients.

# Arguments
- `cfg`: SHTns configuration
- `qlm`: Input SH coefficients

# Returns  
- SH coefficients of ∇²f
"""
function apply_laplacian(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}) where T
    validate_config(cfg)
    length(qlm) == cfg.nlm || error("qlm length must equal nlm")
    
    result = similar(qlm)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        result[idx] = -T(l * (l + 1)) * qlm[idx]
    end
    
    return result
end