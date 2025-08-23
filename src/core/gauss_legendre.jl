"""
Gauss-Legendre quadrature implementation for spherical harmonic transforms.
This module provides efficient computation of Gaussian nodes and weights
required for high-precision spherical harmonic transforms.
"""

"""
    compute_gauss_legendre_nodes_weights(n::Integer, T::Type=Float64) -> (Vector{T}, Vector{T})

Compute the nodes (roots) and weights for Gauss-Legendre quadrature of order n.
Returns (nodes, weights) where nodes are in [-1, 1] and weights sum to 2.

This uses the asymptotic formulas for initial guesses followed by Newton-Raphson
iteration for high precision, following the algorithms in SHTns.

# Arguments
- `n`: Number of quadrature points
- `T`: Floating point precision type (default Float64)

# Returns
- `nodes`: Vector of n Gauss-Legendre nodes (roots of P_n(x))
- `weights`: Vector of corresponding quadrature weights
"""
function compute_gauss_legendre_nodes_weights(n::Integer, T::Type=Float64)
    n > 0 || error("Number of quadrature points must be positive")
    
    nodes = Vector{T}(undef, n)
    weights = Vector{T}(undef, n)
    
    # Use symmetry: only compute half the points
    m = (n + 1) ÷ 2
    
    # Tolerances for Newton-Raphson iteration
    tolerance = eps(T) * 100
    max_iter = 100
    
    for i in 1:m
        # Better initial guess using Chebyshev nodes
        theta = T(π) * (i - 0.25) / (n + 0.5)
        x = cos(theta)
        
        # Newton-Raphson iteration to refine the root
        local final_derivative
        for iter in 1:max_iter
            # Compute Legendre polynomial P_n(x) and its derivative P_n'(x)
            # using the stable three-term recurrence relation
            p0, p1 = T(1), x
            dp0, dp1 = T(0), T(1)
            
            @inbounds for j in 2:n
                # Standard recurrence for Legendre polynomials with optimized arithmetic
                # P_j(x) = ((2j-1)*x*P_{j-1}(x) - (j-1)*P_{j-2}(x)) / j
                two_j_minus_1 = 2*j - 1
                j_minus_1 = j - 1
                inv_j = one(T) / j
                
                p2 = (two_j_minus_1 * x * p1 - j_minus_1 * p0) * inv_j
                
                # Derivative recurrence with precomputed factors
                # j*P_j'(x) = (2j-1)*(P_{j-1}(x) + x*P_{j-1}'(x)) - (j-1)*P_{j-2}'(x)
                dp2 = (two_j_minus_1 * (p1 + x * dp1) - j_minus_1 * dp0) * inv_j
                
                p0, p1 = p1, p2
                dp0, dp1 = dp1, dp2
            end
            
            final_derivative = dp1
            
            # Newton-Raphson update
            dx = -p1 / dp1
            x += dx
            
            # Check convergence
            abs(dx) < tolerance && break
            
            if iter == max_iter
                error("Gauss-Legendre iteration did not converge for point $i")
            end
        end
        
        # Store the node
        nodes[i] = x
        
        # Compute weight using the correct formula:
        # w_i = 2 / ((1 - x_i^2) * [P_n'(x_i)]^2)
        weights[i] = T(2) / ((T(1) - x^2) * final_derivative^2)
    end
    
    # Fill in the negative nodes and weights using symmetry with SIMD
    @inbounds @simd for i in 1:m
        if n + 1 - i != i  # Avoid overwriting middle node for odd n
            nodes[n + 1 - i] = -nodes[i]
            weights[n + 1 - i] = weights[i]
        end
    end
    
    # Final verification and potential normalization
    weight_sum = sum(weights)
    expected_sum = T(2)
    
    if abs(weight_sum - expected_sum) > sqrt(eps(T))
        # If the weights don't sum to 2, normalize them
        weights .*= expected_sum / weight_sum
        @warn "Normalized Gauss-Legendre weights (sum was $weight_sum)"
    end
    
    return nodes, weights
end

"""
    compute_associated_legendre(lmax::Int, x::T, norm::SHTnsNorm=SHT_ORTHONORMAL) where T

Compute associated Legendre polynomials P_l^m(x) for all l ≤ lmax and |m| ≤ l
at a given point x ∈ [-1, 1].

Uses stable recurrence relations to compute all values efficiently.
Different normalizations are supported.

# Arguments
- `lmax`: Maximum degree
- `x`: Evaluation point in [-1, 1]  
- `norm`: Normalization convention

# Returns
- `plm`: Matrix where plm[lmidx(l,m)] contains P_l^m(x)
"""
function compute_associated_legendre(lmax::Int, x::T, norm::SHTnsNorm=SHT_ORTHONORMAL) where T
    abs(x) <= 1 || error("x must be in [-1, 1]")
    
    # Precompute commonly used values
    sint = sqrt(max(zero(T), 1 - x^2))  # sin(θ) where x = cos(θ), ensure non-negative
    
    # Total number of coefficients
    nlm = nlm_calc(lmax, lmax, 1)  # Assume full triangular truncation
    plm = zeros(T, nlm)
    
    # Starting values for associated Legendre polynomials
    # P_0^0 = 1 (normalized according to convention)
    # For m=0, l=0 case - it's always the first coefficient (index 1)
    idx = 1
    if norm == SHT_ORTHONORMAL
        plm[idx] = T(1) / sqrt(T(4π))  # Y_0^0 normalization
    elseif norm == SHT_FOURPI
        plm[idx] = T(1) / T(2) / sqrt(T(π))  # 4π normalization
    elseif norm == SHT_SCHMIDT
        plm[idx] = T(1)  # Schmidt normalization
    else
        plm[idx] = T(1)  # Unnormalized
    end
    
    if lmax == 0
        return plm
    end
    
    # Compute P_l^m using stable recurrence relations
    for l in 1:lmax
        # First compute P_l^l using the diagonal recurrence
        if l == 1
            # P_1^1 = -sqrt(1-x^2) = -sin(θ)
            idx_11 = lmidx(1, 1) + 1
            val = -sint
            if norm == SHT_ORTHONORMAL
                val *= sqrt(T(3)) / sqrt(T(4π))  # Y_1^1 normalization  
            elseif norm == SHT_FOURPI
                val *= sqrt(T(3)) / T(2) / sqrt(T(π))  # 4π normalization
            elseif norm == SHT_SCHMIDT
                val *= sqrt(T(3))  # Schmidt normalization
            end
            plm[idx_11] = val
        else
            # P_l^l = -(2l-1) * sin(θ) * P_{l-1}^{l-1} (unnormalized)
            idx_ll = lmidx(l, l) + 1
            idx_l1_l1 = lmidx(l-1, l-1) + 1
            
            if norm == SHT_ORTHONORMAL
                # For orthonormal spherical harmonics
                factor = -sqrt((2*l + 1) / (2*l)) * sint
            elseif norm == SHT_FOURPI
                factor = -(2*l - 1) * sint
            elseif norm == SHT_SCHMIDT
                factor = -(2*l - 1) * sint / sqrt(T(2*l))
            else
                factor = -(2*l - 1) * sint
            end
            plm[idx_ll] = factor * plm[idx_l1_l1]
        end
        
        # Now compute P_l^{l-1}, P_l^{l-2}, ..., P_l^0 using downward recurrence
        for m in (l-1):-1:0
            idx_lm = lmidx(l, m) + 1
            
            if m == l - 1
                # P_l^{l-1} = x * (2l-1) * P_{l-1}^{l-1} (unnormalized)
                if l > 1
                    idx_l1_l1 = lmidx(l-1, l-1) + 1
                    if norm == SHT_ORTHONORMAL
                        factor = x * sqrt(T(2*l + 1)) / sqrt(T(2*l - 1))
                    elseif norm == SHT_FOURPI  
                        factor = x * (2*l - 1)
                    elseif norm == SHT_SCHMIDT
                        factor = x * sqrt(T(2*l - 1))
                    else
                        factor = x * (2*l - 1)
                    end
                    plm[idx_lm] = factor * plm[idx_l1_l1]
                else
                    # P_1^0 = x
                    val = x
                    if norm == SHT_ORTHONORMAL
                        val *= sqrt(T(3)) / sqrt(T(4π))
                    elseif norm == SHT_FOURPI
                        val *= sqrt(T(3)) / T(2) / sqrt(T(π))
                    elseif norm == SHT_SCHMIDT
                        val *= sqrt(T(3))
                    end
                    plm[idx_lm] = val
                end
            else
                # General recurrence: (l-m)*P_l^m = (2l-1)*x*P_{l-1}^m - (l+m-1)*P_{l-2}^m
                if l >= 2 && l-2 >= abs(m)
                    idx_l1_m = lmidx(l-1, m) + 1
                    idx_l2_m = lmidx(l-2, m) + 1
                    
                    if norm == SHT_ORTHONORMAL
                        # Orthonormal recurrence coefficients
                        c1 = sqrt((2*l + 1) * (2*l - 1) / ((l - m) * (l + m)))
                        c2 = sqrt((2*l + 1) * (l - 1 - m) * (l - 1 + m) / ((2*l - 3) * (l - m) * (l + m)))
                        plm[idx_lm] = c1 * x * plm[idx_l1_m] - c2 * plm[idx_l2_m]
                    else
                        # Standard recurrence
                        c1 = T(2*l - 1) / (l - m)
                        c2 = T(l + m - 1) / (l - m)
                        plm[idx_lm] = c1 * x * plm[idx_l1_m] - c2 * plm[idx_l2_m]
                    end
                end
            end
        end
    end
    
    return plm
end

"""
    lmidx(l::Int, m::Int) -> Int

Convert (l,m) spherical harmonic indices to linear array index (0-based).
This follows the SHTns convention for indexing spherical harmonic coefficients.

For real transforms, only m ≥ 0 are stored. Complex transforms store both ±m.
"""
function lmidx(l::Int, m::Int)
    l >= 0 || error("l must be non-negative")
    abs(m) <= l || error("|m| must be <= l")
    
    # Standard triangular storage: sum over l' < l, then add m
    # For l'=0: 1 coefficient, l'=1: 2 coefficients, ..., l'=l-1: l coefficients
    # Total: 1 + 2 + ... + l = l(l+1)/2
    # Then add m for the current l
    return l * (l + 1) ÷ 2 + m
end

"""
    nlm_calc(lmax::Int, mmax::Int, mres::Int) -> Int

Calculate the number of spherical harmonic coefficients for given truncation.
"""
function nlm_calc(lmax::Int, mmax::Int, mres::Int)::Int
    lmax >= 0 || error("lmax must be non-negative")
    mmax >= 0 || error("mmax must be non-negative")
    mres >= 1 || error("mres must be positive")
    
    if mres == 1
        # Optimized formula for mres=1 case
        if mmax >= lmax
            # Full triangular truncation: Σ(l+1) for l=0..lmax = (lmax+1)(lmax+2)/2
            return (lmax + 1) * (lmax + 2) ÷ 2
        else
            # mmax < lmax case - use optimized loop
            nlm = 0
            @inbounds for l in 0:lmax
                nlm += min(l, mmax) + 1
            end
            return nlm
        end
    else
        # General case with mres > 1 - optimized counting
        nlm = lmax + 1  # m=0 contribution for all l
        @inbounds for l in 1:lmax
            max_m = min(l, mmax)
            # Count multiples of mres up to max_m
            nlm += max_m ÷ mres
        end
        return nlm
    end
end