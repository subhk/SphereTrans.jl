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
    tolerance = 10 * eps(T)
    max_iter = 100
    
    for i in 1:m
        # Initial guess using asymptotic formula
        if i == 1
            # For the largest root, use asymptotic expansion
            x = T(1) - T(2.5) / (n + 1)^2
        elseif i == 2
            # Second root
            x = nodes[1] - T(4.1) / n
        elseif i == 3
            # Third root  
            x = T(1.5) * nodes[2] - T(0.5) * nodes[1]
        else
            # Use recurrence for remaining roots
            x = T(3) * nodes[i-1] - T(3) * nodes[i-2] + nodes[i-3]
        end
        
        # Newton-Raphson iteration to refine the root
        final_derivative = T(1)  # Initialize derivative variable
        for iter in 1:max_iter
            # Compute Legendre polynomial P_n(x) and its derivative P_n'(x)
            # using the recurrence relation
            p0, p1 = T(1), x
            dp0, dp1 = T(0), T(1)
            
            for j in 2:n
                # Recurrence: (j)P_j = (2j-1)x*P_{j-1} - (j-1)P_{j-2}
                p2 = ((2j - 1) * x * p1 - (j - 1) * p0) / j
                # Derivative: j*P_j' = (2j-1)(P_{j-1} + x*P_{j-1}') - (j-1)P_{j-2}'
                dp2 = ((2j - 1) * (p1 + x * dp1) - (j - 1) * dp0) / j
                
                p0, p1 = p1, p2
                dp0, dp1 = dp1, dp2
            end
            
            final_derivative = dp1  # Store final derivative
            
            # Newton-Raphson update
            dx = -p1 / dp1
            x += dx
            
            # Check convergence
            abs(dx) < tolerance && break
            
            if iter == max_iter
                @warn "Gauss-Legendre iteration did not converge for i=$i"
            end
        end
        
        # Store the node
        nodes[i] = x
        
        # Compute weight: w_i = 2 / [(1 - x_i^2) * (P_n'(x_i))^2]
        weights[i] = T(2) / ((T(1) - x^2) * final_derivative^2)
        
        # Use symmetry for the other half
        if i <= n ÷ 2
            nodes[n + 1 - i] = -x
            weights[n + 1 - i] = weights[i]
        end
    end
    
    # Ensure exact symmetry and normalization
    for i in 1:n÷2
        nodes[n + 1 - i] = -nodes[i]
        weights[n + 1 - i] = weights[i]
    end
    
    # Verify that weights sum to 2 (integral of 1 from -1 to 1)
    weight_sum = sum(weights)
    if abs(weight_sum - 2) > 1e-12
        @warn "Gauss-Legendre weights sum to $weight_sum instead of 2"
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
    sint = sqrt(1 - x^2)  # sin(θ) where x = cos(θ)
    
    # Total number of coefficients
    nlm = nlm_calc(lmax, lmax, 1)  # Assume full triangular truncation
    plm = zeros(T, nlm)
    
    # P_0^0 = 1 (properly normalized)
    idx = lmidx(0, 0) + 1
    if norm == SHT_ORTHONORMAL
        plm[idx] = T(1) / sqrt(T(4π))
    elseif norm == SHT_FOURPI
        plm[idx] = T(1)
    else
        plm[idx] = T(1)
    end
    
    if lmax == 0
        return plm
    end
    
    # P_1^0 = x (with normalization)
    idx = lmidx(1, 0) + 1
    if norm == SHT_ORTHONORMAL
        plm[idx] = x * sqrt(T(3)) / sqrt(T(4π))
    elseif norm == SHT_FOURPI  
        plm[idx] = x
    else
        plm[idx] = x
    end
    
    # P_1^1 = -sin(θ) (with normalization)
    if sint > 0  # Avoid division by zero at poles
        idx = lmidx(1, 1) + 1
        if norm == SHT_ORTHONORMAL
            plm[idx] = -sint * sqrt(T(3) / T(2)) / sqrt(T(4π))
        elseif norm == SHT_FOURPI
            plm[idx] = -sint
        else
            plm[idx] = -sint
        end
    end
    
    # Compute remaining values using recurrence relations
    for l in 2:lmax
        # P_l^0 from P_{l-1}^0 and P_{l-2}^0
        idx_l0 = lmidx(l, 0) + 1
        idx_l1_0 = lmidx(l-1, 0) + 1
        idx_l2_0 = lmidx(l-2, 0) + 1
        
        if norm == SHT_ORTHONORMAL
            # Orthonormal recurrence
            plm[idx_l0] = sqrt((2l+1)*(2l-1)) / l * x * plm[idx_l1_0] - 
                         sqrt((2l+1)*(l-1)*(l-1)) / (l * sqrt((2l-3))) * plm[idx_l2_0]
        else
            # Standard recurrence  
            plm[idx_l0] = ((2l - 1) * x * plm[idx_l1_0] - (l - 1) * plm[idx_l2_0]) / l
        end
        
        # P_l^l from P_{l-1}^{l-1} (diagonal relation)
        if sint > 0
            idx_ll = lmidx(l, l) + 1
            idx_l1_l1 = lmidx(l-1, l-1) + 1
            
            if norm == SHT_ORTHONORMAL
                plm[idx_ll] = -sqrt((2l+1) / (2l)) * sint * plm[idx_l1_l1]
            else
                plm[idx_ll] = -(2l - 1) * sint * plm[idx_l1_l1]
            end
        end
        
        # P_l^m for 0 < m < l using recurrence
        for m in 1:(l-1)
            if l-2 >= 0 && l-2 >= abs(m)  # Ensure valid indices
                idx_lm = lmidx(l, m) + 1
                idx_l1_m = lmidx(l-1, m) + 1
                idx_l2_m = lmidx(l-2, m) + 1
                
                if norm == SHT_ORTHONORMAL
                    # Orthonormal recurrence for off-diagonal terms
                    c1 = sqrt((2l+1)*(2l-1) / ((l-m)*(l+m)))
                    c2 = sqrt((2l+1)*(l-1-m)*(l-1+m) / ((2l-3)*(l-m)*(l+m)))
                    plm[idx_lm] = c1 * x * plm[idx_l1_m] - c2 * plm[idx_l2_m]
                else
                    # Standard recurrence
                    plm[idx_lm] = ((2l - 1) * x * plm[idx_l1_m] - (l + m - 1) * plm[idx_l2_m]) / (l - m)
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
function nlm_calc(lmax::Int, mmax::Int, mres::Int)
    lmax >= 0 || error("lmax must be non-negative")
    mmax >= 0 || error("mmax must be non-negative")
    mres >= 1 || error("mres must be positive")
    
    nlm = 0
    for l in 0:lmax
        max_m = min(l, mmax)
        if mres == 1
            nlm += max_m + 1  # m = 0, 1, ..., max_m
        else
            # Count m = 0, mres, 2*mres, ... up to max_m
            nlm += 1  # m = 0
            m = mres
            while m <= max_m
                nlm += 1
                m += mres
            end
        end
    end
    
    return nlm
end