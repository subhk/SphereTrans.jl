"""
    Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}

Fill `P` with associated Legendre values `P_l^m(x)` for `l = 0..lmax` with zeros for `l < m`.
Uses the Ferrers definition with the Condon–Shortley phase included.
"""
function Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    @inbounds fill!(P, zero(T))
    m < 0 && throw(ArgumentError("m must be ≥ 0"))
    lmax >= m || return P
    # Base cases
    if m == 0
        P[1] = one(T)                 # P_0^0
        if lmax >= 1
            P[2] = x                  # P_1^0
        end
        for l in 2:lmax
            # P_l^0(x) = ((2l-1)x P_{l-1}^0 - (l-1) P_{l-2}^0)/l
            P[l+1] = ((2l - 1) * x * P[l] - (l - 1) * P[l-1]) / l
        end
        return P
    end

    # P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    pmm = one(T)
    sx2 = max(zero(T), 1 - x*x)  # guard minor negative due to fp
    fact = one(T)
    for k in 1:m
        pmm *= -fact * sqrt(sx2)
        fact += 2
    end
    P[m+1] = pmm
    if lmax == m
        return P
    end
    # P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    P[m+2] = x * (2m + 1) * pmm
    for l in (m+2):lmax
        # P_l^m(x) = ((2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m)/(l-m)
        P[l+1] = ((2l - 1) * x * P[l] - (l + m - 1) * P[l-1]) / (l - m)
    end
    return P
end

"""
    Nlm_table(lmax::Int, mmax::Int)

Precompute normalization factors `N_{l,m} = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)` for 0≤m≤mmax, m≤l≤lmax.
Returns a matrix `N` sized (lmax+1, mmax+1) where indices are `(l+1, m+1)`.
"""
function Nlm_table(lmax::Int, mmax::Int)
    N = Matrix{Float64}(undef, lmax + 1, mmax + 1)
    for m in 0:mmax
        for l in 0:lmax
            if l < m
                N[l+1, m+1] = 0.0
            else
                # Use loggamma to avoid overflow in factorials
                lr = 0.5 * (log(2l + 1.0) - log(4π)) + 0.5 * (loggamma(l - m + 1) - loggamma(l + m + 1))
                N[l+1, m+1] = exp(lr)
            end
        end
    end
    return N
end

