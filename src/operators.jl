"""
Spherical Differential Operators in Spectral Space

This module implements differential operators that act on spherical harmonic coefficients.
The operators are represented as sparse matrices that couple neighboring degrees l at fixed
azimuthal order m, taking advantage of the recurrence relations for spherical harmonics.

The key operators implemented are:
- cos(θ) multiplication: couples Y_l^m to Y_{l±1}^m  
- sin(θ) ∂/∂θ derivative: couples Y_l^m to Y_{l±1}^m with different coefficients

SHTns-compatible functions provided:
- `mul_ct_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for cos(θ) operator
- `st_dt_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for sin(θ) ∂/∂θ operator  
- `SH_mul_mx(cfg, mx, Qlm, Rlm)`: apply a tridiagonal operator that couples (l,m) to l±1 at fixed m

Matrix storage format:
For each packed index `lm` (SHTns LM order, m≥0), we store two coupling coefficients:
- `mx[2*lm+1] = c_minus` for coupling to Y_{l-1}^m (lower degree neighbor)
- `mx[2*lm+2] = c_plus` for coupling to Y_{l+1}^m (higher degree neighbor)
Out-of-range neighbors (l<m or l>lmax) are automatically ignored.
"""

"""
    mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for cosθ operator: cosθ Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m.
"""
function mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]
        m = cfg.mi[lm0+1]
        # a_l^m for Y_{l-1}^m (zero if l==0)
        a = (l == 0) ? 0.0 : sqrt(max(0.0, (l^2 - m^2) / ((2l - 1) * (2l + 1))))
        # b_l^m for Y_{l+1}^m (zero if l==lmax)
        b = (l == cfg.lmax) ? 0.0 : sqrt(max(0.0, ((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3))))
        mx[2*lm0 + 1] = a
        mx[2*lm0 + 2] = b
    end
    return mx
end

"""
    st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for sinθ ∂_θ operator:
sinθ ∂_θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m.
"""
function st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]
        m = cfg.mi[lm0+1]
        a = (l == 0) ? 0.0 : sqrt(max(0.0, (l^2 - m^2) / ((2l - 1) * (2l + 1))))
        b = (l == cfg.lmax) ? 0.0 : sqrt(max(0.0, ((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3))))
        c_minus = -(l + 1) * a
        c_plus  =  l * b
        mx[2*lm0 + 1] = c_minus
        mx[2*lm0 + 2] = c_plus
    end
    return mx
end

"""
    SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply a nearest-neighbor-in-l operator represented by `mx` to `Qlm` and write to `Rlm`.
Both `Qlm` and `Rlm` are length `cfg.nlm` packed vectors (m≥0, SHTns LM order).
"""
function SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    lmax = cfg.lmax; mres = cfg.mres
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
        c_minus = mx[2*lm0 + 1]
        c_plus  = mx[2*lm0 + 2]
        acc = 0.0 + 0.0im
        # l-1 neighbor
        if l > m && l > 0
            lm_prev = LM_index(lmax, mres, l-1, m)
            acc += c_minus * Qlm[lm_prev + 1]
        end
        # l+1 neighbor
        if l < lmax
            lm_next = LM_index(lmax, mres, l+1, m)
            acc += c_plus * Qlm[lm_next + 1]
        end
        Rlm[lm0 + 1] = acc
    end
    return Rlm
end

