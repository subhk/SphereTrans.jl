"""
Dense (non-distributed) helpers that are used by distributed APIs and examples.
These do not depend on MPI/Pencil packages and live in src/ to keep extensions lean.
"""

"""
    dist_apply_laplacian!(cfg, Alm::AbstractMatrix)

In-place multiply by -l(l+1) for dense (l×m) coefficients.
"""
function dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] *= -(l*(l+1))
    end
    return Alm
end

"""
    dist_SH_Zrotate(cfg, Alm::AbstractMatrix, alpha::Real, Rlm::AbstractMatrix)

Z-rotation by alpha (radians) on dense (l×m) coefficients; Rlm = e^{imα} Alm.
"""
function dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix, alpha::Real, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    @inbounds for m in 0:mmax
        phase = cis(m*alpha)
        for l in m:lmax
            Rlm[l+1, m+1] = phase * Alm[l+1, m+1]
        end
    end
    return Rlm
end

"""
    dist_SH_Yrotate(cfg, Alm::AbstractMatrix, beta::Real, Rlm::AbstractMatrix)

Gather/apply/unpack rotation on dense (l×m): packs to LM vector, applies SH_Yrotate, unpacks.
Useful for validation and small problems; distributed variants should prefer per-l Allgatherv.
"""
function dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix, beta::Real, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    Q = Vector{ComplexF64}(undef, cfg.nlm)
    @inbounds for m in 0:mmax, l in m:lmax
        idx = SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1
        Q[idx] = Alm[l+1, m+1]
    end
    R = similar(Q)
    SHTnsKit.SH_Yrotate(cfg, Q, beta, R)
    @inbounds for m in 0:mmax, l in m:lmax
        idx = SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1
        Rlm[l+1, m+1] = R[idx]
    end
    return Rlm
end

"""
    dist_SH_mul_mx!(cfg, mx, Alm, Rlm)

Apply 3-diagonal l±1 operator to dense (l×m) Alm into Rlm. No communication.
`mx` is 2*nlm packed coefficients as in mul_ct_matrix/st_dt_matrix.
"""
function dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Alm::AbstractMatrix, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    fill!(Rlm, 0)
    @inbounds for m in 0:mmax, l in m:lmax
        idx = SHTnsKit.LM_index(lmax, cfg.mres, l, m)
        c_minus = mx[2*idx + 1]
        c_plus  = mx[2*idx + 2]
        acc = 0.0 + 0.0im
        if l > m && l > 0
            acc += c_minus * Alm[l, m+1]
        end
        if l < lmax
            acc += c_plus * Alm[l+2, m+1]
        end
        Rlm[l+1, m+1] = acc
    end
    return Rlm
end

