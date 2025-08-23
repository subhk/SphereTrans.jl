"""
Rotations of spherical harmonic expansions.

Currently supports fast rotation around the Z-axis by angle `alpha` in radians.
"""

"""
    SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Z-axis by angle `alpha`.
Input and output are packed `Qlm` vectors (LM order, m ≥ 0). In-place supported if `Rlm === Qlm`.
"""
function SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    lmax = cfg.lmax; mres = cfg.mres
    @inbounds for m in 0:cfg.mmax
        (m % mres == 0) || continue
        phase = cis(m * alpha)
        for l in m:lmax
            lm = LM_index(lmax, mres, l, m) + 1
            Rlm[lm] = Qlm[lm] * phase
        end
    end
    return Rlm
end

