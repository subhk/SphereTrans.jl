"""
Normalization and Condon–Shortley phase conversions for spherical harmonic
coefficients. Internally, transforms use orthonormal Y_lm with CS phase.
"""

"""
    norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol) -> Float64

Return k such that Y_to = k * Y_orthonormal.
Supported `to`: :orthonormal, :fourpi, :schmidt
"""
function norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol)
    if to === :orthonormal
        return 1.0
    elseif to === :fourpi
        # Common geodesy convention: overall sqrt(4π)
        return sqrt(4π)
    elseif to === :schmidt
        # Schmidt semi-normalized: sqrt(4π/(2l+1)) relative to orthonormal
        return sqrt(4π / (2l + 1))
    else
        throw(ArgumentError("Unsupported normalization: $to"))
    end
end

"""
    cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool) -> Float64

Return α such that Y_to = α * Y_from when toggling Condon–Shortley phase.
If switching from CS to no-CS: α = (-1)^m; from no-CS to CS: α = (-1)^m (inverse).
"""
function cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool)
    if cs_from == cs_to
        return 1.0
    else
        # Toggle CS: multiply by (-1)^m on the basis; coefficients scale by inverse
        # Caller must invert when mapping coefficients.
        return (-1.0)^m
    end
end

"""
    convert_alm_norm!(dest, src, cfg; to_internal::Bool=false)

Convert coefficient matrix `src` between cfg's normalization/phase and internal
orthonormal+CS. If `to_internal=true`, maps from cfg to internal. Otherwise maps
from internal to cfg. Writes into `dest` which must match `src` size.
"""
function convert_alm_norm!(dest::AbstractMatrix, src::AbstractMatrix, cfg; to_internal::Bool=false)
    size(dest) == size(src) || throw(DimensionMismatch("dest/src dims mismatch"))
    lmax, mmax = cfg.lmax, cfg.mmax
    # From orthonormal+CS to cfg: alm_cfg = alm_int / k_norm / α_cs
    # To internal: alm_int = alm_cfg * k_norm * α_cs
    for m in 0:mmax
        α = cs_phase_factor(m, true, cfg.cs_phase)  # Y_cfg = α * Y_int if toggling CS
        for l in m:lmax
            k = norm_scale_from_orthonormal(l, m, cfg.norm)
            s = to_internal ? (k * α) : (1.0 / (k * α))
            dest[l+1, m+1] = s * src[l+1, m+1]
        end
    end
    return dest
end

