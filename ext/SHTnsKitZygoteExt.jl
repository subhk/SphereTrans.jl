module SHTnsKitZygoteExt

using Zygote
using SHTnsKit

"""
    zgrad_scalar_energy(cfg, f) -> ∂E/∂f

Zygote gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.
Returns an array the same size as `f`.
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, x))
    return Zygote.gradient(loss, f)[1]
end

##########
# Generic distributed/array wrappers (avoid hard dependency on PencilArrays)
##########

function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::AbstractArray)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, x))
    return Zygote.gradient(loss, fθφ)[1]
end

function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::AbstractArray, Vpθφ::AbstractArray)
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Xt, Xp)
        SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    g = Zygote.gradient(loss, Vtθφ, Vpθφ)
    return g[1], g[2]
end

"""
    zgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Xt, Xp)
        SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    g = Zygote.gradient(loss, Vt, Vp)
    return g[1], g[2]
end

"""
    zgrad_enstrophy_Tlm(cfg, Tlm) -> ∂Z/∂Tlm

Zygote gradient of enstrophy with respect to toroidal spectrum Tlm.
"""
function SHTnsKit.zgrad_enstrophy_Tlm(cfg::SHTnsKit.SHTConfig, Tlm::AbstractMatrix)
    loss(X) = SHTnsKit.enstrophy(cfg, X)
    return Zygote.gradient(loss, Tlm)[1]
end

"""
    zgrad_rotation_angles_real(cfg, Qlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_real(cfg, Qlm; α,β,γ) using Zygote.
Assumes mres == 1.
"""
function SHTnsKit.zgrad_rotation_angles_real(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector, α::Real, β::Real, γ::Real)
    r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax; α=float(α), β=float(β), γ=float(γ))
    function loss(a, b, c)
        r.α = float(a); r.β = float(b); r.γ = float(c)
        R = similar(Qlm)
        SHTnsKit.shtns_rotation_apply_real(r, Qlm, R)
        return 0.5 * sum(abs2, R)
    end
    g = Zygote.gradient(loss, α, β, γ)
    return g[1], g[2], g[3]
end

"""
    zgrad_rotation_angles_cplx(lmax, mmax, Zlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_cplx(lmax,mmax,Zlm; α,β,γ) using Zygote.
"""
function SHTnsKit.zgrad_rotation_angles_cplx(lmax::Integer, mmax::Integer, Zlm::AbstractVector, α::Real, β::Real, γ::Real)
    r = SHTnsKit.SHTRotation(Int(lmax), Int(mmax); α=float(α), β=float(β), γ=float(γ))
    function loss(a, b, c)
        r.α = float(a); r.β = float(b); r.γ = float(c)
        R = similar(Zlm)
        SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, R)
        return 0.5 * sum(abs2, R)
    end
    g = Zygote.gradient(loss, α, β, γ)
    return g[1], g[2], g[3]
end

# -----------------------------
## Zygote-specific adjoints for rotations/operators to ensure gradients are not `nothing`
## These mirror the ChainRules rrules but live here to guarantee Zygote picks them up.

Zygote.@adjoint function SHTnsKit.SH_Zrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, -alpha, Q̄)
        dα = 0.0
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1
                Rval = Qlm[lm] * cis(m * alpha)
                dα += real(conj(ȳ[lm]) * ((0 + 1im) * m * Rval))
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_Yrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ, -alpha, Q̄)
        # angle gradient via derivative of Wigner-d at beta=alpha
        dα = 0.0
        lmax, mmax = cfg.lmax, cfg.mmax
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ), 2l+1)
            for mp in -mm:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
                if mp == 0
                    b[mp + l + 1] = Qlm[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm[idxp])
                end
            end
            dd = SHTnsKit.wigner_d_matrix_deriv(l, float(alpha))
            for m in 0:mm
                lm = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ[lm]) * s)
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function back(ȳ)
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        mx̄ = zeros(eltype(mx), length(mx))
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            c_minus = mx[2*lm0 + 1]
            c_plus  = mx[2*lm0 + 2]
            rbar = ȳ[lm0 + 1]
            if l > m && l > 0
                lm_prev = SHTnsKit.LM_index(lmax, mres, l-1, m)
                Q̄[lm_prev + 1] += conj(c_minus) * rbar
                mx̄[2*lm0 + 1] += real(conj(rbar) * Qlm[lm_prev + 1])
            end
            if l < lmax
                lm_next = SHTnsKit.LM_index(lmax, mres, l+1, m)
                Q̄[lm_next + 1] += conj(c_plus) * rbar
                mx̄[2*lm0 + 2] += real(conj(rbar) * Qlm[lm_next + 1])
            end
        end
        return (nothing, mx̄, Q̄, nothing)
    end
    return y, back
end

end # module
