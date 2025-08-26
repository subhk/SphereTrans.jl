module SHTnsKitZygoteExt

using Zygote
using ..SHTnsKit

"""
    zgrad_scalar_energy(cfg, f) -> ∂E/∂f

Zygote gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.
Returns an array the same size as `f`.
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, x))
    return Zygote.gradient(loss, f)[1]
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

end # module
