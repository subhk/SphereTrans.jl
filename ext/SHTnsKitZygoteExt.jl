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
    lmax, mmax = cfg.lmax, cfg.mmax
    # Forward rotate to get R (used as cotangent in loss 0.5||R||^2)
    R = similar(Qlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_real(r, Qlm, R)
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        b = Vector{ComplexF64}(undef, n)
        # Build complex A_m' then b = e^{-i m' γ} A
        for mp in -mm:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
            if mp == 0
                A = Qlm[idxp]
            elseif mp > 0
                A = Qlm[idxp]
            else
                A = (-1)^(-mp) * conj(Qlm[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
            end
            b[mp + l + 1] = A * cis(-mp * float(γ))
        end
        c = dl * b
        for m in 0:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
            Rm = c[m + l + 1] * cis(-m * float(α))
            ȳ = R[idxp]
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # β gradient via d'(β)
            sβ = zero(ComplexF64)
            sγ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

"""
    zgrad_rotation_angles_cplx(lmax, mmax, Zlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_cplx(lmax,mmax,Zlm; α,β,γ) using Zygote.
"""
function SHTnsKit.zgrad_rotation_angles_cplx(lmax::Integer, mmax::Integer, Zlm::AbstractVector, α::Real, β::Real, γ::Real)
    lmax = Int(lmax); mmax = Int(mmax)
    R = similar(Zlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, R)
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        # Build b_m' = e^{-i m' γ} Z_{l,m'} for m' in [-l..l]
        b = Vector{ComplexF64}(undef, n)
        for mp in -l:l
            idx = SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) + 1
            b[mp + l + 1] = Zlm[idx] * cis(-mp * float(γ))
        end
        # c_m = sum_{m'} d_{m m'}(β) b_{m'}
        c = dl * b
        for m in -mm:mm
            idxm = SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1
            ȳ = R[idxm]
            Rm = c[m + l + 1] * cis(-m * float(α))
            # α-gradient: -i m R_m
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # γ-gradient: through input phase of b
            sγ = zero(ComplexF64)
            for mp in -l:l
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
            # β-gradient: via d'(β)
            sβ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

# -----------------------------
## Zygote-specific adjoints for rotations/operators to ensure gradients are not `nothing`
## These mirror the ChainRules rrules but live here to guarantee Zygote picks them up.

Zygote.@adjoint function SHTnsKit.SH_Zrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        # For loss 0.5||R||^2 with R = e^{i m α} ∘ Q, gradient wrt Q is conj(Q)
        Q̄ = conj.(Qlm)
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
