"""
Vector spherical harmonic transforms (spheroidal/toroidal) in pure Julia.

We use the decomposition V = ∇_s S + r̂ × ∇_s T where S,T are scalar potentials with SH
coefficients Slm, Tlm. On the unit sphere:
- Vθ = ∂θ S + (im/ sinθ) T
- Vφ = (im/ sinθ) S - ∂θ T

with Y_lm(θ,φ) = N_{l,m} P_l^m(cosθ) e^{imφ}. We implement analysis/synthesis by FFT in φ and
Gauss–Legendre quadrature in θ, like the scalar case.
"""

"""
    SHsphtor_to_spat(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
        -> Vt::Matrix{Float64}, Vp::Matrix{Float64}

Synthesize vector field components (θ, φ) from spheroidal/toroidal spectra.
"""
function SHsphtor_to_spat(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))
    nlat, nlon = cfg.nlat, cfg.nlon
    Fθ = Matrix{ComplexF64}(undef, nlat, nlon)
    Fφ = Matrix{ComplexF64}(undef, nlat, nlon)
    fill!(Fθ, 0.0 + 0.0im); fill!(Fφ, 0.0 + 0.0im)

    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    inv_scaleφ = nlon / (2π)

    @threads for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            x = cfg.x[i]
            sθ = sqrt(max(0.0, 1 - x*x))
            Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
            gθ = 0.0 + 0.0im
            gφ = 0.0 + 0.0im
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            @inbounds for l in m:lmax
                N = cfg.Nlm[l+1, col]
                # ∂θ term: ∂θ Y = -sinθ * N * dPdx
                dθY = -sθ * N * dPdx[l+1]
                Y = N * P[l+1]
                Sl = Slm[l+1, col]
                Tl = Tlm[l+1, col]
                gθ += dθY * Sl + (0 + 1im) * m * inv_sθ * Y * Tl
                gφ += (0 + 1im) * m * inv_sθ * Y * Sl + (sθ * N * dPdx[l+1]) * Tl
            end
            Fθ[i, col] = inv_scaleφ * gθ
            Fφ[i, col] = inv_scaleφ * gφ
        end
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fθ[i, conj_index] = conj(Fθ[i, col])
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end

    Vt = real_output ? real.(ifft(Fθ, 2)) : ifft(Fθ, 2)
    Vp = real_output ? real.(ifft(Fφ, 2)) : ifft(Fφ, 2)
    return Vt, Vp
end

"""
    spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
        -> Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}

Analyze vector field components (θ, φ) to spheroidal/toroidal spectra.
"""
function spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm = zeros(ComplexF64, lmax+1, mmax+1)

    Fθ = fft(ComplexF64.(Vt), 2)
    Fφ = fft(ComplexF64.(Vp), 2)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi

    @threads for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            x = cfg.x[i]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
            Fθ_i = Fθ[i, col]
            Fφ_i = Fφ[i, col]
            wi = cfg.w[i]
            @inbounds for l in m:lmax
                N = cfg.Nlm[l+1, col]
                dθY = -sθ * N * dPdx[l+1]
                Y = N * P[l+1]
                # Projections using vector spherical harmonics orthogonality: divide by l(l+1)
                coeff = wi * scaleφ / (l*(l+1))
                Slm[l+1, col] += coeff * (Fθ_i * dθY + Fφ_i * (-(0 + 1im) * m * inv_sθ * Y))
                Tlm[l+1, col] += coeff * (Fθ_i * ((0 + 1im) * m * inv_sθ * Y) + Fφ_i * (+sθ * N * dPdx[l+1]))
            end
        end
    end
    return Slm, Tlm
end

"""
    SHsph_to_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
        -> Vt, Vp

Synthesize only the spheroidal part (gradient of S).
"""
function SHsph_to_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Slm))
    return SHsphtor_to_spat(cfg, Slm, Z; real_output)
end

"""
    SHtor_to_spat(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
        -> Vt, Vp

Synthesize only the toroidal part.
"""
function SHtor_to_spat(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Tlm))
    return SHsphtor_to_spat(cfg, Z, Tlm; real_output)
end

"""
    SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true) -> Vt, Vp

Alias to `SHsph_to_spat`, for compatibility with SHTns macro.
"""
SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true) = SHsph_to_spat(cfg, Slm; real_output)
