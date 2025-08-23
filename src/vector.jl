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
    SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix;
                   real_output::Bool=true) -> Vr, Vt, Vp

3D synthesis: combine scalar radial (Qlm) with vector tangential (Slm,Tlm).
"""
function SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    Vr = synthesis(cfg, Qlm; real_output=real_output)
    Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=real_output)
    return Vr, Vt, Vp
end

"""
    spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
        -> Qlm, Slm, Tlm

3D analysis: project radial onto scalar Y_lm and tangential onto spheroidal/toroidal.
"""
function spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    size(Vr,1) == cfg.nlat && size(Vr,2) == cfg.nlon || throw(DimensionMismatch("Vr dims"))
    size(Vt,1) == cfg.nlat && size(Vt,2) == cfg.nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == cfg.nlat && size(Vp,2) == cfg.nlon || throw(DimensionMismatch("Vp dims"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    return Qlm, Slm, Tlm
end

"""
    SHqst_to_spat_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
        -> Vr::Matrix{ComplexF64}, Vt::Matrix{ComplexF64}, Vp::Matrix{ComplexF64}

Complex 3D synthesis.
"""
function SHqst_to_spat_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    Vr = synthesis(cfg, Qlm; real_output=false)
    Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=false)
    return Vr, Vt, Vp
end

"""
    spat_cplx_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
        -> Qlm, Slm, Tlm

Complex 3D analysis.
"""
function spat_cplx_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    size(Vr,1) == cfg.nlat && size(Vr,2) == cfg.nlon || throw(DimensionMismatch("Vr dims"))
    size(Vt,1) == cfg.nlat && size(Vt,2) == cfg.nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == cfg.nlat && size(Vp,2) == cfg.nlon || throw(DimensionMismatch("Vp dims"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    return Qlm, Slm, Tlm
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
            @inbounds for l in max(1,m):lmax
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
    SHsphtor_to_spat_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
        -> Vt::Matrix{ComplexF64}, Vp::Matrix{ComplexF64}

Complex synthesis wrapper.
"""
function SHsphtor_to_spat_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    return SHsphtor_to_spat(cfg, Slm, Tlm; real_output=false)
end

"""
    spat_cplx_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
        -> Slm, Tlm

Complex analysis wrapper.
"""
function spat_cplx_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    return spat_to_SHsphtor(cfg, Vt, Vp)
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

"""
    SHsphtor_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
        -> Vt, Vp

Truncated vector synthesis using only degrees l ≤ ltr.
"""
function SHsphtor_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    # Copy and zero out > ltr
    S2 = copy(Slm); T2 = copy(Tlm)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            S2[l+1, m+1] = 0
            T2[l+1, m+1] = 0
        end
    end
    return SHsphtor_to_spat(cfg, S2, T2; real_output)
end

"""
    spat_to_SHsphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
        -> Slm, Tlm

Truncated vector analysis; zeroes Slm/Tlm for l > ltr.
"""
function spat_to_SHsphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end
    end
    return Slm, Tlm
end

"""
    SHsph_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
"""
function SHsph_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Slm))
    return SHsphtor_to_spat_l(cfg, Slm, Z, ltr; real_output)
end

"""
    SHtor_to_spat_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
"""
function SHtor_to_spat_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Tlm))
    return SHsphtor_to_spat_l(cfg, Z, Tlm, ltr; real_output)
end

"""
    spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
        -> Sl::Vector{ComplexF64}, Tl::Vector{ComplexF64}

Per-m vector analysis (no FFT) truncated at ltr.
"""
function spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vt_m) == nlat && length(Vp_m) == nlat || throw(DimensionMismatch("per-m inputs must have length nlat"))
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || throw(ArgumentError("invalid m from im"))
    lstart = max(1, m)
    (lstart ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require max(1,m) ≤ ltr ≤ lmax"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    Sl = zeros(ComplexF64, ltr - lstart + 1)
    Tl = zeros(ComplexF64, ltr - lstart + 1)
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
        wi = cfg.w[i]
        Fθ_i = Vt_m[i]
        Fφ_i = Vp_m[i]
        @inbounds for l in lstart:ltr
            N = cfg.Nlm[l+1, m+1]
            dθY = -sθ * N * dPdx[l+1]
            Y = N * P[l+1]
            coeff = wi / (l*(l+1))
            Sl[l - lstart + 1] += coeff * (Fθ_i * dθY + Fφ_i * (-(0 + 1im) * m * inv_sθ * Y))
            Tl[l - lstart + 1] += coeff * (Fθ_i * ((0 + 1im) * m * inv_sθ * Y) + Fφ_i * (+sθ * N * dPdx[l+1]))
        end
    end
    return Sl, Tl
end

"""
    SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
        -> Vt_m::Vector{ComplexF64}, Vp_m::Vector{ComplexF64}

Per-m vector synthesis (no FFT) truncated at ltr.
"""
function SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    m = im * cfg.mres
    lstart = max(1, m)
    length(Sl) == ltr - lstart + 1 || throw(DimensionMismatch("Sl length must be ltr-max(1,m)+1"))
    length(Tl) == ltr - lstart + 1 || throw(DimensionMismatch("Tl length must be ltr-max(1,m)+1"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    Vt_m = Vector{ComplexF64}(undef, cfg.nlat)
    Vp_m = Vector{ComplexF64}(undef, cfg.nlat)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        @inbounds for l in lstart:ltr
            N = cfg.Nlm[l+1, m+1]
            dθY = -sθ * N * dPdx[l+1]
            Y = N * P[l+1]
            Slv = Sl[l - lstart + 1]
            Tlv = Tl[l - lstart + 1]
            gθ += dθY * Slv + (0 + 1im) * m * inv_sθ * Y * Tlv
            gφ += (0 + 1im) * m * inv_sθ * Y * Slv + (sθ * N * dPdx[l+1]) * Tlv
        end
        Vt_m[i] = gθ
        Vp_m[i] = gφ
    end
    return Vt_m, Vp_m
end

"""
    spat_to_SHqst_l(cfg::SHTConfig, Vr, Vt, Vp, ltr::Int)
        -> Qlm, Slm, Tlm
"""
function spat_to_SHqst_l(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Qlm[l+1, m+1] = 0
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end
    end
    return Qlm, Slm, Tlm
end

"""
    SHqst_to_spat_l(cfg::SHTConfig, Qlm, Slm, Tlm, ltr::Int; real_output::Bool=true)
        -> Vr, Vt, Vp
"""
function SHqst_to_spat_l(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Q2 = copy(Qlm); S2 = copy(Slm); T2 = copy(Tlm)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Q2[l+1, m+1] = 0
            S2[l+1, m+1] = 0
            T2[l+1, m+1] = 0
        end
    end
    Vr = synthesis(cfg, Q2; real_output)
    Vt, Vp = SHsphtor_to_spat(cfg, S2, T2; real_output)
    return Vr, Vt, Vp
end

"""
    spat_to_SHqst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
        -> Ql, Sl, Tl
"""
function spat_to_SHqst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    Ql = spat_to_SH_ml(cfg, im, Vr_m, ltr)
    Sl, Tl = spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr)
    return Ql, Sl, Tl
end

"""
    SHqst_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
        -> Vr_m, Vt_m, Vp_m
"""
function SHqst_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    Vr_m = SH_to_spat_ml(cfg, im, Ql, ltr)
    Vt_m, Vp_m = SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr)
    return Vr_m, Vt_m, Vp_m
end
