"""
    analysis(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Forward spherical harmonic transform on Gauss–Legendre × equiangular grid.
Input grid `f` must be sized `(cfg.nlat, cfg.nlon)` and may be real or complex.
Returns coefficients `alm` of size `(cfg.lmax+1, cfg.mmax+1)` with indices `(l+1, m+1)`.
Normalization uses orthonormal spherical harmonics with Condon–Shortley phase.
"""
function analysis(cfg::SHTConfig, f::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    fC = ComplexF64.(f)

    # FFT along φ (dimension 2): sums f * exp(-i 2π m j/N)
    Fφ = fft(fC, 2)

    lmax, mmax = cfg.lmax, cfg.mmax
    alm = Matrix{ComplexF64}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0.0 + 0.0im)

    # Temporary buffer for P_l^m(x) at one θ
    P = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi  # 2π / nlon
    @threads for m in 0:mmax
        col = m + 1
        # Accumulate over latitudes
        for i in 1:nlat
            Plm_row!(P, cfg.x[i], lmax, m)
            Fi = Fφ[i, col]
            wi = cfg.w[i]
            @inbounds for l in m:lmax
                alm[l+1, col] += (wi * P[l+1]) * Fi
            end
        end
        # Apply normalization and φ scaling
        @inbounds for l in m:lmax
            alm[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    return alm
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Inverse spherical harmonic transform.
Input `alm` sized `(cfg.lmax+1, cfg.mmax+1)` with indices `(l+1, m+1)`.
Returns a grid `f` of shape `(cfg.nlat, cfg.nlon)`. If `real_output=true`,
enforces Hermitian symmetry to produce real-valued output.
"""
function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    Fφ = Matrix{ComplexF64}(undef, nlat, nlon)
    fill!(Fφ, 0.0 + 0.0im)

    # Temporary buffers
    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{ComplexF64}(undef, nlat)
    inv_scaleφ = nlon / (2π)

    # Build azimuthal spectra from alm
    @threads for m in 0:mmax
        col = m + 1
        # G(θ_i) = sum_{l=m}^{lmax} Nlm * P_l^m(x_i) * alm_{l,m}
        for i in 1:nlat
            Plm_row!(P, cfg.x[i], lmax, m)
            g = 0.0 + 0.0im
            @inbounds for l in m:lmax
                g += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
            end
            G[i] = g
        end
        # Place positive m Fourier modes
        @inbounds for i in 1:nlat
            Fφ[i, col] = inv_scaleφ * G[i]
        end
        # Hermitian conjugate for negative m to ensure real output
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end

    # Inverse FFT along φ
    f = ifft(Fφ, 2)
    return real_output ? real.(f) : f
end

"""
    spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real}) -> Vector{ComplexF64}

SHTns-compatible scalar analysis. `Vr` is a flat vector of length `cfg.nspat = nlat*nlon`.
Returns packed coefficients `Qlm` of length `cfg.nlm` with SHTns `LM` ordering.
"""
function spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = analysis(cfg, f)
    Qlm = Vector{ComplexF64}(undef, cfg.nlm)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm_mat[l+1, m+1]
        end
    end
    return Qlm
end

"""
    SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}) -> Vector{Float64}

SHTns-compatible scalar synthesis to a real spatial field. Input is packed `Qlm`.
Returns a flat `Vector{Float64}` length `nlat*nlon`.
"""
function SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

 

"""
    spat_to_SH_l(cfg::SHTConfig, Vr, ltr::Int)

Truncated scalar analysis up to degree `ltr`.
"""
function spat_to_SH_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Qlm = spat_to_SH(cfg, Vr)
    # zero out l > ltr
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in (ltr+1):cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = 0.0 + 0.0im
        end
    end
    return Qlm
end

"""
    SH_to_spat_l(cfg::SHTConfig, Qlm, ltr::Int)

Truncated scalar synthesis using only degrees `l ≤ ltr`.
"""
function SH_to_spat_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:min(ltr, cfg.lmax)
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)

Legendre-only transform at fixed `m = im*mres` truncated at `ltr`.
`Vr_m` is the Fourier mode along φ for each latitude (length `nlat`).
Returns spectrum `Ql` of length `ltr+1-m` for degrees `l=m..ltr`.
"""
function spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vr_m) == nlat || throw(DimensionMismatch("Vr_m must have length nlat"))
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || throw(ArgumentError("invalid m from im"))
    (m ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require m ≤ ltr ≤ lmax"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Ql = Vector{ComplexF64}(undef, ltr - m + 1)
    fill!(Ql, 0.0 + 0.0im)
    for i in 1:nlat
        Plm_row!(P, cfg.x[i], cfg.lmax, m)
        wi = cfg.w[i]
        Fi = Vr_m[i]
        @inbounds for l in m:ltr
            Ql[l - m + 1] += wi * P[l+1] * Fi * cfg.Nlm[l+1, m+1] * cfg.cphi
        end
    end
    return Ql
end

"""
    SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int) -> Vector{ComplexF64}

Legendre-only synthesis at fixed `m = im*mres` truncated at `ltr`.
Returns the Fourier mode across latitudes (length `nlat`).
"""
function SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int)
    m = im * cfg.mres
    (m ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require m ≤ ltr ≤ lmax"))
    length(Ql) == ltr - m + 1 || throw(DimensionMismatch("Ql length must be ltr-m+1"))
    nlat = cfg.nlat
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Vr_m = Vector{ComplexF64}(undef, nlat)
    for i in 1:nlat
        Plm_row!(P, cfg.x[i], cfg.lmax, m)
        g = 0.0 + 0.0im
        @inbounds for l in m:ltr
            g += cfg.Nlm[l+1, m+1] * P[l+1] * Ql[l - m + 1]
        end
        Vr_m[i] = (cfg.nlon / (2π)) * g
    end
    return Vr_m
end

"""
    SH_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real, phi::Real) -> Float64

Evaluate a real field represented by packed `Qlm` at a single point using orthonormal harmonics.
"""
function SH_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    x = float(cost)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    acc = 0.0 + 0.0im
    # m = 0 term
    Plm_row!(P, x, lmax, 0)
    g0 = 0.0 + 0.0im
    @inbounds for l in 0:lmax
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        g0 += cfg.Nlm[l+1, 1] * P[l+1] * Qlm[lm]
    end
    acc += g0
    # m > 0 with Hermitian symmetry for real field
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        Plm_row!(P, x, lmax, m)
        gm = 0.0 + 0.0im
        col = m + 1
        @inbounds for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            gm += cfg.Nlm[l+1, col] * P[l+1] * Qlm[lm]
        end
        acc += 2 * real(gm * cis(m * phi))
    end
    return real(acc)
end
