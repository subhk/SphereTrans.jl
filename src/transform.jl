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

