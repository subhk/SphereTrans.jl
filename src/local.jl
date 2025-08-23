"""
Local/partial evaluations along latitude circles and at points.
"""

"""
    SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real;
              nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vector{Float64}

Evaluate a real field along a latitude (fixed cosθ = cost) at `nphi` equispaced longitudes.
Uses orthonormal harmonics and packed real coefficients `Qlm` (LM order).
"""
function SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    x = float(cost)
    lmax = cfg.lmax
    P = Vector{Float64}(undef, lmax + 1)
    vals = Vector{Float64}(undef, nphi)
    fill!(vals, 0.0)

    # m=0 contribution
    Plm_row!(P, x, lmax, 0)
    g0 = 0.0 + 0.0im
    @inbounds for l in 0:ltr
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        g0 += cfg.Nlm[l+1, 1] * P[l+1] * Qlm[lm]
    end
    @inbounds for j in 0:(nphi-1)
        vals[j+1] = real(g0)
    end

    # m>0
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        Plm_row!(P, x, lmax, m)
        gm = 0.0 + 0.0im
        col = m + 1
        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            gm += cfg.Nlm[l+1, col] * P[l+1] * Qlm[lm]
        end
        for j in 0:(nphi-1)
            vals[j+1] += 2 * real(gm * cis(2π * m * j / nphi))
        end
    end
    return vals
end

"""
    SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                 nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp

Evaluate 3D field along latitude (cosθ = cost) at `nphi` longitudes from packed real spectra.
Inputs `Qlm, Slm, Tlm` are all packed (LM order) vectors for each component.
"""
function SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    x = float(cost)
    lmax = cfg.lmax
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    Vr = Vector{Float64}(undef, nphi)
    Vt = Vector{Float64}(undef, nphi)
    Vp = Vector{Float64}(undef, nphi)
    fill!(Vr, 0.0); fill!(Vt, 0.0); fill!(Vp, 0.0)

    # m=0
    Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)
    g0 = 0.0 + 0.0im
    gθ0 = 0.0 + 0.0im
    gφ0 = 0.0 + 0.0im
    sθ = sqrt(max(0.0, 1 - x*x))
    @inbounds for l in 0:ltr
        N = cfg.Nlm[l+1, 1]
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        Y = N * P[l+1]
        dθY = -sθ * N * dPdx[l+1]
        g0  += Y * Qlm[lm]
        gθ0 += dθY * Slm[lm]
        gφ0 += (sθ * N * dPdx[l+1]) * Tlm[lm]
    end
    @inbounds for j in 1:nphi
        Vr[j] += real(g0); Vt[j] += real(gθ0); Vp[j] += real(gφ0)
    end

    # m>0
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
        g  = 0.0 + 0.0im
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        col = m + 1
        @inbounds for l in m:min(ltr, lmax)
            N = cfg.Nlm[l+1, col]
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            Y = N * P[l+1]
            dθY = -sθ * N * dPdx[l+1]
            g  += Y   * Qlm[lm]
            gθ += dθY * Slm[lm] + (0 + 1im) * m * inv_sθ * Y * Tlm[lm]
            gφ += (0 + 1im) * m * inv_sθ * Y * Slm[lm] + (sθ * N * dPdx[l+1]) * Tlm[lm]
        end
        for j in 0:(nphi-1)
            phase = cis(2π * m * j / nphi)
            Vr[j+1] += 2 * real(g * phase)
            Vt[j+1] += 2 * real(gθ * phase)
            Vp[j+1] += 2 * real(gφ * phase)
        end
    end
    return Vr, Vt, Vp
end

