"""
Configuration for Spherical Harmonic Transforms.

Fields
- `lmax, mmax`: maximum degree and order.
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular).
- `θ, φ`: polar and azimuth arrays.
- `x, w`: Gauss–Legendre nodes and weights for `x = cos(θ)`.
- `Nlm`: normalization factors `(l+1, m+1)`.
"""
Base.@kwdef struct SHTConfig
    lmax::Int
    mmax::Int
    mres::Int
    nlat::Int
    nlon::Int
    θ::Vector{Float64}
    φ::Vector{Float64}
    x::Vector{Float64}
    w::Vector{Float64}
    Nlm::Matrix{Float64}
    cphi::Float64  # 2π / nlon
    # SHTns-compatible helper fields
    nlm::Int
    li::Vector{Int}
    mi::Vector{Int}
    nspat::Int
    ct::Vector{Float64}
    st::Vector{Float64}
end

"""
    create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, nlon::Int=max(2*lmax+1, 4)) -> SHTConfig

Create a Gauss–Legendre based SHT configuration. Constraints:
- `nlat ≥ lmax+1` for exactness up to `lmax` in θ integration.
- `nlon ≥ 2*mmax+1` to resolve azimuthal orders up to `mmax`.
"""
function create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1, nlon::Int=max(2*lmax+1, 4))
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlat ≥ lmax + 1 || throw(ArgumentError("nlat must be ≥ lmax+1 for Gauss–Legendre accuracy"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))

    θ, φ, x, w = thetaphi_from_nodes(nlat, nlon)
    Nlm = Nlm_table(lmax, mmax)
    nlm = nlm_calc(lmax, mmax, mres)
    li, mi = build_li_mi(lmax, mmax, mres)
    ct = cos.(θ)
    st = sin.(θ)
    return SHTConfig(; lmax, mmax, mres, nlat, nlon, θ, φ, x, w, Nlm,
                     cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                     ct, st)
end

"""
    destroy_config(cfg::SHTConfig)

No-op placeholder for API symmetry with libraries that require explicit teardown.
"""
destroy_config(::SHTConfig) = nothing
