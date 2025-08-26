"""
Configuration for Spherical Harmonic Transforms.

Fields
- `lmax, mmax`: maximum degree and order.
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular).
- `θ, φ`: polar and azimuth arrays.
- `x, w`: Gauss–Legendre nodes and weights for `x = cos(θ)`.
- `Nlm`: normalization factors `(l+1, m+1)`.
"""
Base.@kwdef mutable struct SHTConfig
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
    # Options
    norm::Symbol
    cs_phase::Bool
    real_norm::Bool
    robert_form::Bool
    # Optional precomputed Legendre tables for speed on regular grids
    use_plm_tables::Bool = false
    plm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]  # per m: (lmax+1)×nlat
end

"""
    create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, nlon::Int=max(2*lmax+1, 4)) -> SHTConfig

Create a Gauss–Legendre based SHT configuration. Constraints:
- `nlat ≥ lmax+1` for exactness up to `lmax` in θ integration.
- `nlon ≥ 2*mmax+1` to resolve azimuthal orders up to `mmax`.
"""
function create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1, nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal, cs_phase::Bool=true, real_norm::Bool=false, robert_form::Bool=false)
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlat ≥ lmax + 1 || throw(ArgumentError("nlat must be ≥ lmax+1 for Gauss–Legendre accuracy"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))

    θ, φ, x, w = thetaphi_from_nodes(nlat, nlon)
    Nlm = Nlm_table(lmax, mmax)  # currently orthonormal; future: adjust per norm/cs_phase
    nlm = nlm_calc(lmax, mmax, mres)
    li, mi = build_li_mi(lmax, mmax, mres)
    ct = cos.(θ)
    st = sin.(θ)
    return SHTConfig(; lmax, mmax, mres, nlat, nlon, θ, φ, x, w, Nlm,
                     cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                     ct, st, norm, cs_phase, real_norm, robert_form)
end

"""
    prepare_plm_tables!(cfg::SHTConfig)

Precompute associated Legendre tables P_l^m(x_i) for all i and m, stored as
`cfg.plm_tables[m+1][l+1, i]`. Enables faster scalar transforms on regular grids.
"""
function prepare_plm_tables!(cfg::SHTConfig)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat = cfg.nlat
    tables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]
    P = Vector{Float64}(undef, lmax + 1)
    for m in 0:mmax
        tbl = tables[m+1]
        for i in 1:nlat
            Plm_row!(P, cfg.x[i], lmax, m)
            @inbounds @views tbl[:, i] .= P
        end
    end
    cfg.plm_tables = tables
    cfg.use_plm_tables = true
    return cfg
end

"""
    destroy_config(cfg::SHTConfig)

No-op placeholder for API symmetry with libraries that require explicit teardown.
"""
destroy_config(::SHTConfig) = nothing
