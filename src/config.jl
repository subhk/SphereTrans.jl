"""
Configuration for Spherical Harmonic Transforms.

This struct contains all parameters and precomputed data needed for efficient
spherical harmonic transforms. It encapsulates both the mathematical parameters
(degrees, grid sizes) and computational optimizations (precomputed tables).

Fields
- `lmax, mmax`: maximum degree and order for spherical harmonics
- `mres`: resolution parameter for m-modes (typically 1)
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular)
- `θ, φ`: polar and azimuth angle arrays for the computational grid
- `x, w`: Gauss–Legendre nodes and weights for numerical integration (x = cos(θ))
- `Nlm`: normalization factors matrix indexed as (l+1, m+1)
- `cphi`: longitude step size (2π / nlon) for FFT operations
"""
Base.@kwdef mutable struct SHTConfig
    # Core spherical harmonic parameters
    lmax::Int                    # Maximum spherical harmonic degree
    mmax::Int                    # Maximum spherical harmonic order  
    mres::Int                    # M-resolution parameter
    nlat::Int                    # Number of latitude points (Gauss-Legendre)
    nlon::Int                    # Number of longitude points (equiangular)
    
    # Grid coordinates and quadrature
    θ::Vector{Float64}          # Polar angles (colatitude) [0, π]
    φ::Vector{Float64}          # Azimuthal angles [0, 2π)
    x::Vector{Float64}          # Gauss-Legendre nodes: x = cos(θ) ∈ [-1,1]
    w::Vector{Float64}          # Gauss-Legendre integration weights
    Nlm::Matrix{Float64}        # Normalization factors for Y_l^m
    cphi::Float64               # Longitude spacing: 2π / nlon
    
    # SHTns-compatible helper fields for efficient indexing
    nlm::Int                    # Total number of (l,m) modes
    li::Vector{Int}             # Degree indices for flattened (l,m) arrays
    mi::Vector{Int}             # Order indices for flattened (l,m) arrays  
    nspat::Int                  # Total spatial grid points: nlat × nlon
    ct::Vector{Float64}         # Precomputed cos(θ) values
    st::Vector{Float64}         # Precomputed sin(θ) values
    
    # Transform normalization and phase conventions
    norm::Symbol                # Normalization type (:orthonormal, :schmidt, etc.)
    cs_phase::Bool              # Condon-Shortley phase convention
    real_norm::Bool             # Real-valued normalization
    robert_form::Bool           # Robert form for spectral derivatives
    
    # Performance optimization: precomputed Legendre polynomials
    use_plm_tables::Bool = false                              # Enable/disable table lookup
    plm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]   # P_l^m values: [m+1][l+1, lat_idx]
    dplm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]  # dP_l^m/dx values: [m+1][l+1, lat_idx]
end

"""
    create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, nlon::Int=max(2*lmax+1, 4)) -> SHTConfig

Create a Gauss–Legendre based SHT configuration. Constraints:
- `nlat ≥ lmax+1` for exactness up to `lmax` in θ integration.
- `nlon ≥ 2*mmax+1` to resolve azimuthal orders up to `mmax`.
"""
function create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1, nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal, cs_phase::Bool=true, real_norm::Bool=false, robert_form::Bool=false)
    # Validate input parameters to ensure mathematical accuracy requirements
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlat ≥ lmax + 1 || throw(ArgumentError("nlat must be ≥ lmax+1 for Gauss–Legendre accuracy"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))

    # Build the computational grid using Gauss-Legendre quadrature
    θ, φ, x, w = thetaphi_from_nodes(nlat, nlon)
    
    # Compute normalization factors for spherical harmonics
    Nlm = Nlm_table(lmax, mmax)  # currently orthonormal; future: adjust per norm/cs_phase
    
    # Calculate indexing helpers for efficient (l,m) mode access
    nlm = nlm_calc(lmax, mmax, mres)              # Total number of spectral modes
    li, mi = build_li_mi(lmax, mmax, mres)        # Degree and order index arrays
    
    # Precompute trigonometric values for performance
    ct = cos.(θ)  # cosine of colatitude
    st = sin.(θ)  # sine of colatitude
    
    # Construct and return the complete configuration
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
    dtables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    for m in 0:mmax
        tbl = tables[m+1]
        dtbl = dtables[m+1]
        for i in 1:nlat
            Plm_and_dPdx_row!(P, dPdx, cfg.x[i], lmax, m)
            @inbounds @views tbl[:, i] .= P
            @inbounds @views dtbl[:, i] .= dPdx
        end
    end
    cfg.plm_tables = tables
    cfg.dplm_tables = dtables
    cfg.use_plm_tables = true
    return cfg
end

"""
    enable_plm_tables!(cfg::SHTConfig)

Alias for `prepare_plm_tables!`.
"""
enable_plm_tables!(cfg::SHTConfig) = prepare_plm_tables!(cfg)

"""
    disable_plm_tables!(cfg::SHTConfig)

Disable use of precomputed Legendre tables.
"""
function disable_plm_tables!(cfg::SHTConfig)
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    return cfg
end

"""
    destroy_config(cfg::SHTConfig)

No-op placeholder for API symmetry with libraries that require explicit teardown.
"""
destroy_config(::SHTConfig) = nothing
