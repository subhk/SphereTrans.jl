"""
Core data structures and types for SHTns Julia implementation.
"""

# Spherical harmonic normalization types
@enum SHTnsNorm begin
    SHT_ORTHONORMAL = 0  # Orthonormal (4π normalization)
    SHT_FOURPI = 1       # 4π normalization
    SHT_SCHMIDT = 2      # Schmidt normalization
    SHT_REAL_NORM = 3    # Real normalization
end

# Grid types for spherical harmonic transforms
@enum SHTnsGrid begin
    SHT_GAUSS = 0     # Gauss-Legendre grid (recommended)
    SHT_REGULAR = 1   # Regular (equiangular) grid
    SHT_DCT = 2       # Discrete Cosine Transform grid
end

# Transform algorithm types
@enum SHTnsType begin
    SHT_SCALAR = 0    # Scalar transforms only
    SHT_VECTOR = 1    # Vector transforms
    SHT_COMPLEX = 2   # Complex-valued transforms
end

"""
    SHTnsConfig{T<:AbstractFloat}

Configuration structure for spherical harmonic transforms.
Contains all precomputed data needed for efficient transforms.

# Fields
- `lmax::Int`: Maximum spherical harmonic degree
- `mmax::Int`: Maximum spherical harmonic order
- `mres::Int`: Azimuthal resolution parameter
- `nlm::Int`: Number of (l,m) coefficients
- `nlat::Int`: Number of latitude grid points
- `nphi::Int`: Number of longitude grid points
- `grid_type::SHTnsGrid`: Type of spatial grid
- `norm::SHTnsNorm`: Spherical harmonic normalization
- `gauss_weights::Vector{T}`: Gaussian quadrature weights
- `gauss_nodes::Vector{T}`: Gaussian quadrature nodes (cos θ)
- `theta_grid::Vector{T}`: Latitude grid points (colatitude in radians)
- `phi_grid::Vector{T}`: Longitude grid points (in radians)
- `plm_cache::Matrix{T}`: Cached Legendre polynomial values
- `lm_indices::Vector{Tuple{Int,Int}}`: (l,m) index mapping
- `fft_plans::Dict{Symbol,Any}`: Cached FFT plans for efficiency
"""
mutable struct SHTnsConfig{T<:AbstractFloat}
    # Basic configuration
    lmax::Int
    mmax::Int
    mres::Int
    nlm::Int
    nlat::Int
    nphi::Int
    
    # Grid configuration
    grid_type::SHTnsGrid
    norm::SHTnsNorm
    
    # Gaussian quadrature (for Gauss grids)
    gauss_weights::Vector{T}
    gauss_nodes::Vector{T}  # cos(θ) values
    
    # Grid coordinates
    theta_grid::Vector{T}   # colatitude [0, π]
    phi_grid::Vector{T}     # longitude [0, 2π)
    
    # Precomputed data for transforms
    plm_cache::Matrix{T}    # Legendre polynomials P_l^m(cos θ)
    lm_indices::Vector{Tuple{Int,Int}}  # (l,m) pairs for indexing
    
    # FFT plans for efficiency
    fft_plans::Dict{Symbol,Any}
    
    # Thread safety
    lock::ReentrantLock
    
    # Robert form flag for vector transforms
    robert_form::Bool
    
    # Inner constructor
    function SHTnsConfig{T}() where T<:AbstractFloat
        new{T}(
            0, 0, 0, 0, 0, 0,
            SHT_GAUSS, SHT_ORTHONORMAL,
            T[], T[], T[], T[],
            Matrix{T}(undef, 0, 0),
            Tuple{Int,Int}[],
            Dict{Symbol,Any}(),
            ReentrantLock(),
            false
        )
    end
end

# Convenience constructors
SHTnsConfig() = SHTnsConfig{Float64}()
SHTnsConfig(T::Type{<:AbstractFloat}) = SHTnsConfig{T}()

"""
    SHTnsTransform{T}

Container for spherical harmonic transform operations.
Holds temporary arrays and working space for efficient transforms.
"""
mutable struct SHTnsTransform{T<:AbstractFloat}
    config::SHTnsConfig{T}
    work_spectral::Vector{Complex{T}}
    work_spatial::Array{Complex{T}, 2}
    work_temp::Vector{Complex{T}}
    
    function SHTnsTransform{T}(cfg::SHTnsConfig{T}) where T
        nlm = cfg.nlm
        nlat, nphi = cfg.nlat, cfg.nphi
        
        new{T}(
            cfg,
            zeros(Complex{T}, nlm),
            zeros(Complex{T}, nlat, nphi),
            zeros(Complex{T}, max(nlm, nlat*nphi))
        )
    end
end

# Convenience constructors
SHTnsTransform(cfg::SHTnsConfig{T}) where T = SHTnsTransform{T}(cfg)

"""
    validate_config(cfg::SHTnsConfig)

Validate that a spherical harmonic configuration is properly initialized.
"""
function validate_config(cfg::SHTnsConfig)
    cfg.lmax >= 0 || error("lmax must be non-negative")
    cfg.mmax >= 0 || error("mmax must be non-negative") 
    cfg.mmax <= cfg.lmax || error("mmax must be <= lmax")
    cfg.mres >= 1 || error("mres must be >= 1")
    cfg.nlm > 0 || error("nlm must be positive")
    cfg.nlat > 0 || error("nlat must be positive")
    cfg.nphi > 0 || error("nphi must be positive")
    
    if cfg.grid_type == SHT_GAUSS
        cfg.nlat > cfg.lmax || error("For Gauss grid: nlat must be > lmax")
    else
        cfg.nlat >= 2*cfg.lmax + 1 || error("For regular grid: nlat must be >= 2*lmax + 1")
    end
    
    cfg.nphi >= 2*cfg.mmax + 1 || error("nphi must be >= 2*mmax + 1")
    
    return true
end

"""
Show method for SHTnsConfig to provide useful information.
"""
function Base.show(io::IO, cfg::SHTnsConfig{T}) where T
    println(io, "SHTnsConfig{$T}:")
    println(io, "  lmax = $(cfg.lmax), mmax = $(cfg.mmax), mres = $(cfg.mres)")
    println(io, "  nlm = $(cfg.nlm)")
    println(io, "  Grid: $(cfg.nlat) × $(cfg.nphi) ($(cfg.grid_type))")
    println(io, "  Normalization: $(cfg.norm)")
    if !isempty(cfg.gauss_weights)
        println(io, "  Gaussian quadrature initialized")
    end
    if !isempty(cfg.plm_cache)
        println(io, "  Legendre polynomials cached")
    end
end