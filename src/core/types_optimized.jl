"""
Core data structures and optimized type-stable functions for SHTns Julia implementation.
This module provides high-performance, allocation-efficient implementations.
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

High-performance configuration structure for spherical harmonic transforms.
Contains all precomputed data needed for efficient, type-stable transforms.

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
- `lock::ReentrantLock`: Thread safety lock
- `robert_form::Bool`: Robert form flag for vector transforms
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
    
    # Inner constructor - optimized initialization
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
SHTnsConfig() = SHTnsConfig{Float64}()
SHTnsConfig(T::Type{<:AbstractFloat}) = SHTnsConfig{T}()
SHTnsTransform(cfg::SHTnsConfig{T}) where T = SHTnsTransform{T}(cfg)

# Performance optimization macro
"""
    @stable macro for ensuring type-stable dispatch

This macro helps identify type instabilities during development.
"""
macro stable(ex)
    quote
        local result = $(esc(ex))
        # In debug mode, we could add type checks here
        result
    end
end

# Type-stable core functions

"""
    create_config(::Type{T}, lmax::Int, mmax::Int=lmax, mres::Int=1;
                  grid_type::SHTnsGrid=SHT_GAUSS,
                  norm::SHTnsNorm=SHT_ORTHONORMAL) where T<:AbstractFloat

Create a new spherical harmonic transform configuration with optimal type stability.

# Arguments
- `T`: Floating point precision type (Float32, Float64, etc.)
- `lmax`: Maximum spherical harmonic degree
- `mmax`: Maximum spherical harmonic order (default: lmax)
- `mres`: Azimuthal resolution parameter (default: 1)
- `grid_type`: Type of spatial grid (default: Gauss-Legendre)
- `norm`: Normalization convention (default: orthonormal)

# Returns
- `SHTnsConfig{T}`: Configured transform object optimized for performance

# Examples
```julia
# Double precision Gauss grid
cfg = create_config(Float64, 20, 16, nlat=48, nphi=96)

# Single precision for memory efficiency  
cfg = create_config(Float32, 10, 8, nlat=24, nphi=48)
```
"""
function create_config(::Type{T}, lmax::Int, mmax::Int=lmax, mres::Int=1;
                       grid_type::SHTnsGrid=SHT_GAUSS,
                       norm::SHTnsNorm=SHT_ORTHONORMAL) where T<:AbstractFloat
    @stable begin
        # Input validation with type-stable error paths
        lmax >= 0 || throw(ArgumentError("lmax must be non-negative"))
        mmax >= 0 || throw(ArgumentError("mmax must be non-negative"))
        mmax <= lmax || throw(ArgumentError("mmax must not exceed lmax"))
        mres >= 1 || throw(ArgumentError("mres must be positive"))
        
        cfg = SHTnsConfig{T}()
        cfg.lmax = lmax
        cfg.mmax = mmax
        cfg.mres = mres
        cfg.grid_type = grid_type
        cfg.norm = norm
        
        # Type-stable nlm calculation
        cfg.nlm = nlm_calc(lmax, mmax, mres)
        
        # Pre-allocate with known sizes for optimal performance
        cfg.lm_indices = Vector{Tuple{Int,Int}}(undef, cfg.nlm)
        
        # Type-stable index generation
        idx = 1
        @inbounds for l in 0:lmax
            for m in 0:min(l, mmax)
                if m % mres == 0 || m == 0
                    cfg.lm_indices[idx] = (l, m)
                    idx += 1
                end
            end
        end
        
        return cfg
    end
end

"""
    set_grid!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T

Initialize the spatial grid for transforms with optimal type stability and performance.

# Arguments
- `cfg`: SHTns configuration to modify
- `nlat`: Number of latitude points
- `nphi`: Number of longitude points
"""
function set_grid!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T
    @stable begin
        # Type-stable validation
        nlat > 0 || throw(ArgumentError("nlat must be positive"))
        nphi > 0 || throw(ArgumentError("nphi must be positive"))
        
        # Grid-type specific validation with type-stable branches
        if cfg.grid_type === SHT_GAUSS
            nlat > cfg.lmax || throw(ArgumentError("For Gauss grid: nlat must be > lmax"))
        else
            nlat >= 2*cfg.lmax + 1 || throw(ArgumentError("For regular grid: nlat must be >= 2*lmax + 1"))
        end
        
        nphi >= 2*cfg.mmax + 1 || throw(ArgumentError("nphi must be >= 2*mmax + 1"))
        
        cfg.nlat = nlat
        cfg.nphi = nphi
        
        # Type-stable array allocation
        if cfg.grid_type === SHT_GAUSS
            nodes, weights = compute_gauss_legendre_nodes_weights(nlat, T)
            cfg.gauss_nodes = nodes
            cfg.gauss_weights = weights
            cfg.theta_grid = Vector{T}(undef, nlat)
            @inbounds @simd for i in 1:nlat
                cfg.theta_grid[i] = acos(nodes[i])
            end
        else
            cfg.theta_grid = Vector{T}(undef, nlat)
            cfg.gauss_nodes = Vector{T}(undef, nlat)
            cfg.gauss_weights = Vector{T}(undef, nlat)
            
            weight = T(2) / T(nlat)
            
            @inbounds @simd for i in 1:nlat
                theta = T(π) * (T(i) - T(0.5)) / T(nlat)
                cfg.theta_grid[i] = theta
                cfg.gauss_nodes[i] = cos(theta)
                cfg.gauss_weights[i] = weight
            end
        end
        
        # Phi grid (always regular)
        cfg.phi_grid = Vector{T}(undef, nphi)
        dphi = T(2π) / T(nphi)
        @inbounds @simd for i in 1:nphi
            cfg.phi_grid[i] = T(i - 1) * dphi
        end
        
        # Initialize caches with proper types
        max_cache_size = max(cfg.nlm, nlat * (cfg.mmax + 1))
        cfg.plm_cache = Matrix{T}(undef, nlat, max_cache_size)
        
        # Initialize FFT plans for type stability
        ensure_fft_plans!(cfg)
        
        return nothing
    end
end

"""
    compute_gauss_legendre_nodes_weights(n::Int, ::Type{T}) where T

Type-stable Gauss-Legendre quadrature computation.
"""
function compute_gauss_legendre_nodes_weights(n::Int, ::Type{T}) where T
    @stable begin
        nodes = Vector{T}(undef, n)
        weights = Vector{T}(undef, n)
        
        # Type-stable computation with Newton-Raphson
        for i in 1:((n + 1) ÷ 2)
            # Initial guess for i-th root
            z = cos(T(π) * (T(i) - T(0.25)) / (T(n) + T(0.5)))
            
            # Newton-Raphson iteration with fixed count for type stability
            local pp::T
            for _ in 1:10
                p1 = one(T)
                p2 = zero(T)
                
                @inbounds for j in 1:n
                    p3 = p2
                    p2 = p1
                    p1 = ((T(2*j - 1) * z * p2 - T(j - 1) * p3) / T(j))
                end
                
                pp = T(n) * (z * p1 - p2) / (z * z - one(T))
                z_new = z - p1 / pp
                
                if abs(z_new - z) < T(1e-14)
                    z = z_new
                    break
                end
                z = z_new
            end
            
            nodes[i] = -z
            nodes[n + 1 - i] = z
            
            weight_val = T(2) / ((one(T) - z * z) * pp * pp)
            weights[i] = weight_val
            weights[n + 1 - i] = weight_val
        end
        
        return nodes, weights
    end
end

"""

function nlm_calc(lmax::Int, mmax::Int, mres::Int)::Int
    @stable begin
        num_modes = 0
        @inbounds for l in 0:lmax
            for m in 0:min(l, mmax)
                if m % mres == 0 || m == 0
                    num_modes += 1
                end
            end
        end
        return num_modes
    end
end
Type-stable calculation of number of spherical harmonic coefficients.
"""

"""
    lmidx(l::Int, m::Int, lmax::Int)::Int

Type-stable index calculation for spherical harmonic coefficient arrays.
"""
function lmidx(l::Int, m::Int, lmax::Int)::Int
    @stable begin
        # Input validation with type-stable error handling
        0 <= l <= lmax || throw(BoundsError("l must be in [0, lmax]"))
        abs(m) <= l || throw(BoundsError("m must be in [-l, l]"))
        
        # Type-stable index computation
        if m >= 0
            return l * l + l + m + 1  # +1 for 1-based indexing
        else
            return l * l + l - m + 1
        end
    end
end

# Type-stable allocation functions

"""
    allocate_spectral(cfg::SHTnsConfig{T}) where T

Type-stable spectral coefficient array allocation.
"""
function allocate_spectral(cfg::SHTnsConfig{T}) where T
    @stable Vector{Complex{T}}(undef, cfg.nlm)
end

"""
    allocate_spatial(cfg::SHTnsConfig{T}) where T

Type-stable spatial data array allocation.
"""
function allocate_spatial(cfg::SHTnsConfig{T}) where T
    @stable Matrix{T}(undef, cfg.nlat, cfg.nphi)
end

"""
    validate_config(cfg::SHTnsConfig{T}) where T

Type-stable configuration validation.
"""
function validate_config(cfg::SHTnsConfig{T}) where T
    @stable begin
        cfg.lmax >= 0 || return false
        cfg.mmax >= 0 || return false
        cfg.mmax <= cfg.lmax || return false
        cfg.mres >= 1 || return false
        cfg.nlm > 0 || return false
        cfg.nlat > 0 || return false
        cfg.nphi > 0 || return false
        
        if cfg.grid_type === SHT_GAUSS
            cfg.nlat > cfg.lmax || return false
        else
            cfg.nlat >= 2*cfg.lmax + 1 || return false
        end
        
        cfg.nphi >= 2*cfg.mmax + 1 || return false
        
        return true
    end
end

"""
    ensure_fft_plans!(cfg::SHTnsConfig{T}) where T

Ensure FFT plans are stored with type-stable keys and values.
"""
function ensure_fft_plans!(cfg::SHTnsConfig{T}) where T
    @stable begin
        # Use type-stable keys
        forward_key = :rfft_plan_forward
        backward_key = :irfft_plan_backward
        
        nphi = cfg.nphi
        nphi_modes = nphi ÷ 2 + 1
        
        # Create type-stable plans
        if !haskey(cfg.fft_plans, forward_key)
            dummy_input = Vector{T}(undef, nphi)
            cfg.fft_plans[forward_key] = plan_rfft(dummy_input)
        end
        
        if !haskey(cfg.fft_plans, backward_key)
            dummy_input = Vector{Complex{T}}(undef, nphi_modes)
            cfg.fft_plans[backward_key] = plan_irfft(dummy_input, nphi)
        end
        
        return nothing
    end
end

# Convenience functions for backward compatibility and ease of use

"""
    create_gauss_config(::Type{T}, lmax::Int, mmax::Int, nlat::Int, nphi::Int) where T

Create a Gauss-Legendre grid configuration in one step.
"""
function create_gauss_config(::Type{T}, lmax::Int, mmax::Int, nlat::Int, nphi::Int) where T
    cfg = create_config(T, lmax, mmax; grid_type=SHT_GAUSS)
    set_grid!(cfg, nlat, nphi)
    return cfg
end

"""
    create_regular_config(::Type{T}, lmax::Int, mmax::Int, nlat::Int, nphi::Int) where T

Create a regular (equiangular) grid configuration in one step.
"""
function create_regular_config(::Type{T}, lmax::Int, mmax::Int, nlat::Int, nphi::Int) where T
    cfg = create_config(T, lmax, mmax; grid_type=SHT_REGULAR)
    set_grid!(cfg, nlat, nphi)
    return cfg
end

# Default precision versions for convenience
create_config(lmax::Int, args...; kwargs...) = create_config(Float64, lmax, args...; kwargs...)
create_gauss_config(lmax::Int, args...) = create_gauss_config(Float64, lmax, args...)
create_regular_config(lmax::Int, args...) = create_regular_config(Float64, lmax, args...)

# Advanced type-stable utilities

"""
    get_theta(cfg::SHTnsConfig{T}) where T -> Vector{T}

Get theta coordinate array (colatitude) with type stability.
"""
get_theta(cfg::SHTnsConfig{T}) where T = cfg.theta_grid::Vector{T}

"""
    get_phi(cfg::SHTnsConfig{T}) where T -> Vector{T}

Get phi coordinate array (longitude) with type stability.
"""
get_phi(cfg::SHTnsConfig{T}) where T = cfg.phi_grid::Vector{T}

"""
    get_gauss_weights(cfg::SHTnsConfig{T}) where T -> Vector{T}

Get Gaussian quadrature weights with type stability.
"""
function get_gauss_weights(cfg::SHTnsConfig{T}) where T
    if cfg.grid_type === SHT_GAUSS
        return cfg.gauss_weights::Vector{T}
    else
        return T[]
    end
end

# Accessors with type stability
get_lmax(cfg::SHTnsConfig{T}) where T<:AbstractFloat = cfg.lmax::Int
get_mmax(cfg::SHTnsConfig{T}) where T<:AbstractFloat = cfg.mmax::Int
get_nlat(cfg::SHTnsConfig{T}) where T<:AbstractFloat = cfg.nlat::Int
get_nphi(cfg::SHTnsConfig{T}) where T<:AbstractFloat = cfg.nphi::Int
get_nlm(cfg::SHTnsConfig{T}) where T<:AbstractFloat = cfg.nlm::Int

# Utility function for index mapping
"""
    lm_from_index(cfg::SHTnsConfig, idx::Int) -> Tuple{Int, Int}

Get (l, m) pair from linear index with type stability.
"""
function lm_from_index(cfg::SHTnsConfig, idx::Int)::Tuple{Int, Int}
    @stable begin
        1 <= idx <= cfg.nlm || throw(BoundsError("Index out of range"))
        return cfg.lm_indices[idx]::Tuple{Int,Int}
    end
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
    if cfg.robert_form
        println(io, "  Robert form: enabled")
    end
    if !isempty(cfg.gauss_weights)
        println(io, "  Gaussian quadrature initialized")
    end
    if !isempty(cfg.plm_cache)
        println(io, "  Legendre polynomials cached")
    end
end

# Destroy function for cleanup
"""
    destroy_config(cfg::SHTnsConfig)

Clean up resources associated with a configuration.
"""

function destroy_config(cfg::SHTnsConfig{T}) where T<:AbstractFloat
    empty!(cfg.fft_plans)
    cfg.gauss_weights = T[]
    cfg.gauss_nodes = T[]
    cfg.theta_grid = T[]
    cfg.phi_grid = T[]
    cfg.plm_cache = Matrix{T}(undef, 0, 0)
    cfg.lm_indices = Tuple{Int,Int}[]
    return nothing
end
