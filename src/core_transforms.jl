"""
Core spherical harmonic transform functions.
Implements the fundamental scalar transforms between spectral and spatial domains.
"""

"""
    create_config(lmax::Int, mmax::Int=lmax, mres::Int=1; 
                 grid_type::SHTnsGrid=SHT_GAUSS, 
                 norm::SHTnsNorm=SHT_ORTHONORMAL,
                 T::Type=Float64) -> SHTnsConfig{T}

Create a new spherical harmonic transform configuration.

# Arguments
- `lmax`: Maximum spherical harmonic degree
- `mmax`: Maximum spherical harmonic order (default: lmax)
- `mres`: Azimuthal resolution parameter (default: 1)
- `grid_type`: Type of spatial grid (default: Gauss-Legendre)
- `norm`: Normalization convention (default: orthonormal)
- `T`: Floating point precision (default: Float64)

# Returns
- `SHTnsConfig{T}`: Configured transform object
"""
function create_config(lmax::Int, mmax::Int=lmax, mres::Int=1;
                      grid_type::SHTnsGrid=SHT_GAUSS,
                      norm::SHTnsNorm=SHT_ORTHONORMAL,
                      T::Type=Float64)
    # Input validation
    lmax >= 0 || error("lmax must be non-negative")
    mmax >= 0 || error("mmax must be non-negative")
    mmax <= lmax || error("mmax must not exceed lmax")
    mres >= 1 || error("mres must be positive")
    
    cfg = SHTnsConfig{T}()
    cfg.lmax = lmax
    cfg.mmax = mmax
    cfg.mres = mres
    cfg.grid_type = grid_type
    cfg.norm = norm
    
    # Calculate number of spectral coefficients
    cfg.nlm = nlm_calc(lmax, mmax, mres)
    
    # Create (l,m) index mapping
    cfg.lm_indices = Tuple{Int,Int}[]
    for l in 0:lmax
        for m in 0:min(l, mmax)
            if m % mres == 0 || m == 0
                push!(cfg.lm_indices, (l, m))
            end
        end
    end
    
    length(cfg.lm_indices) == cfg.nlm || error("Inconsistent nlm calculation")
    
    return cfg
end

"""
    set_grid!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T

Initialize the spatial grid for transforms.
This precomputes all grid-dependent quantities.

# Arguments
- `cfg`: SHTns configuration to modify
- `nlat`: Number of latitude points
- `nphi`: Number of longitude points
"""
function set_grid!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T
    # Validation
    nlat > 0 || error("nlat must be positive")
    nphi > 0 || error("nphi must be positive")
    
    if cfg.grid_type == SHT_GAUSS
        nlat > cfg.lmax || error("For Gauss grid: nlat must be > lmax")
    else
        nlat >= 2*cfg.lmax + 1 || error("For regular grid: nlat must be >= 2*lmax + 1")
    end
    
    nphi >= 2*cfg.mmax + 1 || error("nphi must be >= 2*mmax + 1")
    
    cfg.nlat = nlat
    cfg.nphi = nphi
    
    # Setup spatial grid coordinates
    if cfg.grid_type == SHT_GAUSS
        # Gauss-Legendre nodes and weights
        nodes, weights = compute_gauss_legendre_nodes_weights(nlat, T)
        cfg.gauss_nodes = nodes          # cos(θ) values
        cfg.gauss_weights = weights      # quadrature weights
        cfg.theta_grid = acos.(nodes)    # θ = acos(cos(θ))
    else
        # Regular equiangular grid
        cfg.theta_grid = T[π * (i - 0.5) / nlat for i in 1:nlat]
        cfg.gauss_nodes = cos.(cfg.theta_grid)
        cfg.gauss_weights = ones(T, nlat) * (T(2) / nlat)  # Uniform weights
    end
    
    # Longitude grid (always equispaced)
    cfg.phi_grid = T[2π * (j - 1) / nphi for j in 1:nphi]
    
    # Precompute Legendre polynomials for all grid points
    cfg.plm_cache = Matrix{T}(undef, nlat, cfg.nlm)
    for (i, theta) in enumerate(cfg.theta_grid)
        cost = cos(theta)
        plm_values = compute_associated_legendre(cfg.lmax, cost, cfg.norm)
        cfg.plm_cache[i, :] = plm_values
    end
    
    # Setup FFT plans
    setup_fft_plans!(cfg)
    
    return cfg
end

"""
    destroy_config(cfg::SHTnsConfig)

Clean up resources associated with a configuration.
"""
function destroy_config(cfg::SHTnsConfig)
    # Clear cached data
    empty!(cfg.gauss_weights)
    empty!(cfg.gauss_nodes)
    empty!(cfg.theta_grid)
    empty!(cfg.phi_grid)
    empty!(cfg.lm_indices)
    empty!(cfg.fft_plans)
    cfg.plm_cache = Matrix{eltype(cfg.plm_cache)}(undef, 0, 0)
    return nothing
end

"""
    sh_to_spat!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}, 
               spatial_data::AbstractMatrix{T}) where T

Transform spherical harmonic coefficients to spatial grid (synthesis).
This is the fundamental backward transform: spectral → spatial.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Input spherical harmonic coefficients (length nlm)
- `spatial_data`: Output spatial field (nlat × nphi, pre-allocated)
"""
function sh_to_spat!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                    spatial_data::AbstractMatrix{T}) where T
    validate_config(cfg)
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch")
    
    lock(cfg.lock) do
        _sh_to_spat_impl!(cfg, sh_coeffs, spatial_data)
    end
    
    return spatial_data
end

"""
    spat_to_sh!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
               sh_coeffs::AbstractVector{T}) where T

Transform spatial grid to spherical harmonic coefficients (analysis).
This is the fundamental forward transform: spatial → spectral.

# Arguments
- `cfg`: SHTns configuration  
- `spatial_data`: Input spatial field (nlat × nphi)
- `sh_coeffs`: Output spherical harmonic coefficients (length nlm, pre-allocated)
"""
function spat_to_sh!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                    sh_coeffs::AbstractVector{T}) where T
    validate_config(cfg)
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch") 
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    lock(cfg.lock) do
        _spat_to_sh_impl!(cfg, spatial_data, sh_coeffs)
    end
    
    return sh_coeffs
end

# Non-mutating versions that allocate output
"""
    sh_to_spat(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T

Transform spherical harmonic coefficients to spatial grid (allocating version).
"""
function sh_to_spat(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    spatial_data = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    return sh_to_spat!(cfg, sh_coeffs, spatial_data)
end

"""
    spat_to_sh(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T

Transform spatial grid to spherical harmonic coefficients (allocating version).
"""  
function spat_to_sh(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    sh_coeffs = Vector{T}(undef, cfg.nlm)
    return spat_to_sh!(cfg, spatial_data, sh_coeffs)
end

# Implementation functions (internal)

"""
    _sh_to_spat_impl!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                     spatial_data::AbstractMatrix{T}) where T

Internal implementation of spherical harmonic synthesis using direct real approach.
"""
function _sh_to_spat_impl!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                          spatial_data::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Allocate working arrays for Fourier coefficients
    nphi_modes = nphi ÷ 2 + 1
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    fill!(fourier_coeffs, zero(Complex{T}))
    
    # For each azimuthal mode m (only m >= 0)
    for m in 0:min(cfg.mmax, nphi÷2)
        # Skip if this m is not included due to mres
        (m == 0 || m % cfg.mres == 0) || continue
        
        # Collect all (l,m) coefficients for this m
        mode_coeffs = Vector{Complex{T}}(undef, nlat)
        fill!(mode_coeffs, zero(Complex{T}))
        
        # Sum over l for this m: Σ_l c_{l,m} P_l^m(cos θ)
        for i in 1:nlat
            value = zero(Complex{T})
            for (coeff_idx, (l, m_coeff)) in enumerate(cfg.lm_indices)
                if m_coeff == m
                    # Get Legendre polynomial value
                    plm_val = cfg.plm_cache[i, coeff_idx]
                    coeff_val = sh_coeffs[coeff_idx]
                    
                    # For synthesis, we need to convert back from the analysis representation
                    if m == 0
                        value += coeff_val * plm_val
                    else
                        # For m > 0, we stored the coefficient with factor of 2
                        # So we need to divide by 2 to get back the complex amplitude
                        value += (coeff_val / 2) * plm_val
                    end
                end
            end
            mode_coeffs[i] = value
        end
        
        # Store in Fourier coefficient array
        insert_fourier_mode!(fourier_coeffs, m, mode_coeffs, nlat)
    end
    
    # Transform from Fourier coefficients to spatial domain
    spatial_data .= compute_spatial_from_fourier(fourier_coeffs, cfg)
    
    return nothing
end

"""
    _spat_to_sh_impl!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                     sh_coeffs::AbstractVector{T}) where T

Internal implementation of spherical harmonic analysis using direct real approach.
"""
function _spat_to_sh_impl!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                          sh_coeffs::AbstractVector{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Transform spatial data to Fourier coefficients in longitude
    fourier_coeffs = compute_fourier_coefficients_spatial(spatial_data, cfg)
    
    # For each (l,m) coefficient (only m >= 0 stored)
    fill!(sh_coeffs, zero(T))
    
    for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        # Extract Fourier mode m
        if m <= nphi ÷ 2
            mode_data = Vector{Complex{T}}(undef, nlat)
            extract_fourier_mode!(fourier_coeffs, m, mode_data, nlat)
            
            # Integrate over latitude using Gaussian quadrature
            integral = zero(Complex{T})
            for i in 1:nlat
                plm_val = cfg.plm_cache[i, coeff_idx]
                weight = cfg.gauss_weights[i]
                integral += mode_data[i] * plm_val * weight
            end
            
            # Apply proper normalization for φ integration
            # FFT gives sum over nphi points, but we need integral over [0,2π]
            phi_normalization = T(2π) / nphi
            integral *= phi_normalization
            
            # For real fields, extract appropriate part
            if m == 0
                # m=0: coefficient is real
                sh_coeffs[coeff_idx] = real(integral)
            else
                # m>0: for real fields, use real part and account for m>0 normalization
                # The factor of 2 accounts for the ±m symmetry in real representation
                sh_coeffs[coeff_idx] = real(integral) * 2
            end
        end
    end
    
    return nothing
end

"""
    get_lmax(cfg::SHTnsConfig) -> Int

Get the maximum spherical harmonic degree.
"""
get_lmax(cfg::SHTnsConfig) = cfg.lmax

"""
    get_mmax(cfg::SHTnsConfig) -> Int  

Get the maximum spherical harmonic order.
"""
get_mmax(cfg::SHTnsConfig) = cfg.mmax

"""
    get_nlat(cfg::SHTnsConfig) -> Int

Get the number of latitude grid points.
"""
get_nlat(cfg::SHTnsConfig) = cfg.nlat

"""
    get_nphi(cfg::SHTnsConfig) -> Int

Get the number of longitude grid points.
"""
get_nphi(cfg::SHTnsConfig) = cfg.nphi

"""
    get_nlm(cfg::SHTnsConfig) -> Int

Get the number of spherical harmonic coefficients.
"""
get_nlm(cfg::SHTnsConfig) = cfg.nlm

"""
    get_theta(cfg::SHTnsConfig, i::Int) -> T

Get the colatitude (theta) coordinate of grid point i.
"""
function get_theta(cfg::SHTnsConfig{T}, i::Int) where T
    1 <= i <= cfg.nlat || error("Grid point index out of range")
    return cfg.theta_grid[i]
end

"""
    get_phi(cfg::SHTnsConfig, j::Int) -> T

Get the longitude (phi) coordinate of grid point j.
"""
function get_phi(cfg::SHTnsConfig{T}, j::Int) where T
    1 <= j <= cfg.nphi || error("Grid point index out of range")
    return cfg.phi_grid[j]
end

"""
    get_gauss_weights(cfg::SHTnsConfig) -> Vector{T}

Get the Gaussian quadrature weights.
"""
get_gauss_weights(cfg::SHTnsConfig) = cfg.gauss_weights

# Convenience functions for grid setup

"""
    create_gauss_config(lmax::Int, mmax::Int=lmax; T::Type=Float64) -> SHTnsConfig{T}

Create a configuration with Gauss-Legendre grid.
Automatically sets nlat = lmax + 1, nphi = 2*mmax + 1.
"""
function create_gauss_config(lmax::Int, mmax::Int=lmax; T::Type=Float64)
    cfg = create_config(lmax, mmax, 1; grid_type=SHT_GAUSS, T=T)
    nlat = max(lmax + 1, 16)  # Minimum of 16 for stability
    nphi = max(2*mmax + 1, 17)  # Odd number for efficiency
    set_grid!(cfg, nlat, nphi)
    return cfg
end

"""
    create_regular_config(lmax::Int, mmax::Int=lmax; T::Type=Float64) -> SHTnsConfig{T}

Create a configuration with regular equiangular grid.
Automatically sets nlat = 2*lmax + 1, nphi = 2*mmax + 1.
"""
function create_regular_config(lmax::Int, mmax::Int=lmax; T::Type=Float64)
    cfg = create_config(lmax, mmax, 1; grid_type=SHT_REGULAR, T=T)
    nlat = 2*lmax + 1
    nphi = 2*mmax + 1
    set_grid!(cfg, nlat, nphi)
    return cfg
end