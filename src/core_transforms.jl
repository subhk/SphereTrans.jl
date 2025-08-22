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
    return create_config(T, lmax, mmax, mres; grid_type=grid_type, norm=norm)
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
    return set_grid_stable!(cfg, nlat, nphi)
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
    validate_config_stable(cfg)
    length(sh_coeffs) == cfg.nlm || throw(DimensionMismatch("sh_coeffs length $(length(sh_coeffs)) must equal nlm $(cfg.nlm)"))
    size(spatial_data) == (cfg.nlat, cfg.nphi) || throw(DimensionMismatch("spatial_data size $(size(spatial_data)) must be ($(cfg.nlat), $(cfg.nphi))"))
    
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
    validate_config_stable(cfg)
    size(spatial_data) == (cfg.nlat, cfg.nphi) || throw(DimensionMismatch("spatial_data size $(size(spatial_data)) must be ($(cfg.nlat), $(cfg.nphi))"))
    length(sh_coeffs) == cfg.nlm || throw(DimensionMismatch("sh_coeffs length $(length(sh_coeffs)) must equal nlm $(cfg.nlm)"))
    
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
    _build_m_coefficient_mapping(cfg::SHTnsConfig) -> Dict{Int, Vector{Int}}

Build a mapping from azimuthal mode m to coefficient indices for efficient lookup.
This avoids repeated enumeration in the synthesis inner loops.
"""
function _build_m_coefficient_mapping(cfg::SHTnsConfig)::Dict{Int, Vector{Int}}
    m_indices = Dict{Int, Vector{Int}}()
    
    @inbounds for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        if !haskey(m_indices, m)
            m_indices[m] = Int[]
        end
        push!(m_indices[m], coeff_idx)
    end
    
    return m_indices
end

"""
    _sh_to_spat_impl!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                     spatial_data::AbstractMatrix{T}) where T

Internal implementation of spherical harmonic synthesis using direct real approach.
"""
function _sh_to_spat_impl!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                          spatial_data::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Use pre-allocated workspace with type-stable access
    workspace_key = :workspace_fourier_coeffs
    if haskey(cfg.fft_plans, workspace_key)
        fourier_coeffs = cfg.fft_plans[workspace_key]::Matrix{Complex{T}}
        if size(fourier_coeffs) != (nlat, nphi_modes)
            fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
            cfg.fft_plans[workspace_key] = fourier_coeffs
        end
    else
        fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
        cfg.fft_plans[workspace_key] = fourier_coeffs
    end
    fill!(fourier_coeffs, zero(Complex{T}))
    
    # Pre-compute m-coefficient mapping with type-stable access
    mapping_key = :m_coefficient_mapping
    if haskey(cfg.fft_plans, mapping_key)
        m_indices = cfg.fft_plans[mapping_key]::Dict{Int, Vector{Int}}
    else
        m_indices = _build_m_coefficient_mapping(cfg)
        cfg.fft_plans[mapping_key] = m_indices
    end
    
    # For each azimuthal mode m (only m >= 0)
    @inbounds for m in 0:min(cfg.mmax, nphi÷2)
        # Skip if this m is not included due to mres
        (m == 0 || m % cfg.mres == 0) || continue
        
        # Get precomputed indices for this m
        coeff_indices = get(m_indices, m, Int[])
        isempty(coeff_indices) && continue
        
        # Direct computation without temporary array allocation with SIMD optimization
        m_col = m + 1  # Convert to 1-based indexing
        if m_col <= nphi_modes
            if m == 0
                # Optimized path for m=0 (no scaling needed)
                @inbounds @simd for i in 1:nlat
                    value = zero(Complex{T})
                    @simd for coeff_idx in coeff_indices
                        plm_val = cfg.plm_cache[i, coeff_idx]
                        coeff_val = sh_coeffs[coeff_idx]
                        value += coeff_val * plm_val
                    end
                    fourier_coeffs[i, m_col] = value
                end
            else
                # Optimized path for m>0 with precomputed scaling
                scale_factor = T(0.5)
                @inbounds @simd for i in 1:nlat
                    value = zero(Complex{T})
                    @simd for coeff_idx in coeff_indices
                        plm_val = cfg.plm_cache[i, coeff_idx]
                        coeff_val = sh_coeffs[coeff_idx]
                        value += (coeff_val * scale_factor) * plm_val
                    end
                    fourier_coeffs[i, m_col] = value
                end
            end
        end
    end
    
    # Transform from Fourier coefficients to spatial domain
    spatial_temp = compute_spatial_from_fourier(fourier_coeffs, cfg)
    
    # FFTW irfft scaling issue: need to multiply by nphi to get correct amplitude
    spatial_data .= spatial_temp .* T(nphi)
    
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
    
    # Pre-allocate workspace for mode extraction with type-stable access
    mode_workspace_key = :workspace_mode_data
    if haskey(cfg.fft_plans, mode_workspace_key)
        mode_data = cfg.fft_plans[mode_workspace_key]::Vector{Complex{T}}
        if length(mode_data) != nlat
            resize!(mode_data, nlat)
        end
    else
        mode_data = Vector{Complex{T}}(undef, nlat)
        cfg.fft_plans[mode_workspace_key] = mode_data
    end
    
    # Precompute normalization factor
    phi_normalization = T(2π) / nphi
    
    @inbounds for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        # Extract Fourier mode m
        if m <= nphi ÷ 2
            extract_fourier_mode!(fourier_coeffs, m, mode_data, nlat)
            
            # Integrate over latitude using Gaussian quadrature with SIMD
            integral = zero(Complex{T})
            @inbounds @simd for i in 1:nlat
                plm_val = cfg.plm_cache[i, coeff_idx]
                weight = cfg.gauss_weights[i]
                integral += mode_data[i] * plm_val * weight
            end
            
            # Apply proper normalization for φ integration  
            integral *= phi_normalization
            
            # For real fields, extract appropriate part with optimized conditionals
            if m == 0
                sh_coeffs[coeff_idx] = real(integral)
            else
                sh_coeffs[coeff_idx] = real(integral) * T(2)
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