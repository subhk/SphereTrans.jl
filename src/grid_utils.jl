"""
Grid utilities for spherical harmonic transforms.
Functions for working with spherical coordinate grids and transformations.
"""

"""
    create_latitude_weights(cfg::SHTnsConfig{T}) -> Vector{T}

Create latitude-dependent weights for spatial integration.
These are the sine-weighted differential area elements: sin(θ) dθ dφ.

# Arguments
- `cfg`: SHTns configuration

# Returns
- Vector of weights for each latitude (length nlat)
"""
function create_latitude_weights(cfg::SHTnsConfig{T}) where T
    weights = Vector{T}(undef, cfg.nlat)
    
    for i in 1:cfg.nlat
        theta = cfg.theta_grid[i]
        sint = sin(theta)
        
        if cfg.grid_type == SHT_GAUSS
            # For Gauss grids, use the quadrature weights
            weights[i] = cfg.gauss_weights[i] * sint
        else
            # For regular grids, use uniform weights
            dtheta = π / cfg.nlat
            dphi = 2π / cfg.nphi
            weights[i] = sint * dtheta * dphi
        end
    end
    
    return weights
end

"""
    spatial_integral(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) -> T

Compute spatial integral of a field over the sphere using appropriate quadrature.

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field to integrate (nlat × nphi)

# Returns
- Integral value ∫∫ f(θ,φ) sin(θ) dθ dφ
"""
function spatial_integral(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch")
    
    lat_weights = create_latitude_weights(cfg)
    integral = zero(T)
    
    for i in 1:cfg.nlat
        for j in 1:cfg.nphi
            integral += spatial_data[i, j] * lat_weights[i]
        end
    end
    
    return integral
end

"""
    spatial_mean(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) -> T

Compute area-weighted mean of a field over the sphere.

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field (nlat × nphi)

# Returns  
- Area-weighted mean value
"""
function spatial_mean(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    integral = spatial_integral(cfg, spatial_data)
    return integral / (4π)
end

"""
    spatial_variance(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) -> T

Compute area-weighted variance of a field over the sphere.

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field (nlat × nphi)

# Returns
- Area-weighted variance
"""
function spatial_variance(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    mean_val = spatial_mean(cfg, spatial_data)
    deviation_squared = (spatial_data .- mean_val).^2
    return spatial_mean(cfg, deviation_squared)
end

"""
    spatial_divergence(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) -> Matrix{T}

Compute the horizontal divergence of a tangential vector field on the unit sphere.
Uses exact FFT in φ and centered finite differences in θ.
Formula: div = (1/sinθ) ∂(sinθ u_θ)/∂θ + (1/sinθ) ∂u_φ/∂φ.
"""
function spatial_divergence(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    size(u_theta) == (nlat, nphi) || error("u_theta size mismatch")
    size(u_phi) == (nlat, nphi) || error("u_phi size mismatch")
    # φ-derivative via FFT
    duphi_dphi = spatial_derivative_phi(cfg, u_phi)
    # θ-derivative via centered differences on sinθ uθ
    result = Matrix{T}(undef, nlat, nphi)
    for i in 1:nlat
        θ = cfg.theta_grid[i]
        sθ = sin(θ)
        inv_sθ = sθ > 1e-12 ? one(T)/sθ : zero(T)
        for j in 1:nphi
            if 1 < i < nlat
                d_suθ_dθ = ((sin(cfg.theta_grid[i+1]) * u_theta[i+1, j]) - (sin(cfg.theta_grid[i-1]) * u_theta[i-1, j])) /
                            (cfg.theta_grid[i+1] - cfg.theta_grid[i-1])
            elseif i == 1
                d_suθ_dθ = ((sin(cfg.theta_grid[i+1]) * u_theta[i+1, j]) - (sin(cfg.theta_grid[i]) * u_theta[i, j])) /
                            (cfg.theta_grid[i+1] - cfg.theta_grid[i])
            else
                d_suθ_dθ = ((sin(cfg.theta_grid[i]) * u_theta[i, j]) - (sin(cfg.theta_grid[i-1]) * u_theta[i-1, j])) /
                            (cfg.theta_grid[i] - cfg.theta_grid[i-1])
            end
            result[i, j] = inv_sθ * (d_suθ_dθ + duphi_dphi[i, j])
        end
    end
    return result
end

"""
    spatial_vorticity(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) -> Matrix{T}

Compute the vertical component (radial) of curl for a tangential vector field on the unit sphere:
ζ = (1/sinθ) ∂(u_φ sinθ)/∂θ - (1/sinθ) ∂u_θ/∂φ.
"""
function spatial_vorticity(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    size(u_theta) == (nlat, nphi) || error("u_theta size mismatch")
    size(u_phi) == (nlat, nphi) || error("u_phi size mismatch")
    # φ-derivative via FFT
    dutheta_dphi = spatial_derivative_phi(cfg, u_theta)
    # θ-derivative via centered differences on uφ sinθ
    result = Matrix{T}(undef, nlat, nphi)
    for i in 1:nlat
        θ = cfg.theta_grid[i]
        sθ = sin(θ)
        inv_sθ = sθ > 1e-12 ? one(T)/sθ : zero(T)
        for j in 1:nphi
            if 1 < i < nlat
                d_uφs_dθ = ((u_phi[i+1, j] * sin(cfg.theta_grid[i+1])) - (u_phi[i-1, j] * sin(cfg.theta_grid[i-1]))) /
                           (cfg.theta_grid[i+1] - cfg.theta_grid[i-1])
            elseif i == 1
                d_uφs_dθ = ((u_phi[i+1, j] * sin(cfg.theta_grid[i+1])) - (u_phi[i, j] * sin(cfg.theta_grid[i]))) /
                           (cfg.theta_grid[i+1] - cfg.theta_grid[i])
            else
                d_uφs_dθ = ((u_phi[i, j] * sin(cfg.theta_grid[i])) - (u_phi[i-1, j] * sin(cfg.theta_grid[i-1]))) /
                           (cfg.theta_grid[i] - cfg.theta_grid[i-1])
            end
            result[i, j] = inv_sθ * (d_uφs_dθ - dutheta_dphi[i, j])
        end
    end
    return result
end

"""
    create_coordinate_matrices(cfg::SHTnsConfig{T}) -> (Matrix{T}, Matrix{T})

Create matrices of theta and phi coordinates for each grid point.

# Arguments
- `cfg`: SHTns configuration

# Returns
- `(theta_matrix, phi_matrix)`: Coordinate matrices (nlat × nphi)
"""
function create_coordinate_matrices(cfg::SHTnsConfig{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    theta_matrix = Matrix{T}(undef, nlat, nphi)
    phi_matrix = Matrix{T}(undef, nlat, nphi)
    
    for i in 1:nlat
        for j in 1:nphi
            theta_matrix[i, j] = cfg.theta_grid[i]
            phi_matrix[i, j] = cfg.phi_grid[j]
        end
    end
    
    return theta_matrix, phi_matrix
end

"""
    create_cartesian_coordinates(cfg::SHTnsConfig{T}) -> (Matrix{T}, Matrix{T}, Matrix{T})

Create Cartesian coordinate matrices (x, y, z) for unit sphere.

# Arguments
- `cfg`: SHTns configuration

# Returns
- `(x, y, z)`: Cartesian coordinate matrices (nlat × nphi)
"""
function create_cartesian_coordinates(cfg::SHTnsConfig{T}) where T
    theta_matrix, phi_matrix = create_coordinate_matrices(cfg)
    
    sint = sin.(theta_matrix)
    cost = cos.(theta_matrix)
    sinp = sin.(phi_matrix)
    cosp = cos.(phi_matrix)
    
    x = sint .* cosp
    y = sint .* sinp
    z = cost
    
    return x, y, z
end

"""
    interpolate_to_point(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                        theta_target::T, phi_target::T) -> T

Interpolate spatial field to a target point using bilinear interpolation.

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field (nlat × nphi)
- `theta_target`: Target colatitude [0, π]
- `phi_target`: Target longitude [0, 2π]

# Returns
- Interpolated field value
"""
function interpolate_to_point(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                             theta_target::T, phi_target::T) where T
    0 <= theta_target <= π || error("theta_target must be in [0, π]")
    0 <= phi_target <= 2π || error("phi_target must be in [0, 2π]")
    
    # Find surrounding grid points
    # Theta direction
    i_low = 1
    for i in 1:(cfg.nlat-1)
        if cfg.theta_grid[i] <= theta_target <= cfg.theta_grid[i+1]
            i_low = i
            break
        end
    end
    i_high = min(i_low + 1, cfg.nlat)
    
    # Phi direction (with periodic boundary conditions)
    phi_normalized = mod(phi_target, 2π)
    j_low = 1
    for j in 1:(cfg.nphi-1)
        if cfg.phi_grid[j] <= phi_normalized < cfg.phi_grid[j+1]
            j_low = j
            break
        end
    end
    j_high = j_low == cfg.nphi ? 1 : j_low + 1  # Wrap around
    
    # Bilinear interpolation weights
    if i_high > i_low
        w_theta = (theta_target - cfg.theta_grid[i_low]) / (cfg.theta_grid[i_high] - cfg.theta_grid[i_low])
    else
        w_theta = zero(T)
    end
    
    if j_high != j_low
        dphi = cfg.phi_grid[j_high] - cfg.phi_grid[j_low]
        if dphi < 0  # Wrap-around case
            dphi += 2π
            phi_diff = phi_normalized - cfg.phi_grid[j_low]
            if phi_diff < 0
                phi_diff += 2π
            end
        else
            phi_diff = phi_normalized - cfg.phi_grid[j_low]
        end
        w_phi = phi_diff / dphi
    else
        w_phi = zero(T)
    end
    
    # Interpolate
    f11 = spatial_data[i_low, j_low]
    f12 = spatial_data[i_low, j_high]
    f21 = spatial_data[i_high, j_low]
    f22 = spatial_data[i_high, j_high]
    
    f1 = f11 * (1 - w_phi) + f12 * w_phi
    f2 = f21 * (1 - w_phi) + f22 * w_phi
    
    result = f1 * (1 - w_theta) + f2 * w_theta
    
    return result
end

"""
    interpolate_to_point(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}},
                         theta_target::T, phi_target::T) -> Complex{T}

Bilinear interpolation for complex spatial fields. Interpolates real and imaginary
parts independently using the real-valued interpolator.
"""
function interpolate_to_point(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{Complex{T}},
                             theta_target::T, phi_target::T) where T
    re = interpolate_to_point(cfg, real.(spatial_data), theta_target, phi_target)
    im = interpolate_to_point(cfg, imag.(spatial_data), theta_target, phi_target)
    return Complex{T}(re, im)
end

"""
    extract_latitude_slice(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                          latitude_deg::T) -> Vector{T}

Extract data along a constant latitude (in degrees).

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field (nlat × nphi)
- `latitude_deg`: Latitude in degrees [-90, 90]

# Returns
- Values along the specified latitude (length nphi)
"""
function extract_latitude_slice(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                               latitude_deg::T) where T
    -90 <= latitude_deg <= 90 || error("latitude_deg must be in [-90, 90]")
    
    # Convert to colatitude
    theta_target = π/2 - deg2rad(latitude_deg)
    
    # Find nearest grid latitude
    i_nearest = 1
    min_diff = abs(cfg.theta_grid[1] - theta_target)
    for i in 2:cfg.nlat
        diff = abs(cfg.theta_grid[i] - theta_target)
        if diff < min_diff
            min_diff = diff
            i_nearest = i
        end
    end
    
    return spatial_data[i_nearest, :]
end

"""
    extract_longitude_slice(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                           longitude_deg::T) -> Vector{T}

Extract data along a constant longitude (in degrees).

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Spatial field (nlat × nphi)
- `longitude_deg`: Longitude in degrees [0, 360]

# Returns
- Values along the specified longitude (length nlat)
"""
function extract_longitude_slice(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                                longitude_deg::T) where T
    0 <= longitude_deg <= 360 || error("longitude_deg must be in [0, 360]")
    
    # Convert to radians and normalize
    phi_target = deg2rad(mod(longitude_deg, 360))
    
    # Find nearest grid longitude
    j_nearest = 1
    min_diff = abs(cfg.phi_grid[1] - phi_target)
    for j in 2:cfg.nphi
        diff = abs(cfg.phi_grid[j] - phi_target)
        if diff < min_diff
            min_diff = diff
            j_nearest = j
        end
    end
    
    return spatial_data[:, j_nearest]
end

"""
    regrid_to_regular(cfg_in::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                     nlat_out::Int, nphi_out::Int) -> Matrix{T}

Regrid spatial data to a regular lat-lon grid using spectral interpolation.

# Arguments
- `cfg_in`: Input SHTns configuration
- `spatial_data`: Input spatial field
- `nlat_out`: Output number of latitudes
- `nphi_out`: Output number of longitudes

# Returns
- Regridded data on regular grid (nlat_out × nphi_out)
"""
function regrid_to_regular(cfg_in::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                          nlat_out::Int, nphi_out::Int) where T
    # Transform to spectral domain
    sh_coeffs = analyze(cfg_in, spatial_data)
    
    # Create output configuration with regular grid
    cfg_out = create_regular_config(cfg_in.lmax, cfg_in.mmax; T=T)
    set_grid!(cfg_out, nlat_out, nphi_out)
    
    # Transform to output grid
    output_data = synthesize(cfg_out, sh_coeffs)
    
    # Clean up
    destroy_config(cfg_out)
    
    return output_data
end

"""
    compute_grid_spacing(cfg::SHTnsConfig{T}) -> (T, T)

Compute average grid spacing in theta and phi directions.

# Arguments
- `cfg`: SHTns configuration

# Returns
- `(dtheta, dphi)`: Average spacing in radians
"""
function compute_grid_spacing(cfg::SHTnsConfig{T}) where T
    if cfg.grid_type == SHT_GAUSS
        # For Gauss grids, spacing is non-uniform
        # Compute average from actual grid points
        dtheta_avg = π / cfg.nlat  # Approximate
    else
        # For regular grids
        dtheta_avg = π / cfg.nlat
    end
    
    dphi_avg = 2π / cfg.nphi
    
    return dtheta_avg, dphi_avg
end

"""
    create_test_data_gaussian(cfg::SHTnsConfig{T}, amplitude::T, 
                             center_lat::T, center_lon::T, width::T) -> Matrix{T}

Create test data with a Gaussian blob at specified location.

# Arguments
- `cfg`: SHTns configuration
- `amplitude`: Peak amplitude
- `center_lat`: Center latitude in degrees [-90, 90]
- `center_lon`: Center longitude in degrees [0, 360]
- `width`: Gaussian width in degrees

# Returns
- Spatial field with Gaussian blob (nlat × nphi)
"""
function create_test_data_gaussian(cfg::SHTnsConfig{T}, amplitude::T,
                                  center_lat::T, center_lon::T, width::T) where T
    # Convert to spherical coordinates
    theta_center = π/2 - deg2rad(center_lat)
    phi_center = deg2rad(center_lon)
    width_rad = deg2rad(width)
    
    # Create coordinate matrices
    theta_matrix, phi_matrix = create_coordinate_matrices(cfg)
    
    # Compute angular distances (great circle distance)
    cost_center = cos(theta_center)
    sint_center = sin(theta_center)
    cosp_center = cos(phi_center)
    sinp_center = sin(phi_center)
    
    result = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    
    for i in 1:cfg.nlat
        theta = theta_matrix[i, 1]  # Same for all longitudes
        cost = cos(theta)
        sint = sin(theta)
        
        for j in 1:cfg.nphi
            phi = phi_matrix[i, j]
            cosp = cos(phi)
            sinp = sin(phi)
            
            # Great circle distance
            cos_dist = cost_center * cost + sint_center * sint * (cosp_center * cosp + sinp_center * sinp)
            cos_dist = clamp(cos_dist, -1, 1)  # Handle numerical errors
            angular_dist = acos(cos_dist)
            
            # Gaussian blob
            result[i, j] = amplitude * exp(-(angular_dist / width_rad)^2)
        end
    end
    
    return result
end
