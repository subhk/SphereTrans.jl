"""
Utility functions for spherical harmonic operations.
Includes allocation helpers, indexing utilities, and convenience functions.
"""

"""
    allocate_spectral(cfg::SHTnsConfig{T}) -> Vector{T}

Allocate array for real spherical harmonic coefficients.
"""
function allocate_spectral(cfg::SHTnsConfig{T}) where T
    return Vector{T}(undef, cfg.nlm)
end

"""
    allocate_spatial(cfg::SHTnsConfig{T}) -> Matrix{T}

Allocate array for real spatial field.
"""
function allocate_spatial(cfg::SHTnsConfig{T}) where T
    return Matrix{T}(undef, cfg.nlat, cfg.nphi)
end

"""
    lmidx(cfg::SHTnsConfig, l::Int, m::Int) -> Int

Get the linear index for spherical harmonic coefficient (l,m).
Returns 1-based index for Julia arrays.

# Arguments
- `cfg`: SHTns configuration
- `l`: Spherical harmonic degree
- `m`: Spherical harmonic order

# Returns
- Linear index into coefficient arrays (1-based)
"""
function lmidx(cfg::SHTnsConfig, l::Int, m::Int)
    0 <= l <= cfg.lmax || error("l must be in range [0, lmax]")
    abs(m) <= min(l, cfg.mmax) || error("m must be in range [-min(l,mmax), min(l,mmax)]")
    
    # Use cached lookup for O(1) access
    return find_plm_index(cfg, l, abs(m))
end

"""
    lm_from_index(cfg::SHTnsConfig, idx::Int) -> (Int, Int)

Get the (l,m) values for a given linear index.

# Arguments
- `cfg`: SHTns configuration  
- `idx`: Linear index (1-based)

# Returns
- `(l, m)`: Spherical harmonic degree and order
"""
function lm_from_index(cfg::SHTnsConfig, idx::Int)
    1 <= idx <= cfg.nlm || error("Index must be in range [1, nlm]")
    return cfg.lm_indices[idx]
end

"""
    power_spectrum(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) -> Vector{T}

Compute power spectrum from spherical harmonic coefficients.
Returns power as a function of degree l.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Spherical harmonic coefficients

# Returns
- `power`: Power spectrum P(l) for l = 0, 1, ..., lmax
"""
function power_spectrum(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    power = zeros(T, cfg.lmax + 1)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        coeff = sh_coeffs[idx]
        # For real spherical harmonics where only m ≥ 0 are stored,
        # each coefficient for m > 0 already represents both +m and -m modes
        # So no additional factor of 2 is needed
        power[l + 1] += coeff^2
    end
    
    return power
end

"""
    total_power(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) -> T

Compute total power (L2 norm squared) of spherical harmonic field.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Spherical harmonic coefficients

# Returns
- Total power of the field
"""
function total_power(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    return sum(power_spectrum(cfg, sh_coeffs))
end

"""
    rms_value(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) -> T

Compute RMS (root mean square) value of spherical harmonic field.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Spherical harmonic coefficients

# Returns
- RMS value of the field
"""
function rms_value(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    return sqrt(total_power(cfg, sh_coeffs) / (4π))
end

"""
    synthesize(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T

Convenience function for synthesis (same as sh_to_spat).
"""
function synthesize(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    return sh_to_spat(cfg, sh_coeffs)
end

"""
    analyze(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T

Convenience function for analysis (same as spat_to_sh).
"""
function analyze(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    return spat_to_sh(cfg, spatial_data)
end

"""
    synthesize!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}, 
               spatial_data::AbstractMatrix{T}) where T

Convenience function for in-place synthesis (same as sh_to_spat!).
"""
function synthesize!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                    spatial_data::AbstractMatrix{T}) where T
    return sh_to_spat!(cfg, sh_coeffs, spatial_data)
end

"""
    analyze!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
            sh_coeffs::AbstractVector{T}) where T

Convenience function for in-place analysis (same as spat_to_sh!).
"""
function analyze!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                 sh_coeffs::AbstractVector{T}) where T
    return spat_to_sh!(cfg, spatial_data, sh_coeffs)
end

"""
    create_test_field(cfg::SHTnsConfig{T}, l::Int, m::Int) -> Matrix{T}

Create a test field consisting of a single real spherical harmonic Y_l^m.
Useful for testing and validation.

# Arguments
- `cfg`: SHTns configuration
- `l`: Spherical harmonic degree
- `m`: Spherical harmonic order (must be ≥ 0 for real harmonics)

# Returns
- Real spatial field containing Y_l^m(θ,φ)
"""
function create_test_field(cfg::SHTnsConfig{T}, l::Int, m::Int) where T
    0 <= l <= cfg.lmax || error("l must be in range [0, lmax]")
    0 <= m <= min(l, cfg.mmax) || error("m must be in range [0, min(l,mmax)]")
    
    # Create coefficients with single mode
    sh_coeffs = zeros(T, cfg.nlm)
    idx = lmidx(cfg, l, m)
    sh_coeffs[idx] = one(T)
    
    # Synthesize to spatial domain
    return synthesize(cfg, sh_coeffs)
end

"""
    grid_latitudes(cfg::SHTnsConfig{T}) -> Vector{T}

Get latitude coordinates in degrees for the spatial grid.
Converts colatitude (θ) to latitude (90° - θ).

# Arguments
- `cfg`: SHTns configuration

# Returns
- Latitude coordinates in degrees [-90, 90]
"""
function grid_latitudes(cfg::SHTnsConfig{T}) where T
    return T[90 - rad2deg(theta) for theta in cfg.theta_grid]
end

"""
    grid_longitudes(cfg::SHTnsConfig{T}) -> Vector{T}

Get longitude coordinates in degrees for the spatial grid.

# Arguments
- `cfg`: SHTnsConfig configuration

# Returns
- Longitude coordinates in degrees [0, 360)
"""
function grid_longitudes(cfg::SHTnsConfig{T}) where T
    return T[rad2deg(phi) for phi in cfg.phi_grid]
end

"""
    evaluate_at_point(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                     theta::T, phi::T) -> T

Evaluate spherical harmonic field at a specific point.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Spherical harmonic coefficients
- `theta`: Colatitude in radians [0, π]
- `phi`: Longitude in radians [0, 2π]

# Returns
- Field value at the specified point
"""
function evaluate_at_point(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                          theta::T, phi::T) where T
    0 <= theta <= π || error("theta must be in range [0, π]")
    0 <= phi <= 2π || error("phi must be in range [0, 2π]")
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    cost = cos(theta)
    plm_values = compute_associated_legendre(cfg.lmax, cost, cfg.norm)
    
    result = zero(Complex{T})
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        plm_val = plm_values[idx]
        
        if m == 0
            result += sh_coeffs[idx] * plm_val
        else
            # For real harmonics, we need both cos(mφ) and sin(mφ) terms
            # This is simplified - full implementation needs proper handling
            phase = cis(m * phi)  # e^(imφ) = cos(mφ) + i*sin(mφ)
            result += sh_coeffs[idx] * plm_val * real(phase)
        end
    end
    
    return real(result)
end

"""
    transform_roundtrip_error(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) -> T

Compute roundtrip error for transform pair: spatial → spectral → spatial.
Useful for testing transform accuracy.

# Arguments
- `cfg`: SHTns configuration
- `spatial_data`: Input spatial field

# Returns
- Maximum absolute difference between input and roundtrip result
"""
function transform_roundtrip_error(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    # Forward transform
    sh_coeffs = analyze(cfg, spatial_data)
    
    # Backward transform
    reconstructed = synthesize(cfg, sh_coeffs)
    
    # Compute error
    return maximum(abs.(spatial_data .- reconstructed))
end

"""
    spectral_derivative_theta(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) -> Vector{T}

Compute θ-derivative of a scalar field in spectral domain.
Uses recurrence relations for associated Legendre polynomials.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Input spherical harmonic coefficients

# Returns
- Coefficients of ∂f/∂θ
"""
function spectral_derivative_theta(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    deriv_coeffs = zeros(T, cfg.nlm)
    
    # This is a placeholder - full implementation requires proper recurrence relations
    # for derivatives of associated Legendre polynomials
    @warn "spectral_derivative_theta is not fully implemented"
    
    return deriv_coeffs
end

"""
    spectral_derivative_phi(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) -> Vector{T}

Compute φ-derivative of a scalar field in spectral domain.
Simple multiplication by im for each m mode.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Input spherical harmonic coefficients

# Returns
- Coefficients of ∂f/∂φ
"""
function spectral_derivative_phi(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}) where T
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    
    deriv_coeffs = zeros(T, cfg.nlm)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if m > 0
            # ∂/∂φ Y_l^m = im * m * Y_l^m (for complex harmonics)
            # For real harmonics, this is more complex
            deriv_coeffs[idx] = m * sh_coeffs[idx]  # Simplified
        end
    end
    
    return deriv_coeffs
end

"""
    spatial_derivative_phi(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) -> Matrix{T}

Compute φ-derivative of a scalar field in the spatial domain using FFT in φ.
This is exact for the azimuthal direction given the grid and FFT plans.
"""
function spatial_derivative_phi(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T}) where T
    size(spatial_data) == (cfg.nlat, cfg.nphi) || error("spatial_data size mismatch")
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    # Forward FFT
    fourier = compute_fourier_coefficients_spatial(spatial_data, cfg)
    # Multiply by im*m for each mode
    for m in 0:(nphi_modes-1)
        factor = Complex{T}(0, m)
        @inbounds fourier[:, m+1] .= factor .* fourier[:, m+1]
    end
    # Inverse FFT to spatial
    return compute_spatial_from_fourier(fourier, cfg)
end

"""
    filter_spectral(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T}, 
                   l_cutoff::Int) -> Vector{T}

Apply spectral filtering by setting coefficients with l > l_cutoff to zero.

# Arguments
- `cfg`: SHTns configuration
- `sh_coeffs`: Input spherical harmonic coefficients
- `l_cutoff`: Maximum degree to retain

# Returns
- Filtered coefficients
"""
function filter_spectral(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                        l_cutoff::Int) where T
    length(sh_coeffs) == cfg.nlm || error("sh_coeffs length must equal nlm")
    0 <= l_cutoff <= cfg.lmax || error("l_cutoff must be in range [0, lmax]")
    
    filtered_coeffs = copy(sh_coeffs)
    
    for (idx, (l, m)) in enumerate(cfg.lm_indices)
        if l > l_cutoff
            filtered_coeffs[idx] = zero(T)
        end
    end
    
    return filtered_coeffs
end

# Internal: fast lookup for (l, m) -> plm column index using config-local cache
function find_plm_index(cfg::SHTnsConfig, l::Int, m::Int)::Int
    # Use config-local cache to avoid global dictionary overhead
    cache_key = :plm_index_cache
    cache = get!(cfg.fft_plans, cache_key) do
        d = Dict{Tuple{Int,Int}, Int}()
        @inbounds for (k, lm) in enumerate(cfg.lm_indices)
            d[lm] = k
        end
        d
    end::Dict{Tuple{Int,Int}, Int}
    
    idx = get(cache, (l, m), 0)
    idx != 0 && return idx
    error("plm index not found for (l=$(l), m=$(m))")
end

"""
    real_nlm(cfg) -> Int

Number of real-basis coefficients with explicit cos/sin for m>0 and single for m=0.
"""
function real_nlm(cfg::SHTnsConfig)::Int
    total = 0
    @inbounds for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m=0
        total += 1
        # m > 0 in steps of mres: 2 per m (cos/sin)
        total += 2 * (maxm ÷ cfg.mres)
    end
    return total
end

"""
    complex_to_real_coeffs(cfg, cplx::AbstractVector{Complex{T}}) -> Vector{T}

Convert canonical complex coefficients (±m) to real-basis cos/sin coefficients.
Assumes orthonormal normalization. For m>0: a_c = √2·Re(c_{l,m}), a_s = -√2·Im(c_{l,m}); m=0: a_0 = Re(c_{l,0}).
"""
function complex_to_real_coeffs(cfg::SHTnsConfig{T}, cplx::AbstractVector{Complex{T}}) where T
    length(cplx) == SHTnsKit._cplx_nlm(cfg) || error("complex coeff length mismatch")
    out = Vector{T}(undef, real_nlm(cfg))
    pos = 1
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m = 0
        # find complex index for (l,0)
        c0 = zero(Complex{T})
        for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
            if ll == l && mm == 0
                c0 = cplx[idx]
                break
            end
        end
        out[pos] = real(c0)
        pos += 1
        # m = mres, 2mres, ..., maxm
        for m in cfg.mres:cfg.mres:maxm
            cm = zero(Complex{T})
            for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
                if ll == l && mm == m
                    cm = cplx[idx]
                    break
                end
            end
            out[pos] = sqrt(T(2)) * real(cm); pos += 1
            out[pos] = -sqrt(T(2)) * imag(cm); pos += 1
        end
    end
    return out
end

"""
    real_to_complex_coeffs(cfg, real::AbstractVector{T}) -> Vector{Complex{T}}

Convert real-basis cos/sin coefficients to canonical complex coefficients (±m).
For m>0: c_{l,m} = (a_c - i a_s)/√2, c_{l,-m} will be implied by real field on synthesis.
"""
function real_to_complex_coeffs(cfg::SHTnsConfig{T}, real::AbstractVector{T}) where T
    length(real) == real_nlm(cfg) || error("real coeff length mismatch")
    cplx = Vector{Complex{T}}(undef, SHTnsKit._cplx_nlm(cfg))
    fill!(cplx, zero(Complex{T}))
    pos = 1
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m = 0
        a0 = real[pos]; pos += 1
        for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
            if ll == l && mm == 0
                cplx[idx] = Complex{T}(a0, 0)
                break
            end
        end
        for m in cfg.mres:cfg.mres:maxm
            ac = real[pos]; as = real[pos+1]; pos += 2
            cm = (ac - im*as)/sqrt(T(2))
            for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
                if ll == l && mm == m
                    cplx[idx] = cm
                    break
                end
            end
        end
    end
    return cplx
end

"""
    analyze_real(cfg, spatial::AbstractMatrix{T}) -> Vector{T}

Analyze real spatial field to real-basis coefficients via complex route.
"""
function analyze_real(cfg::SHTnsConfig{T}, spatial::AbstractMatrix{T}) where T
    complex_spatial = Complex{T}.(spatial)
    c = SHTnsKit.cplx_spat_to_sh(cfg, complex_spatial)
    return complex_to_real_coeffs(cfg, c)
end

"""
    synthesize_real(cfg, real_coeffs::AbstractVector{T}) -> Matrix{T}

Synthesize real spatial field from real-basis coefficients via complex route.
"""
function synthesize_real(cfg::SHTnsConfig{T}, real_coeffs::AbstractVector{T}) where T
    c = real_to_complex_coeffs(cfg, real_coeffs)
    spat_c = SHTnsKit.cplx_sh_to_spat(cfg, c)
    return real.(spat_c)
end
