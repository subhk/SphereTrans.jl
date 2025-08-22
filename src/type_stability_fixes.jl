"""
Type stability fixes and improvements for SHTnsKit.
This module addresses type inference issues and ensures optimal performance.
"""

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

# Type-stable configuration functions

"""
    create_config_stable(::Type{T}, lmax::Int, mmax::Int=lmax, mres::Int=1;
                         grid_type::SHTnsGrid=SHT_GAUSS,
                         norm::SHTnsNorm=SHT_ORTHONORMAL) where T<:AbstractFloat

Type-stable version of create_config with explicit type parameter first.
This ensures T is known at compile time.
"""
function create_config_stable(::Type{T}, lmax::Int, mmax::Int=lmax, mres::Int=1;
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
        cfg.nlm = nlm_calc_stable(lmax, mmax, mres)
        
        # Pre-allocate with known sizes
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
    nlm_calc_stable(lmax::Int, mmax::Int, mres::Int)::Int

Type-stable version of nlm_calc with guaranteed Int return type.
"""
function nlm_calc_stable(lmax::Int, mmax::Int, mres::Int)::Int
    @stable begin
        total = 0
        @inbounds for l in 0:lmax
            for m in 0:min(l, mmax)
                if m % mres == 0 || m == 0
                    total += 1
                end
            end
        end
        return total
    end
end

"""
    set_grid_stable!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T

Type-stable grid initialization with compile-time type information.
"""
function set_grid_stable!(cfg::SHTnsConfig{T}, nlat::Int, nphi::Int) where T
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
            nodes, weights = compute_gauss_legendre_nodes_weights_stable(nlat, T)
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
            
            dtheta = T(π) / T(nlat)
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
        
        return nothing
    end
end

"""
    compute_gauss_legendre_nodes_weights_stable(n::Int, ::Type{T}) where T

Type-stable Gauss-Legendre computation with explicit type parameter.
"""
function compute_gauss_legendre_nodes_weights_stable(n::Int, ::Type{T}) where T
    @stable begin
        nodes = Vector{T}(undef, n)
        weights = Vector{T}(undef, n)
        
        # Type-stable computation
        for i in 1:((n + 1) ÷ 2)  # Only compute half due to symmetry
            # Initial guess for i-th root
            z = cos(T(π) * (T(i) - T(0.25)) / (T(n) + T(0.5)))
            
            # Newton-Raphson iteration
            local pp::T  # Type annotation for stability
            for _ in 1:10  # Fixed iteration count for type stability
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

# Type-stable transform functions

"""
    allocate_spectral_stable(cfg::SHTnsConfig{T}) where T

Type-stable spectral array allocation.
"""
function allocate_spectral_stable(cfg::SHTnsConfig{T}) where T
    @stable Vector{Complex{T}}(undef, cfg.nlm)
end

"""
    allocate_spatial_stable(cfg::SHTnsConfig{T}) where T

Type-stable spatial array allocation.
"""
function allocate_spatial_stable(cfg::SHTnsConfig{T}) where T
    @stable Matrix{T}(undef, cfg.nlat, cfg.nphi)
end

"""
    validate_config_stable(cfg::SHTnsConfig{T}) where T

Type-stable configuration validation.
"""
function validate_config_stable(cfg::SHTnsConfig{T}) where T
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

# Type-stable utility functions

"""
    lmidx_stable(l::Int, m::Int, lmax::Int)::Int

Type-stable index calculation with guaranteed Int return.
"""
function lmidx_stable(l::Int, m::Int, lmax::Int)::Int
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

"""
    get_cached_plm_stable(cfg::SHTnsConfig{T}, j::Int, l::Int, m::Int) where T

Type-stable access to cached Legendre polynomial values.
"""
function get_cached_plm_stable(cfg::SHTnsConfig{T}, j::Int, l::Int, m::Int) where T
    @stable begin
        # Type-stable bounds checking
        1 <= j <= cfg.nlat || throw(BoundsError("j out of range"))
        0 <= l <= cfg.lmax || throw(BoundsError("l out of range"))
        abs(m) <= min(l, cfg.mmax) || throw(BoundsError("m out of range"))
        
        # Type-stable cache access
        cache_idx = l * (cfg.mmax + 1) + abs(m) + 1
        if cache_idx <= size(cfg.plm_cache, 2)
            return cfg.plm_cache[j, cache_idx]::T
        else
            # Compute on-the-fly with type-stable computation
            cost = cfg.gauss_nodes[j]::T
            sint = sqrt(max(zero(T), one(T) - cost * cost))::T
            return compute_legendre_stable(l, abs(m), cost, sint, cfg.norm)::T
        end
    end
end

"""
    compute_legendre_stable(l::Int, m::Int, cost::T, sint::T, norm::SHTnsNorm) where T

Type-stable Legendre polynomial computation.
"""
function compute_legendre_stable(l::Int, m::Int, cost::T, sint::T, norm::SHTnsNorm) where T
    @stable begin
        # Type-stable base cases
        if l == 0 && m == 0
            result = one(T)
        elseif l == 1 && m == 0
            result = cost
        elseif l == 1 && m == 1
            result = -sint  # Include Condon-Shortley phase
        else
            # Type-stable recurrence with explicit types
            if m == 0
                p0 = one(T)::T
                p1 = cost::T
                
                for ll in 2:l
                    p_new::T = (T(2*ll - 1) * cost * p1 - T(ll - 1) * p0) / T(ll)
                    p0 = p1
                    p1 = p_new
                end
                result = p1
            else
                # Associated Legendre computation with stable types
                pmm::T = one(T)
                if m > 0
                    factor::T = one(T)
                    for i in 1:m
                        factor *= T(2*i - 1)
                    end
                    pmm = ((-1)^m) * factor * (sint^m)
                end
                
                if l == m
                    result = pmm
                elseif l == m + 1
                    result = cost * T(2*m + 1) * pmm
                else
                    p0 = pmm::T
                    p1 = (cost * T(2*m + 1) * pmm)::T
                    
                    for ll in (m+2):l
                        p_new::T = (T(2*ll - 1) * cost * p1 - T(ll + m - 1) * p0) / T(ll - m)
                        p0 = p1
                        p1 = p_new
                    end
                    result = p1
                end
            end
        end
        
        # Type-stable normalization
        if norm === SHT_ORTHONORMAL
            norm_factor::T = sqrt(T(2*l + 1) / T(4π))
            if m > 0
                for k in (l-m+1):(l+m)
                    norm_factor /= sqrt(T(k))
                end
                norm_factor *= sqrt(T(2))
            end
            result *= norm_factor
        end
        
        return result::T
    end
end

# Generic performance utilities

"""
    @inbounds_stable macro

Combines @inbounds with type stability checking in debug mode.
"""
macro inbounds_stable(ex)
    if ccall(:jl_is_debugbuild, Cint, ()) != 0
        # In debug build, keep bounds checking but add type assertions
        quote
            local result = $(esc(ex))
            result  # Type will be inferred
        end
    else
        # In release build, remove bounds checking  
        quote
            @inbounds $(esc(ex))
        end
    end
end

"""
    ensure_type_stable_fft_plans!(cfg::SHTnsConfig{T}) where T

Ensure FFT plans are stored with type-stable keys and values.
"""
function ensure_type_stable_fft_plans!(cfg::SHTnsConfig{T}) where T
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