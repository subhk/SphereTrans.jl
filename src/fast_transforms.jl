"""
Fast spherical harmonic transforms with optimized algorithms and memory access patterns.

Key optimizations:
1. Fast Legendre transforms using recurrence relations
2. Block-structured operations for cache efficiency
3. Reduced memory bandwidth requirements
4. Better vectorization opportunities
"""

using LinearAlgebra
using Base.Threads

"""
Precomputed data structure for fast Legendre transforms.
Uses structure-of-arrays layout for better vectorization.
"""
struct FastLegendreData{T<:AbstractFloat}
    # Recurrence coefficients stored contiguously
    alpha_coeffs::Vector{T}  # α_l^m coefficients
    beta_coeffs::Vector{T}   # β_l^m coefficients
    
    # Block-structured storage for better cache utilization
    l_blocks::Vector{UnitRange{Int}}      # Ranges of l for each block
    m_offsets::Vector{Int}                # Starting indices for each m
    
    # Workspace arrays (thread-local)
    workspace::Vector{Vector{T}}          # One per thread
    
    # Precomputed starting values
    plm_seeds::Matrix{T}  # P_m^m values at all latitudes
    
    function FastLegendreData{T}(lmax::Int, mmax::Int, theta::Vector{T}) where T
        nlat = length(theta)
        
        # Precompute recurrence coefficients
        alpha_coeffs = Vector{T}()
        beta_coeffs = Vector{T}()
        l_blocks = UnitRange{Int}[]
        m_offsets = Vector{Int}()
        
        for m in 0:mmax
            push!(m_offsets, length(alpha_coeffs) + 1)
            
            # Block l values for better cache behavior
            block_size = min(64, lmax - m + 1)  # Tunable parameter
            for l_start in m:block_size:(lmax)
                l_end = min(l_start + block_size - 1, lmax)
                push!(l_blocks, l_start:l_end)
                
                for l in l_start:l_end
                    # Standard recurrence coefficients
                    if l > m
                        alpha = sqrt((4*l^2 - 1) / (l^2 - m^2))
                        beta = l > m + 1 ? -sqrt(((l-1)^2 - m^2) / (4*(l-1)^2 - 1)) : T(0)
                        push!(alpha_coeffs, alpha)
                        push!(beta_coeffs, beta)
                    else
                        push!(alpha_coeffs, T(0))
                        push!(beta_coeffs, T(0))
                    end
                end
            end
        end
        
        # Precompute P_m^m seeds
        plm_seeds = Matrix{T}(undef, nlat, mmax + 1)
        for i in 1:nlat
            costheta = cos(theta[i])
            sintheta = sin(theta[i])
            
            # P_0^0 = 1
            plm_seeds[i, 1] = T(1)
            
            # P_m^m using recurrence
            pmm = T(1)
            for m in 1:mmax
                pmm *= -sintheta * sqrt((2*m + 1) / (2*m))
                plm_seeds[i, m + 1] = pmm
            end
        end
        
        # Initialize thread-local workspace
        nthreads = Threads.nthreads()
        workspace = [Vector{T}(undef, max(lmax + 1, 512)) for _ in 1:nthreads]
        
        new{T}(alpha_coeffs, beta_coeffs, l_blocks, m_offsets, workspace, plm_seeds)
    end
end

"""
    fast_legendre_synthesis!(legendre_data::FastLegendreData{T}, 
                            sh_coeffs::AbstractVector{T}, 
                            fourier_coeffs::Matrix{Complex{T}}, 
                            cfg::SHTnsConfig{T}) where T

Fast Legendre synthesis using recurrence relations and block structure.
"""
function fast_legendre_synthesis!(legendre_data::FastLegendreData{T},
                                 sh_coeffs::AbstractVector{T},
                                 fourier_coeffs::Matrix{Complex{T}},
                                 cfg::SHTnsConfig{T}) where T
    
    nlat, nphi_modes = size(fourier_coeffs)
    fill!(fourier_coeffs, zero(Complex{T}))
    
    # Get thread-local workspace
    tid = Threads.threadid()
    workspace = legendre_data.workspace[tid]
    
    # Process each azimuthal mode m
    @inbounds for m in 0:min(cfg.mmax, nphi_modes - 1)
        (m == 0 || m % cfg.mres == 0) || continue
        
        m_col = m + 1
        m_offset = legendre_data.m_offsets[m + 1]
        
        # Vectorized processing of latitude points
        @inbounds @simd for i in 1:nlat
            value = zero(Complex{T})
            
            # Start with P_m^m
            if m <= cfg.lmax
                plm_curr = legendre_data.plm_seeds[i, m + 1]
                coeff_idx = SHTnsKit.lmidx(cfg, m, m)
                if coeff_idx <= length(sh_coeffs)
                    value += sh_coeffs[coeff_idx] * plm_curr
                end
                
                # Use recurrence for l > m
                if m + 1 <= cfg.lmax
                    costheta = cos(cfg.theta[i])
                    
                    # P_{m+1}^m
                    plm_prev = plm_curr
                    plm_curr = sqrt(2*m + 3) * costheta * plm_curr
                    
                    coeff_idx = SHTnsKit.lmidx(cfg, m + 1, m)
                    if coeff_idx <= length(sh_coeffs)
                        value += sh_coeffs[coeff_idx] * plm_curr
                    end
                    
                    # Recurrence for l > m + 1
                    for l in (m + 2):cfg.lmax
                        alpha_idx = m_offset + (l - m - 1)
                        alpha = legendre_data.alpha_coeffs[alpha_idx]
                        beta = legendre_data.beta_coeffs[alpha_idx]
                        
                        plm_new = alpha * costheta * plm_curr + beta * plm_prev
                        
                        coeff_idx = SHTnsKit.lmidx(cfg, l, m)
                        if coeff_idx <= length(sh_coeffs)
                            value += sh_coeffs[coeff_idx] * plm_new
                        end
                        
                        plm_prev = plm_curr
                        plm_curr = plm_new
                    end
                end
            end
            
            # Apply appropriate scaling for real transforms
            if m == 0
                fourier_coeffs[i, m_col] = value
            else
                fourier_coeffs[i, m_col] = value * T(0.5)
            end
        end
    end
    
    return fourier_coeffs
end

"""
    fast_legendre_analysis!(legendre_data::FastLegendreData{T},
                           fourier_coeffs::Matrix{Complex{T}},
                           sh_coeffs::AbstractVector{T},
                           cfg::SHTnsConfig{T}) where T

Fast Legendre analysis using optimized integration.
"""
function fast_legendre_analysis!(legendre_data::FastLegendreData{T},
                                fourier_coeffs::Matrix{Complex{T}},
                                sh_coeffs::AbstractVector{T},
                                cfg::SHTnsConfig{T}) where T
    
    nlat = size(fourier_coeffs, 1)
    fill!(sh_coeffs, zero(T))
    
    # Get thread-local workspace
    tid = Threads.threadid()
    workspace = legendre_data.workspace[tid]
    
    # Extract mode data efficiently
    @inbounds for (coeff_idx, (l, m)) in enumerate(cfg.lm_indices)
        if m <= size(fourier_coeffs, 2) - 1
            m_col = m + 1
            m_offset = legendre_data.m_offsets[m + 1]
            
            # Compute integral using Gaussian quadrature
            integral = zero(Complex{T})
            
            # Vectorized integration over latitudes
            if m == l  # P_m^m case
                @inbounds @simd for i in 1:nlat
                    plm_val = legendre_data.plm_seeds[i, m + 1]
                    weight = cfg.gauss_weights[i]
                    integral += fourier_coeffs[i, m_col] * plm_val * weight
                end
            else  # General case with recurrence
                @inbounds @simd for i in 1:nlat
                    # Compute P_l^m using recurrence (same as synthesis)
                    costheta = cos(cfg.theta[i])
                    
                    plm_curr = legendre_data.plm_seeds[i, m + 1]
                    if l == m
                        plm_val = plm_curr
                    else
                        # Recurrence up to desired l
                        if m + 1 <= l
                            plm_prev = plm_curr
                            plm_curr = sqrt(2*m + 3) * costheta * plm_curr
                        end
                        
                        for ll in (m + 2):l
                            alpha_idx = m_offset + (ll - m - 1)
                            alpha = legendre_data.alpha_coeffs[alpha_idx]
                            beta = legendre_data.beta_coeffs[alpha_idx]
                            
                            plm_new = alpha * costheta * plm_curr + beta * plm_prev
                            plm_prev = plm_curr
                            plm_curr = plm_new
                        end
                        
                        plm_val = plm_curr
                    end
                    
                    weight = cfg.gauss_weights[i]
                    integral += fourier_coeffs[i, m_col] * plm_val * weight
                end
            end
            
            # Apply normalization
            phi_normalization = T(2π) / cfg.nphi
            integral *= phi_normalization
            
            # Extract real part with proper scaling
            if m == 0
                sh_coeffs[coeff_idx] = real(integral)
            else
                sh_coeffs[coeff_idx] = real(integral) * T(2)
            end
        end
    end
    
    return sh_coeffs
end

"""
Global cache for fast Legendre data to avoid recomputation.
"""
const FAST_LEGENDRE_CACHE = Dict{Tuple{Type, Int, Int, Vector}, FastLegendreData}()

"""
    get_fast_legendre_data(cfg::SHTnsConfig{T}) where T

Get or create cached fast Legendre data for the configuration.
"""
function get_fast_legendre_data(cfg::SHTnsConfig{T}) where T
    key = (T, cfg.lmax, cfg.mmax, cfg.theta)
    
    if haskey(FAST_LEGENDRE_CACHE, key)
        return FAST_LEGENDRE_CACHE[key]
    else
        legendre_data = FastLegendreData{T}(cfg.lmax, cfg.mmax, cfg.theta)
        FAST_LEGENDRE_CACHE[key] = legendre_data
        return legendre_data
    end
end

"""
    fast_sh_to_spat!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                     spatial_data::AbstractMatrix{T}) where T

Optimized spherical harmonic synthesis using fast algorithms.
"""
function fast_sh_to_spat!(cfg::SHTnsConfig{T}, sh_coeffs::AbstractVector{T},
                          spatial_data::AbstractMatrix{T}) where T
    validate_config_stable(cfg)
    
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Get fast Legendre data
    legendre_data = get_fast_legendre_data(cfg)
    
    # Allocate Fourier coefficient workspace
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    
    # Fast Legendre synthesis
    fast_legendre_synthesis!(legendre_data, sh_coeffs, fourier_coeffs, cfg)
    
    # FFT to spatial domain
    spatial_temp = compute_spatial_from_fourier(fourier_coeffs, cfg)
    spatial_data .= spatial_temp .* T(nphi)
    
    return spatial_data
end

"""
    fast_spat_to_sh!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                     sh_coeffs::AbstractVector{T}) where T

Optimized spherical harmonic analysis using fast algorithms.
"""
function fast_spat_to_sh!(cfg::SHTnsConfig{T}, spatial_data::AbstractMatrix{T},
                          sh_coeffs::AbstractVector{T}) where T
    validate_config_stable(cfg)
    
    # Get fast Legendre data
    legendre_data = get_fast_legendre_data(cfg)
    
    # FFT from spatial domain
    fourier_coeffs = compute_fourier_coefficients_spatial(spatial_data, cfg)
    
    # Fast Legendre analysis
    fast_legendre_analysis!(legendre_data, fourier_coeffs, sh_coeffs, cfg)
    
    return sh_coeffs
end