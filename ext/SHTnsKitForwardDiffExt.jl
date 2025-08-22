module SHTnsKitForwardDiffExt

using SHTnsKit
import ForwardDiff
using LinearAlgebra

# Advanced ForwardDiff optimizations with faster FFT alternatives

const π2 = 2π

# Cache for optimized FFT plans with Dual numbers
const DUAL_FFT_CACHE = Dict{Tuple{Type, Int}, Any}()

function _dft_full(row)
    N = length(row)
    T = eltype(row)
    out = Vector{Complex{T}}(undef, N)
    # Precompute common values for better performance
    inv_N = one(real(T)) / N
    @inbounds for k in 0:N-1
        acc = zero(Complex{T})
        # Use @simd for vectorization when possible
        @simd for n in 0:N-1
            phase = -π2 * k * n * inv_N
            acc += row[n+1] * cis(phase)
        end
        out[k+1] = acc
    end
    return out
end

function _idft_full(spectrum)
    N = length(spectrum)
    T = eltype(spectrum)
    out = Vector{Complex{T}}(undef, N)
    # Precompute common values
    inv_N = one(real(T)) / N
    @inbounds for n in 0:N-1
        acc = zero(Complex{T})
        @simd for k in 0:N-1
            phase = π2 * k * n * inv_N
            acc += spectrum[k+1] * cis(phase)
        end
        out[n+1] = acc * inv_N
    end
    return out
end

# Real rFFT forward: produce first N/2+1 coefficients
function _rfft_naive(row)
    full = _dft_full(row)
    K = length(row) ÷ 2 + 1
    return full[1:K]
end

# Real irFFT backward: reconstruct real row from first N/2+1 coefficients
function _irfft_naive(half, N::Int)
    K = length(half)
    full = Vector{eltype(half)}(undef, N)
    # Use copyto! for better performance
    copyto!(full, 1, half, 1, K)
    # Use @inbounds for the symmetry loop
    @inbounds for k in 2:K-1
        full[N - (k - 2)] = conj(half[k])
    end
    # Nyquist for even N already implied
    time = _idft_full(full)
    # Use @simd for vectorized real extraction
    result = Vector{real(eltype(half))}(undef, N)
    @inbounds @simd for i in 1:N
        result[i] = real(time[i])
    end
    return result
end

# Complex C2C forward/backward
function _cfft_naive(row)
    return _dft_full(row)
end

function _icfft_naive(spec)
    return _idft_full(spec)
end

# Optimized FFT using Cooley-Tukey algorithm for power-of-2 sizes
function _cooley_tukey_fft!(x::Vector{T}) where T
    N = length(x)
    if N <= 1
        return x
    end
    
    # Only use for power of 2
    if N & (N - 1) != 0
        return _dft_full(x)  # Fallback to DFT for non-power-of-2
    end
    
    # Divide
    even = x[1:2:end]
    odd = x[2:2:end]
    
    # Conquer
    _cooley_tukey_fft!(even)
    _cooley_tukey_fft!(odd)
    
    # Combine
    half_N = N ÷ 2
    for k in 0:half_N-1
        t = cis(-π2 * k / N) * odd[k+1]
        x[k+1] = even[k+1] + t
        x[k+half_N+1] = even[k+1] - t
    end
    
    return x
end

# Optimized real FFT for Dual numbers
function _rfft_optimized(row::Vector{T}) where T
    N = length(row)
    
    # Use Cooley-Tukey for power-of-2 sizes
    if N & (N - 1) == 0 && N >= 8
        # Convert to complex and use optimized FFT
        complex_row = convert(Vector{Complex{T}}, row)
        _cooley_tukey_fft!(complex_row)
        K = N ÷ 2 + 1
        return complex_row[1:K]
    else
        return _rfft_naive(row)
    end
end

# Matrix operator support for ForwardDiff
function SHTnsKit.apply_laplacian!(cfg::SHTnsKit.SHTnsConfig{T}, 
                                   qlm_in::AbstractVector{Complex{T}}, 
                                   qlm_out::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    
    # Direct computation for Dual numbers - avoid matrix operations
    lm_indices = cfg.lm_indices
    
    @inbounds for (idx, (l, m)) in enumerate(lm_indices)
        eigenvalue = -T(l * (l + 1))
        qlm_out[idx] = eigenvalue * qlm_in[idx]
    end
    
    return qlm_out
end

function SHTnsKit.apply_costheta_operator!(cfg::SHTnsKit.SHTnsConfig{T}, 
                                          qlm_in::AbstractVector{Complex{T}}, 
                                          qlm_out::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    
    lm_indices = cfg.lm_indices
    fill!(qlm_out, zero(Complex{T}))
    
    # Direct coupling computation for Dual numbers
    @inbounds for (idx_out, (l_out, m_out)) in enumerate(lm_indices)
        for (idx_in, (l_in, m_in)) in enumerate(lm_indices)
            if m_out == m_in && abs(l_in - l_out) == 1
                coeff = SHTnsKit._costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                qlm_out[idx_out] += T(coeff) * qlm_in[idx_in]
            end
        end
    end
    
    return qlm_out
end

# Point evaluation support for ForwardDiff (critical for PINNs)
function SHTnsKit.sh_to_point(cfg::SHTnsKit.SHTnsConfig{T}, 
                              qlm::AbstractVector{T2}, 
                              theta::T3, phi::T4) where {T<:ForwardDiff.Dual, T2, T3, T4}
    
    result = zero(promote_type(T2, T3, T4))
    lm_indices = cfg.lm_indices
    
    # Direct evaluation avoiding FFT
    @inbounds for (idx, (l, m)) in enumerate(lm_indices)
        ylm_val = SHTnsKit._evaluate_spherical_harmonic_dual(cfg, l, m, theta, phi)
        result += qlm[idx] * ylm_val
    end
    
    return result
end

# Optimized spherical harmonic evaluation for Dual numbers
function SHTnsKit._evaluate_spherical_harmonic_dual(cfg::SHTnsKit.SHTnsConfig{T}, 
                                                   l::Int, m::Int, 
                                                   theta::T2, phi::T3) where {T, T2, T3}
    
    RetType = promote_type(T, T2, T3)
    
    # Compute associated Legendre polynomial
    cos_theta = cos(theta)
    plm = SHTnsKit._compute_legendre_polynomial_dual(l, abs(m), cos_theta)
    
    # Spherical harmonic normalization
    norm_factor = SHTnsKit._spherical_harmonic_normalization(cfg, l, m)
    
    # Azimuthal part
    if m == 0
        return RetType(norm_factor * plm)
    elseif m > 0
        return RetType(norm_factor * plm * cos(m * phi))
    else # m < 0
        return RetType(norm_factor * plm * sin(abs(m) * phi))
    end
end

# Optimized Legendre polynomial for Dual numbers
function SHTnsKit._compute_legendre_polynomial_dual(l::Int, m::Int, x::T) where T
    if m > l
        return zero(T)
    end
    
    # Use recurrence relation for efficiency
    if l == 0
        return one(T)
    elseif l == 1
        if m == 0
            return x
        elseif m == 1
            return sqrt(one(T) - x*x)
        else
            return zero(T)
        end
    end
    
    # General recurrence (simplified version)
    # Full implementation would use proper three-term recurrence
    sin_theta = sqrt(one(T) - x*x)
    
    # Start with P_m^m
    pmm = one(T)
    for i in 1:m
        pmm *= -(2*i - 1) * sin_theta
    end
    
    if l == m
        return pmm
    end
    
    # P_{m+1}^m
    pmmp1 = x * (2*m + 1) * pmm
    if l == m + 1
        return pmmp1
    end
    
    # Use three-term recurrence for higher degrees
    for ll in (m+2):l
        pll = (x * (2*ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end
    
    return pmmp1
end

# Overloads specialized for Dual-friendly paths

function SHTnsKit.compute_fourier_coefficients_spatial(spatial_data::AbstractMatrix{T}, cfg::SHTnsKit.SHTnsConfig{T}) where {T<:ForwardDiff.Dual}
    nlat, nphi = size(spatial_data)
    nphi_modes = nphi ÷ 2 + 1
    fourier = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    # Process rows with potential threading for large grids
    if nlat > 32 && SHTnsKit.get_threading()
        Threads.@threads for i in 1:nlat
            @inbounds row = view(spatial_data, i, :)
            @inbounds fourier[i, :] = _rfft_naive(row)
        end
    else
        @inbounds for i in 1:nlat
            row = view(spatial_data, i, :)
            fourier[i, :] = _rfft_naive(row)
        end
    end
    return fourier
end

function SHTnsKit.compute_spatial_from_fourier(fourier_coeffs::AbstractMatrix{Complex{T}}, cfg::SHTnsKit.SHTnsConfig{T}) where {T<:ForwardDiff.Dual}
    nlat, nphi_modes = size(fourier_coeffs)
    N = cfg.nphi
    spatial = Matrix{T}(undef, nlat, N)
    # Process rows with potential threading for large grids
    if nlat > 32 && SHTnsKit.get_threading()
        Threads.@threads for i in 1:nlat
            @inbounds half = view(fourier_coeffs, i, :)
            @inbounds spatial[i, :] = _irfft_naive(half, N)
        end
    else
        @inbounds for i in 1:nlat
            half = view(fourier_coeffs, i, :)
            spatial[i, :] = _irfft_naive(half, N)
        end
    end
    return spatial
end

function SHTnsKit.azimuthal_fft_complex_forward!(cfg::SHTnsKit.SHTnsConfig{T}, spatial_row::AbstractVector{Complex{T}}, fourier_coeffs::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    full = _cfft_naive(spatial_row)
    # Use copyto! for better performance
    copyto!(fourier_coeffs, 1, full, 1, length(full))
    return nothing
end

function SHTnsKit.azimuthal_fft_complex_backward!(cfg::SHTnsKit.SHTnsConfig{T}, fourier_coeffs::AbstractVector{Complex{T}}, spatial_row::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    time = _icfft_naive(fourier_coeffs)
    # Use copyto! for better performance
    copyto!(spatial_row, 1, time, 1, length(time))
    return nothing
end

end # module

