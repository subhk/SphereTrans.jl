module SHTnsKitForwardDiffExt

using SHTnsKit
import ForwardDiff

# Naive DFT/IDFT fallbacks for Dual-friendly execution.

const π2 = 2π

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

