module SHTnsKitForwardDiffExt

using SHTnsKit
import ForwardDiff

# Naive DFT/IDFT fallbacks for Dual-friendly execution.

const π2 = 2π

function _dft_full(row)
    N = length(row)
    T = eltype(row)
    out = Vector{Complex{T}}(undef, N)
    for k in 0:N-1
        acc = zero(Complex{T})
        for n in 0:N-1
            acc += row[n+1] * cis(-π2 * k * n / N)
        end
        out[k+1] = acc
    end
    return out
end

function _idft_full(spectrum)
    N = length(spectrum)
    T = eltype(spectrum)
    out = Vector{Complex{T}}(undef, N)
    for n in 0:N-1
        acc = zero(Complex{T})
        for k in 0:N-1
            acc += spectrum[k+1] * cis(π2 * k * n / N)
        end
        out[n+1] = acc / N
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
    full[1:K] = half
    for k in 2:K-1
        full[N - (k - 2)] = conj(half[k])
    end
    # Nyquist for even N already implied
    time = _idft_full(full)
    return real.(time)
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
    for i in 1:nlat
        row = view(spatial_data, i, :)
        fourier[i, :] = _rfft_naive(row)
    end
    return fourier
end

function SHTnsKit.compute_spatial_from_fourier(fourier_coeffs::AbstractMatrix{Complex{T}}, cfg::SHTnsKit.SHTnsConfig{T}) where {T<:ForwardDiff.Dual}
    nlat, nphi_modes = size(fourier_coeffs)
    N = cfg.nphi
    spatial = Matrix{T}(undef, nlat, N)
    for i in 1:nlat
        half = view(fourier_coeffs, i, :)
        spatial[i, :] = _irfft_naive(half, N)
    end
    return spatial
end

function SHTnsKit.azimuthal_fft_complex_forward!(cfg::SHTnsKit.SHTnsConfig{T}, spatial_row::AbstractVector{Complex{T}}, fourier_coeffs::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    full = _cfft_naive(spatial_row)
    fourier_coeffs[1:length(full)] .= full
    return nothing
end

function SHTnsKit.azimuthal_fft_complex_backward!(cfg::SHTnsKit.SHTnsConfig{T}, fourier_coeffs::AbstractVector{Complex{T}}, spatial_row::AbstractVector{Complex{T}}) where {T<:ForwardDiff.Dual}
    time = _icfft_naive(fourier_coeffs)
    spatial_row[1:length(time)] .= time
    return nothing
end

end # module

