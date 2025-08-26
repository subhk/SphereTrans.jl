"""
Fallback DFT along longitude to enable AD (e.g., ForwardDiff) when FFTW
does not support the element type (like Dual numbers).
"""

const _TWO_PI = 2π

function _dft_phi(A::AbstractMatrix, dir::Int)
    nlat, nlon = size(A)
    Y = similar(complex.(A))
    @inbounds for i in 1:nlat
        for k in 0:(nlon-1)
            s = zero(eltype(Y))
            for j in 0:(nlon-1)
                s += A[i, j+1] * cis(dir * _TWO_PI * k * j / nlon)
            end
            Y[i, k+1] = s
        end
    end
    return Y
end

"""fft_phi(A): FFT along second dimension with AD-friendly fallback"""
function fft_phi(A::AbstractMatrix)
    try
        return fft(A, 2)
    catch
        return _dft_phi(A, -1)
    end
end

"""ifft_phi(A): inverse FFT along second dimension with AD-friendly fallback"""
function ifft_phi(A::AbstractMatrix)
    nlon = size(A,2)
    try
        return ifft(A, 2)
    catch
        return (1/nlon) * _dft_phi(A, +1)
    end
end

