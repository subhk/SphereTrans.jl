"""
FFT utilities for spherical harmonic transforms.
Handles the azimuthal (longitude) Fourier transforms efficiently.
"""

"""
    setup_fft_plans!(cfg::SHTnsConfig{T}) where T

Setup and cache FFT plans for efficient azimuthal transforms.
This precomputes the FFT plans to avoid overhead during transforms.
"""
function setup_fft_plans!(cfg::SHTnsConfig{T}) where T
    nphi = cfg.nphi
    
    # Real-to-complex FFT for forward transforms (spatial -> spectral)
    real_array = zeros(T, nphi)
    cfg.fft_plans[:r2c] = plan_rfft(real_array)
    
    # Complex-to-real FFT for backward transforms (spectral -> spatial)  
    complex_array = zeros(Complex{T}, nphi ÷ 2 + 1)
    cfg.fft_plans[:c2r] = plan_irfft(complex_array, nphi)
    
    # Complex-to-complex FFTs for complex-valued fields
    complex_full = zeros(Complex{T}, nphi)
    cfg.fft_plans[:c2c_forward] = plan_fft(complex_full)
    cfg.fft_plans[:c2c_backward] = plan_ifft(complex_full)
    
    return nothing
end

"""
    azimuthal_fft_forward!(cfg::SHTnsConfig{T}, spatial_row::AbstractVector{T}, 
                          fourier_coeffs::AbstractVector{Complex{T}}) where T

Perform forward azimuthal FFT on a single latitude row.
Transforms spatial data to Fourier coefficients in longitude.

# Arguments
- `cfg`: SHTns configuration
- `spatial_row`: Input spatial data at one latitude (length nphi)
- `fourier_coeffs`: Output Fourier coefficients (length nphi÷2+1 for real input)
"""
function azimuthal_fft_forward!(cfg::SHTnsConfig{T}, 
                               spatial_row::AbstractVector{T},
                               fourier_coeffs::AbstractVector{Complex{T}}) where T
    nphi = cfg.nphi
    length(spatial_row) == nphi || error("spatial_row length must equal nphi")
    expected_fc_length = nphi ÷ 2 + 1
    length(fourier_coeffs) >= expected_fc_length || error("fourier_coeffs too short")
    
    # Use cached FFT plan (ensure contiguous array)
    fft_plan = cfg.fft_plans[:r2c]
    if spatial_row isa SubArray
        # Copy to contiguous array for FFTW
        temp_row = Vector{T}(spatial_row)
        fourier_coeffs .= fft_plan * temp_row
    else
        fourier_coeffs .= fft_plan * spatial_row
    end
    
    return nothing
end

"""
    azimuthal_fft_backward!(cfg::SHTnsConfig{T}, fourier_coeffs::AbstractVector{Complex{T}},
                           spatial_row::AbstractVector{T}) where T

Perform backward azimuthal FFT on a single latitude row.
Transforms Fourier coefficients to spatial data in longitude.

# Arguments  
- `cfg`: SHTns configuration
- `fourier_coeffs`: Input Fourier coefficients (length nphi÷2+1)
- `spatial_row`: Output spatial data at one latitude (length nphi)
"""
function azimuthal_fft_backward!(cfg::SHTnsConfig{T},
                                fourier_coeffs::AbstractVector{Complex{T}},
                                spatial_row::AbstractVector{T}) where T
    nphi = cfg.nphi
    expected_fc_length = nphi ÷ 2 + 1
    length(fourier_coeffs) >= expected_fc_length || error("fourier_coeffs too short")
    length(spatial_row) == nphi || error("spatial_row length must equal nphi")
    
    # Use cached FFT plan (ensure contiguous array)
    ifft_plan = cfg.fft_plans[:c2r]
    if fourier_coeffs isa SubArray
        # Copy to contiguous array for FFTW
        temp_coeffs = Vector{Complex{T}}(fourier_coeffs[1:expected_fc_length])
        result = ifft_plan * temp_coeffs
        if spatial_row isa SubArray
            spatial_row .= result
        else
            spatial_row .= result
        end
    else
        result = ifft_plan * fourier_coeffs[1:expected_fc_length]
        spatial_row .= result
    end
    
    return nothing
end

"""
    azimuthal_fft_complex_forward!(cfg::SHTnsConfig{T}, spatial_row::AbstractVector{Complex{T}},
                                  fourier_coeffs::AbstractVector{Complex{T}}) where T

Forward FFT for complex-valued spatial fields.
"""
function azimuthal_fft_complex_forward!(cfg::SHTnsConfig{T},
                                       spatial_row::AbstractVector{Complex{T}},
                                       fourier_coeffs::AbstractVector{Complex{T}}) where T
    nphi = cfg.nphi
    length(spatial_row) == nphi || error("spatial_row length must equal nphi")
    length(fourier_coeffs) >= nphi || error("fourier_coeffs too short")
    
    fft_plan = cfg.fft_plans[:c2c_forward]
    
    # Ensure contiguous arrays for FFTW
    if spatial_row isa SubArray
        temp_row = Vector{Complex{T}}(spatial_row)
        result = fft_plan * temp_row
        fourier_coeffs[1:nphi] .= result
    else
        fourier_coeffs[1:nphi] .= fft_plan * spatial_row
    end
    
    return nothing
end

"""
    azimuthal_fft_complex_backward!(cfg::SHTnsConfig{T}, fourier_coeffs::AbstractVector{Complex{T}},
                                   spatial_row::AbstractVector{Complex{T}}) where T

Backward FFT for complex-valued spatial fields.
"""
function azimuthal_fft_complex_backward!(cfg::SHTnsConfig{T},
                                        fourier_coeffs::AbstractVector{Complex{T}},
                                        spatial_row::AbstractVector{Complex{T}}) where T
    nphi = cfg.nphi
    length(fourier_coeffs) >= nphi || error("fourier_coeffs too short")  
    length(spatial_row) == nphi || error("spatial_row length must equal nphi")
    
    ifft_plan = cfg.fft_plans[:c2c_backward]
    
    # Ensure contiguous arrays for FFTW
    if fourier_coeffs isa SubArray
        temp_coeffs = Vector{Complex{T}}(fourier_coeffs[1:nphi])
        result = ifft_plan * temp_coeffs
        if spatial_row isa SubArray
            spatial_row .= result
        else
            spatial_row .= result
        end
    else
        result = ifft_plan * fourier_coeffs[1:nphi]
        spatial_row .= result
    end
    
    return nothing
end

"""
    extract_fourier_mode!(fourier_coeffs::AbstractMatrix{Complex{T}}, m::Int, 
                         output::AbstractVector{Complex{T}}, nlat::Int) where T

Extract Fourier mode m from azimuthal FFT coefficients for all latitudes.
This is used to gather coefficients for a specific azimuthal wavenumber m.

# Arguments
- `fourier_coeffs`: Fourier coefficients for all latitudes (nlat × (nphi÷2+1))
- `m`: Azimuthal wavenumber to extract
- `output`: Output array for mode m coefficients (length nlat)
- `nlat`: Number of latitude points
"""
function extract_fourier_mode!(fourier_coeffs::AbstractMatrix{Complex{T}}, m::Int,
                               output::AbstractVector{Complex{T}}, nlat::Int) where T
    size(fourier_coeffs, 1) >= nlat || error("fourier_coeffs has insufficient latitude points")
    length(output) >= nlat || error("output array too short")
    
    nphi_half = size(fourier_coeffs, 2)
    m_idx = m + 1  # Convert to 1-based indexing
    
    if m_idx <= nphi_half
        @inbounds for i in 1:nlat
            output[i] = fourier_coeffs[i, m_idx]
        end
    else
        # Mode m is beyond Nyquist frequency, set to zero
        @inbounds output[1:nlat] .= zero(Complex{T})
    end
    
    return nothing
end

"""
    insert_fourier_mode!(output::AbstractMatrix{Complex{T}}, m::Int,
                         mode_coeffs::AbstractVector{Complex{T}}, nlat::Int) where T

Insert Fourier mode m coefficients into the full azimuthal FFT array.
This is the inverse operation of extract_fourier_mode!.

# Arguments
- `output`: Full Fourier coefficient array (nlat × nphi_modes)
- `m`: Azimuthal wavenumber being inserted
- `mode_coeffs`: Coefficients for mode m (length nlat)
- `nlat`: Number of latitude points
"""
function insert_fourier_mode!(output::AbstractMatrix{Complex{T}}, m::Int,
                              mode_coeffs::AbstractVector{Complex{T}}, nlat::Int) where T
    size(output, 1) >= nlat || error("output has insufficient latitude points")
    length(mode_coeffs) >= nlat || error("mode_coeffs array too short")
    
    nphi_half = size(output, 2)
    m_idx = m + 1  # Convert to 1-based indexing
    
    if m_idx <= nphi_half
        @inbounds for i in 1:nlat
            output[i, m_idx] = mode_coeffs[i]
        end
    end
    # If m is beyond Nyquist, simply ignore (can't represent this mode)
    
    return nothing
end

"""
    compute_fourier_coefficients_spatial(spatial_data::AbstractMatrix{T}, 
                                        cfg::SHTnsConfig{T}) where T

Transform spatial grid data to Fourier coefficients in the azimuthal direction.
This operates on all latitudes simultaneously with optional threading.

# Arguments
- `spatial_data`: Spatial field on grid (nlat × nphi)
- `cfg`: SHTns configuration

# Returns
- `fourier_coeffs`: Complex Fourier coefficients (nlat × (nphi÷2+1))
"""
function compute_fourier_coefficients_spatial(spatial_data::AbstractMatrix{T},
                                             cfg::SHTnsConfig{T}) where T
    nlat, nphi = size(spatial_data)
    nlat == cfg.nlat || error("spatial_data latitude dimension mismatch")
    nphi == cfg.nphi || error("spatial_data longitude dimension mismatch")
    
    nphi_modes = nphi ÷ 2 + 1
    fourier_coeffs = Matrix{Complex{T}}(undef, nlat, nphi_modes)
    
    # Transform each latitude row with optional threading and optimized memory access
    if SHTnsKit.get_threading() && nlat > 32
        @threads for i in 1:nlat
            @inbounds azimuthal_fft_forward!(cfg, view(spatial_data, i, :), view(fourier_coeffs, i, :))
        end
    else
        @inbounds for i in 1:nlat
            azimuthal_fft_forward!(cfg, view(spatial_data, i, :), view(fourier_coeffs, i, :))
        end
    end
    
    return fourier_coeffs
end

"""
    compute_spatial_from_fourier(fourier_coeffs::AbstractMatrix{Complex{T}},
                                cfg::SHTnsConfig{T}) where T

Transform Fourier coefficients back to spatial grid data with optional threading.

# Arguments  
- `fourier_coeffs`: Complex Fourier coefficients (nlat × nphi_modes)
- `cfg`: SHTns configuration

# Returns
- `spatial_data`: Real spatial field on grid (nlat × nphi)
"""
function compute_spatial_from_fourier(fourier_coeffs::AbstractMatrix{Complex{T}},
                                     cfg::SHTnsConfig{T}) where T
    nlat, nphi_modes = size(fourier_coeffs)
    nlat == cfg.nlat || error("fourier_coeffs latitude dimension mismatch")
    expected_modes = cfg.nphi ÷ 2 + 1
    nphi_modes >= expected_modes || error("fourier_coeffs insufficient modes")
    
    spatial_data = Matrix{T}(undef, nlat, cfg.nphi)
    
    # Transform each latitude row with optional threading and optimized memory access
    if SHTnsKit.get_threading() && nlat > 32
        @threads for i in 1:nlat
            @inbounds azimuthal_fft_backward!(cfg, view(fourier_coeffs, i, :), view(spatial_data, i, :))
        end
    else
        @inbounds for i in 1:nlat
            azimuthal_fft_backward!(cfg, view(fourier_coeffs, i, :), view(spatial_data, i, :))
        end
    end
    
    return spatial_data
end