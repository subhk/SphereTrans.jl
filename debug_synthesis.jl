using LinearAlgebra, FFTW, Statistics
include("src/SHTnsKit.jl")
using .SHTnsKit

function debug_synthesis()
    cfg = create_gauss_config(4, 4)
    
    println("=== DEBUGGING SYNTHESIS STEP BY STEP ===")
    println("Grid: $(get_nlat(cfg)) x $(get_nphi(cfg))")
    
    # Create coefficient array with only Y_0^0 = sqrt(4π)
    sh_coeffs = zeros(get_nlm(cfg))
    sh_coeffs[1] = sqrt(4π)  # Y_0^0 coefficient
    
    println("Input coefficients:")
    println("  Y_0^0 = $(sh_coeffs[1])")
    println("  All others = 0")
    
    # Manual synthesis following my algorithm
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    nphi_modes = nphi ÷ 2 + 1
    fourier_coeffs = Matrix{ComplexF64}(undef, nlat, nphi_modes)
    fill!(fourier_coeffs, zero(ComplexF64))
    
    println("\nStep 1: Building Fourier coefficients...")
    
    # Only m=0 mode should be non-zero
    m = 0
    mode_coeffs = Vector{ComplexF64}(undef, nlat)
    fill!(mode_coeffs, zero(ComplexF64))
    
    # For each latitude point
    for i in 1:nlat
        value = zero(ComplexF64)
        # Only (0,0) coefficient is non-zero
        plm_val = cfg.plm_cache[i, 1]  # P_0^0 value
        coeff_val = sh_coeffs[1]  # Y_0^0 coefficient
        
        # m=0 case
        value += coeff_val * plm_val
        mode_coeffs[i] = value
        
        if i <= 3
            println("  lat $i: PLM = $plm_val, coeff = $coeff_val, result = $value")
        end
    end
    
    # Insert m=0 mode into Fourier array
    SHTnsKit.insert_fourier_mode!(fourier_coeffs, 0, mode_coeffs, nlat)
    
    println("\nStep 2: Fourier coefficients array:")
    println("  fourier_coeffs[1,1] (m=0): $(fourier_coeffs[1,1])")
    println("  fourier_coeffs[1,2] (m=1): $(fourier_coeffs[1,2])")
    
    # Step 3: Convert to spatial
    println("\nStep 3: Converting to spatial...")
    spatial_data = SHTnsKit.compute_spatial_from_fourier(fourier_coeffs, cfg)
    
    println("Result:")
    println("  spatial_data[1,1] = $(spatial_data[1,1])")
    println("  spatial_data mean = $(mean(spatial_data))")
    println("  Expected: 1.0")
    
    destroy_config(cfg)
    return spatial_data[1,1]
end

result = debug_synthesis()
println("\nFinal result: $result")
println("Factor needed: $(1.0 / result)")