#!/usr/bin/env julia

"""
Vector Field Analysis Demo
==========================

This example demonstrates vector field transforms and analysis:
- Spheroidal-toroidal decomposition
- Gradient and curl operations
- Vector field synthesis and analysis
- Energy spectrum of vector components
"""

using SHTnsKit
using LinearAlgebra
using Printf

println("Vector Field Analysis Demo")
println("=========================")

# Create configuration
lmax = 32
cfg = create_gauss_config(lmax, lmax)
nlat, nphi = get_nlat(cfg), get_nphi(cfg)
nlm = get_nlm(cfg)

println("Grid: $nlat × $nphi points")
println("Spectral: $nlm coefficients")

# 1. Create a realistic vector field
println("\n1. Creating Test Vector Field")
println("-" * 30)

function create_jet_stream(cfg)
    """Create a simple jet stream pattern"""
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    u = zeros(nlat, nphi)  # Zonal component
    v = zeros(nlat, nphi)  # Meridional component
    
    for i in 1:nlat
        theta = get_theta(cfg, i-1)
        lat = π/2 - theta  # Convert to latitude
        
        for j in 1:nphi
            lon = get_phi(cfg, j)
            
            # Jet stream at mid-latitudes
            jet_strength = 20.0  # m/s
            u[i, j] = jet_strength * exp(-(lat - π/6)^2 / (2*(π/12)^2))
            
            # Add some wave pattern
            wave_amplitude = 3.0
            v[i, j] = wave_amplitude * sin(3*lon) * cos(2*lat)
        end
    end
    
    return u, v
end

u, v = create_jet_stream(cfg)
max_speed = maximum(sqrt.(u.^2 + v.^2))
println("Maximum wind speed: $(@sprintf("%.1f", max_speed)) m/s")

# 2. Vector Field Decomposition
println("\n2. Vector Field Decomposition")
println("-" * 29)

# Decompose into spheroidal and toroidal components
Slm, Tlm = analyze_vector(cfg, u, v)

# Compute energy in each component
total_energy = sum(Slm.^2 + Tlm.^2)
spheroidal_energy = sum(Slm.^2)
toroidal_energy = sum(Tlm.^2)

println("Total kinetic energy: $(@sprintf("%.2e", total_energy))")
println("Spheroidal (divergent) component: $(@sprintf("%.1f", spheroidal_energy/total_energy*100))%")
println("Toroidal (rotational) component: $(@sprintf("%.1f", toroidal_energy/total_energy*100))%")

# 3. Reconstruct vector field
println("\n3. Vector Field Reconstruction")
println("-" * 30)

u_reconstructed, v_reconstructed = synthesize_vector(cfg, Slm, Tlm)

# Check reconstruction error
u_error = maximum(abs.(u - u_reconstructed))
v_error = maximum(abs.(v - v_reconstructed))
println("Reconstruction errors - u: $(@sprintf("%.2e", u_error)), v: $(@sprintf("%.2e", v_error))")

# 4. Gradient and Curl Analysis
println("\n4. Gradient and Curl Operations")
println("-" * 31)

# Compute gradient of a scalar field (using spheroidal part)
Vt_grad, Vp_grad = compute_gradient(cfg, Slm)
max_grad = maximum(sqrt.(Vt_grad.^2 + Vp_grad.^2))
println("Maximum gradient magnitude: $(@sprintf("%.2e", max_grad))")

# Compute curl (using toroidal part)
Vt_curl, Vp_curl = compute_curl(cfg, Tlm)
max_curl = maximum(sqrt.(Vt_curl.^2 + Vp_curl.^2))
println("Maximum curl magnitude: $(@sprintf("%.2e", max_curl))")

# 5. Energy Spectrum Analysis
println("\n5. Energy Spectrum Analysis")
println("-" * 27)

function compute_vector_energy_spectrum(cfg, Slm, Tlm)
    """Compute energy spectrum for vector field"""
    lmax = get_lmax(cfg)
    spheroidal_spectrum = zeros(lmax + 1)
    toroidal_spectrum = zeros(lmax + 1)
    
    for l in 0:lmax
        sph_energy = 0.0
        tor_energy = 0.0
        
        for m in 0:min(l, get_mmax(cfg))
            idx = lmidx(cfg, l, m) + 1  # Convert to 1-based
            if idx <= length(Slm)
                weight = (m == 0) ? 1.0 : 2.0  # Account for negative m
                sph_energy += weight * Slm[idx]^2
                tor_energy += weight * Tlm[idx]^2
            end
        end
        
        spheroidal_spectrum[l + 1] = sph_energy
        toroidal_spectrum[l + 1] = tor_energy
    end
    
    return spheroidal_spectrum, toroidal_spectrum
end

sph_spectrum, tor_spectrum = compute_vector_energy_spectrum(cfg, Slm, Tlm)
total_spectrum = sph_spectrum + tor_spectrum

println("Energy distribution by scale:")
for scale in [5, 10, 20, 32]
    if scale <= lmax
        cumulative = sum(total_spectrum[1:scale+1])
        percentage = cumulative / sum(total_spectrum) * 100
        println("  l ≤ $scale: $(@sprintf("%.1f", percentage))%")
    end
end

# 6. Helmholtz Decomposition Verification
println("\n6. Helmholtz Decomposition Verification")
println("-" * 39)

# The spheroidal component should represent the irrotational part
# The toroidal component should represent the solenoidal part

# Compute divergence and vorticity
function compute_divergence_vorticity(cfg, u, v)
    """Simplified divergence and vorticity computation"""
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    
    # This is a simplified finite difference approximation
    # For proper spectral computation, you'd use the spectral derivatives
    
    div_field = zeros(nlat, nphi)
    vort_field = zeros(nlat, nphi)
    
    # Simple finite differences (not rigorous on sphere)
    for i in 2:nlat-1
        for j in 1:nphi
            jp = (j == nphi) ? 1 : j + 1
            jm = (j == 1) ? nphi : j - 1
            
            # Approximate derivatives
            du_dlat = (u[i+1,j] - u[i-1,j]) / 2.0
            dv_dlon = (v[i,jp] - v[i,jm]) / 2.0
            
            # Very simplified - proper computation needs metric terms
            div_field[i,j] = du_dlat + dv_dlon
            vort_field[i,j] = dv_dlon - du_dlat  # Simplified
        end
    end
    
    return div_field, vort_field
end

div_field, vort_field = compute_divergence_vorticity(cfg, u, v)

max_div = maximum(abs.(div_field))
max_vort = maximum(abs.(vort_field))

println("Maximum divergence: $(@sprintf("%.2e", max_div))")
println("Maximum vorticity: $(@sprintf("%.2e", max_vort))")

# 7. Test with pure patterns
println("\n7. Pure Pattern Tests")
println("-" * 21)

# Test with purely spheroidal field (divergent, no rotation)
Slm_pure = copy(Slm)
Tlm_zero = zeros(size(Tlm))

u_sph, v_sph = synthesize_vector(cfg, Slm_pure, Tlm_zero)
println("Pure spheroidal field created (should be irrotational)")

# Test with purely toroidal field (rotational, no divergence)  
Slm_zero = zeros(size(Slm))
Tlm_pure = copy(Tlm)

u_tor, v_tor = synthesize_vector(cfg, Slm_zero, Tlm_pure)
println("Pure toroidal field created (should be solenoidal)")

# 8. Save results
println("\n8. Saving Results")
println("-" * 17)

try
    # Save energy spectra
    open("vector_energy_spectrum.dat", "w") do f
        println(f, "# l  Spheroidal_Energy  Toroidal_Energy  Total_Energy")
        for i in 1:length(sph_spectrum)
            l = i - 1
            println(f, "$l  $(sph_spectrum[i])  $(tor_spectrum[i])  $(total_spectrum[i])")
        end
    end
    println("Energy spectrum saved to 'vector_energy_spectrum.dat'")
    
    # Save original vector field
    open("vector_field.dat", "w") do f
        println(f, "# i  j  theta  phi  u  v")
        for i in 1:nlat
            theta = get_theta(cfg, i-1)
            for j in 1:nphi
                phi = get_phi(cfg, j)
                println(f, "$i  $j  $theta  $phi  $(u[i,j])  $(v[i,j])")
            end
        end
    end
    println("Vector field saved to 'vector_field.dat'")
    
catch e
    println("Could not save results: $e")
end

# 9. Cleanup
println("\n9. Cleanup")
println("-" * 10)

free_config(cfg)
println("Configuration freed successfully")
println("\nVector field demo completed!")