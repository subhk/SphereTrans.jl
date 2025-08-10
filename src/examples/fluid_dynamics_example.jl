"""
Fluid Dynamics Example using SHTnsKit.jl
=========================================

This example demonstrates how to use SHTnsKit.jl for fluid dynamics simulations,
specifically showing:
- Vector field decomposition (velocity -> vorticity + divergence)
- Stream function computation
- Spectral differentiation on the sphere
- Energy spectrum analysis
"""

using SHTnsKit
using LinearAlgebra
using Printf

println("Fluid Dynamics Example with SHTnsKit.jl")
println("=" * 40)

# Physical parameters
const R_earth = 6.371e6  # Earth radius in meters
const Omega = 7.292e-5   # Earth rotation rate in rad/s

# Set up spherical harmonic configuration
lmax = 32
cfg = create_gauss_config(lmax, lmax)
nlat, nphi = get_nlat(cfg), get_nphi(cfg)
nlm = get_nlm(cfg)

println("Grid: $nlat × $nphi points")
println("Spectral: $(lmax+1)² = $(nlm) coefficients")

# 1. Create a realistic atmospheric flow pattern
println("\n1. Creating Test Velocity Field")
println("-" * 32)

function create_atmospheric_flow(cfg)
    """Create a realistic atmospheric flow with jet streams and vortices."""
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    u = zeros(nlat, nphi)  # Zonal velocity (eastward)
    v = zeros(nlat, nphi)  # Meridional velocity (northward)
    
    for i in 1:nlat
        theta = get_theta(cfg, i-1)
        lat = π/2 - theta  # Convert to latitude
        
        for j in 1:nphi
            lon = get_phi(cfg, j)
            
            # Jet stream (strong westerly winds at mid-latitudes)
            jet_strength = 30.0  # m/s
            u[i, j] = jet_strength * exp(-(lat - π/6)^2 / (2*(π/12)^2))
            u[i, j] += jet_strength * exp(-(lat + π/6)^2 / (2*(π/12)^2))
            
            # Add some meridional component (Rossby waves)
            wave_amplitude = 5.0  # m/s
            wave_number = 3
            v[i, j] = wave_amplitude * sin(wave_number * lon) * cos(2*lat)
            
            # Add small-scale vortices
            if abs(lat - π/8) < π/16 && abs(lon - π) < π/8
                vortex_strength = 10.0
                r = sqrt((lat - π/8)^2 + (lon - π)^2)
                if r < π/16
                    circulation = vortex_strength * exp(-r^2 / (2*(π/32)^2))
                    u[i, j] += -circulation * (lat - π/8) / r
                    v[i, j] += circulation * (lon - π) / r
                end
            end
        end
    end
    
    return u, v
end

u, v = create_atmospheric_flow(cfg)
max_speed = maximum(sqrt.(u.^2 + v.^2))
println("Maximum wind speed: $(max_speed:.1f) m/s")

# 2. Decompose velocity into spheroidal and toroidal components
println("\n2. Vector Field Decomposition")
println("-" * 29)

# Transform velocity to spectral space
Slm, Tlm = analyze_vector(cfg, u, v)

# Compute energy in each component
total_energy = sum(Slm.^2 + Tlm.^2)
spheroidal_energy = sum(Slm.^2)
toroidal_energy = sum(Tlm.^2)

println("Total kinetic energy: $(total_energy:.2e)")
println("Spheroidal (divergent) component: $(spheroidal_energy/total_energy*100:.1f)%")
println("Toroidal (rotational) component: $(toroidal_energy/total_energy*100:.1f)%")

# 3. Compute vorticity and divergence
println("\n3. Vorticity and Divergence")
println("-" * 27)

function compute_vorticity_divergence(cfg, u, v)
    """Compute vorticity and divergence from velocity field."""
    # Get spheroidal and toroidal coefficients
    Slm, Tlm = analyze_vector(cfg, u, v)
    
    # Vorticity is related to toroidal component
    # Divergence is related to spheroidal component
    # For proper scaling, we need to apply the Laplacian operator
    
    # Simple approximation: synthesize gradient of streamfunction and velocity potential
    vorticity_approx = zeros(size(u))
    divergence_approx = zeros(size(u))
    
    # This is a simplified calculation - proper implementation would use
    # spectral derivatives with proper scaling factors
    for l in 0:get_lmax(cfg)
        for m in 0:min(l, get_mmax(cfg))
            idx = lmidx(cfg, l, m) + 1  # Convert to 1-based indexing
            if idx <= length(Slm) && l > 0
                # Scale by l(l+1) for proper differentiation
                scale_factor = l * (l + 1) / R_earth^2
                Slm[idx] *= scale_factor  # For divergence
                Tlm[idx] *= scale_factor  # For vorticity
            end
        end
    end
    
    # Convert back to physical space
    div_field = synthesize(cfg, Slm)
    vort_field = synthesize(cfg, Tlm)
    
    return vort_field, div_field
end

vorticity, divergence = compute_vorticity_divergence(cfg, u, v)

max_vorticity = maximum(abs.(vorticity))
max_divergence = maximum(abs.(divergence))

println("Maximum vorticity: $(max_vorticity:.2e) s⁻¹")
println("Maximum divergence: $(max_divergence:.2e) s⁻¹")

# 4. Energy spectrum analysis
println("\n4. Energy Spectrum Analysis")
println("-" * 27)

function compute_energy_spectrum(cfg, Slm, Tlm)
    """Compute kinetic energy spectrum."""
    lmax = get_lmax(cfg)
    energy_spectrum = zeros(lmax + 1)
    
    for l in 0:lmax
        energy_l = 0.0
        for m in 0:min(l, get_mmax(cfg))
            idx = lmidx(cfg, l, m) + 1  # Convert to 1-based
            if idx <= length(Slm)
                # Energy contribution from this mode
                if m == 0
                    energy_l += Slm[idx]^2 + Tlm[idx]^2
                else
                    energy_l += 2 * (Slm[idx]^2 + Tlm[idx]^2)  # Factor of 2 for m ≠ 0
                end
            end
        end
        energy_spectrum[l + 1] = energy_l
    end
    
    return energy_spectrum
end

energy_spectrum = compute_energy_spectrum(cfg, Slm, Tlm)
total_spectral_energy = sum(energy_spectrum)

println("Energy by scale:")
for decade in [1, 5, 10, 20, 32]
    if decade <= lmax
        cumulative_energy = sum(energy_spectrum[1:decade+1])
        percentage = cumulative_energy / total_spectral_energy * 100
        println("  l ≤ $decade: $(percentage:.1f)%")
    end
end

# 5. Spectral slope analysis
println("\n5. Spectral Slope Analysis")
println("-" * 26)

# Fit power law to energy spectrum (skip l=0 and very small scales)
valid_range = 2:min(20, lmax)
log_l = log10.(valid_range)
log_E = log10.(energy_spectrum[valid_range .+ 1])

# Simple linear fit
A = [ones(length(log_l)) log_l]
coeffs = A \ log_E
slope = coeffs[2]

println("Energy spectrum slope: $(slope:.2f)")
if slope < -2.5
    println("  → Steeper than Kolmogorov (-5/3), suggests strong dissipation")
elseif slope > -1.5
    println("  → Shallower than Kolmogorov (-5/3), suggests energy accumulation")
else
    println("  → Close to Kolmogorov (-5/3), indicates inertial range")
end

# 6. Demonstrate rotation and Coriolis effects
println("\n6. Coriolis Force Effects")
println("-" * 25)

function apply_coriolis_force(cfg, u, v, dt)
    """Apply Coriolis force to velocity field."""
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    u_new = copy(u)
    v_new = copy(v)
    
    for i in 1:nlat
        theta = get_theta(cfg, i-1)
        lat = π/2 - theta
        f = 2 * Omega * sin(lat)  # Coriolis parameter
        
        for j in 1:nphi
            # Coriolis force: F = -f × v
            u_new[i, j] = u[i, j] + f * v[i, j] * dt
            v_new[i, j] = v[i, j] - f * u[i, j] * dt
        end
    end
    
    return u_new, v_new
end

# Apply a small Coriolis correction
dt = 3600.0  # 1 hour in seconds
u_coriolis, v_coriolis = apply_coriolis_force(cfg, u, v, dt)

speed_change = maximum(sqrt.((u_coriolis - u).^2 + (v_coriolis - v).^2))
println("Maximum velocity change due to Coriolis force: $(speed_change:.3f) m/s")

# 7. GPU acceleration test (if available)
println("\n7. GPU Performance Test")
println("-" * 23)

try
    using CUDA
    if CUDA.functional()
        gpu_initialized = initialize_gpu(0, verbose=false)
        
        if gpu_initialized
            cfg_gpu = create_gpu_config(lmax, lmax)
            
            # Transfer data to GPU
            u_gpu = CUDA.CuArray(u)
            v_gpu = CUDA.CuArray(v)
            
            # Time GPU vector analysis
            println("Timing GPU vector analysis...")
            @time Slm_gpu, Tlm_gpu = analyze_vector(cfg_gpu, Array(u_gpu), Array(v_gpu))
            
            # Compare with CPU result
            gpu_error_S = maximum(abs.(Slm - Slm_gpu))
            gpu_error_T = maximum(abs.(Tlm - Tlm_gpu))
            
            println("GPU vs CPU differences: S=$(gpu_error_S:.2e), T=$(gpu_error_T:.2e)")
            
            cleanup_gpu(verbose=false)
            free_config(cfg_gpu)
        end
    else
        println("CUDA not functional")
    end
catch
    println("CUDA not available")
end

# 8. Summary and cleanup
println("\n8. Summary")
println("-" * 10)

println("Fluid dynamics analysis completed:")
println("  • Velocity field decomposed into spheroidal/toroidal components")
println("  • Energy spectrum computed and analyzed")
println("  • Spectral slope: $(slope:.2f)")
println("  • Coriolis effects demonstrated")
println("  • All computations performed using SHTnsKit.jl")

free_config(cfg)
println("\nCleanup completed successfully!")

# Optional: Save results for further analysis
try
    using JLD2
    @save "fluid_dynamics_results.jld2" u v Slm Tlm energy_spectrum vorticity divergence
    println("Results saved to 'fluid_dynamics_results.jld2'")
catch
    println("JLD2 not available - results not saved")
end