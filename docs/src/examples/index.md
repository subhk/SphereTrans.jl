# Examples Gallery

Real-world examples and tutorials demonstrating SHTnsKit.jl capabilities.

## Basic Examples

### Simple Scalar Transform

```julia
using SHTnsKit

# Create configuration for moderate resolution
cfg = create_gauss_config(32, 32)

# Create a test field: temperature variation
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
temperature = @. 273.15 + 30 * cos(2*θ) * (1 + 0.3 * cos(3*φ))

# Transform to spectral domain
T_lm = analyze(cfg, temperature)
println("Dominant modes: ", findmax(abs.(T_lm)))

# Reconstruct and verify accuracy
T_reconstructed = synthesize(cfg, T_lm)
error = norm(temperature - T_reconstructed)
println("Reconstruction error: $error")

destroy_config(cfg)
```

### Spherical Harmonic Visualization

```julia
using SHTnsKit
using Plots

cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create pure Y_4^2 spherical harmonic
sh = zeros(get_nlm(cfg))
idx = lmidx(cfg, 4, 2)  # l=4, m=2
sh[idx] = 1.0

# Synthesize to spatial domain
Y42 = synthesize(cfg, sh)

# Plot on sphere (requires plotting package)
surface(φ*180/π, θ*180/π, Y42, 
        xlabel="Longitude (°)", ylabel="Colatitude (°)",
        title="Y₄² Spherical Harmonic")

destroy_config(cfg)
```

## Fluid Dynamics Applications

### Vorticity-Divergence Decomposition

```julia
using SHTnsKit

cfg = create_gauss_config(64, 64)

# Create a realistic atmospheric flow pattern
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
nlat, nphi = size(θ)

# Jet stream pattern with vortices
u = @. 20 * sin(2θ) * (1 + 0.4 * cos(4φ))  # Zonal wind
v = @. 5 * cos(3θ) * sin(2φ)                # Meridional wind

S_lm, T_lm = analyze_vector(cfg, u, v)

# Spatial divergence and vorticity
divergence = SHTnsKit.spatial_divergence(cfg, u, v)
vorticity  = SHTnsKit.spatial_vorticity(cfg, u, v)

println("Max vorticity: ", maximum(abs.(vorticity)))
println("Max divergence: ", maximum(abs.(divergence)))

# Reconstruct original velocity
u_recon, v_recon = synthesize_vector(cfg, S_lm, T_lm)
velocity_error = norm(u - u_recon) + norm(v - v_recon)
println("Velocity reconstruction error: $velocity_error")

destroy_config(cfg)
```

### Stream Function from Vorticity

```julia
using SHTnsKit
using LinearAlgebra

cfg = create_gauss_config(48, 48)

# Create vorticity field (e.g., from observations)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
vorticity = @. exp(-((θ - π/2)^2 + (φ - π)^2) / 0.5^2) * sin(4φ)

# Transform vorticity to spectral domain
ζ_lm = analyze(cfg, vorticity)

# Solve ∇²ψ = ζ for stream function ψ
# In spectral domain: -l(l+1) ψ_lm = ζ_lm
ψ_lm = similar(ζ_lm)
for i in 1:get_nlm(cfg)
    l, m = lm_from_index(cfg, i)
    if l > 0
        ψ_lm[i] = -ζ_lm[i] / (l * (l + 1))
    else
        ψ_lm[i] = 0.0  # l=0 mode: constant not uniquely determined
    end
end

u_stream, v_stream = synthesize_vector(cfg, zero(ψ_lm), ψ_lm)

# Convert stream function to spatial domain
stream_function = synthesize(cfg, ψ_lm)

println("Stream function range: ", extrema(stream_function))
println("Max velocity from stream: ", maximum(sqrt.(u_stream.^2 + v_stream.^2)))

destroy_config(cfg)
```

## Geophysics Applications

### Gravitational Potential Analysis

```julia
using SHTnsKit

cfg = create_gauss_config(72, 72)  # High resolution for Earth

# Simulate Earth's gravitational field coefficients
# (In practice, these would come from satellite measurements)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create realistic gravity anomalies
# J₂ (Earth's oblate shape) + smaller harmonics
gravity_field = @. -9.81 * (1 + 0.001082 * (1.5 * cos(θ)^2 - 0.5) + 
                           0.0001 * sin(3θ) * cos(2φ))

# Analyze gravity field
g_lm = analyze(cfg, gravity_field)

# Extract major components
J2_coeff = g_lm[lmidx(cfg, 2, 0)]  # J₂ term
println("J₂ coefficient: $J2_coeff")

# Compute power spectrum
power = power_spectrum(cfg, g_lm)

# Plot power vs degree
using Plots
plot(0:length(power)-1, log10.(power), 
     xlabel="Spherical Harmonic Degree l", 
     ylabel="log₁₀(Power)",
     title="Gravity Field Power Spectrum")

destroy_config(cfg)
```

### Magnetic Field Modeling

```julia
using SHTnsKit

cfg = create_gauss_config(48, 48)

# Simulate magnetic field measurements (3 components)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Dipole + quadrupole + small-scale fields
Br = @. 30000 * cos(θ) * (1 + 0.1 * cos(2θ) * sin(φ))  # Radial
Bθ = @. 15000 * sin(θ) * (1 - 0.05 * sin(3φ))          # Colatitude  
Bφ = @. 5000 * sin(θ) * cos(θ) * cos(2φ)                # Azimuthal

# Magnetic field is potential: B = -∇V
# So horizontal components relate to potential derivatives
# This is a simplified analysis - real magnetic modeling is more complex

# Analyze radial component (related to potential)
V_lm = analyze(cfg, -Br / 30000)  # Normalized

# Compute horizontal components from potential (spheroidal only)
Bθ_computed, Bφ_computed = synthesize_vector(cfg, V_lm, zeros(V_lm))

# Compare with input
θ_error = norm(Bθ/15000 - Bθ_computed) / norm(Bθ/15000)
φ_error = norm(Bφ/5000 - Bφ_computed) / norm(Bφ/5000)

println("Magnetic field modeling errors:")
println("θ component: $θ_error")  
println("φ component: $φ_error")

destroy_config(cfg)
```

## Climate Science Applications

### Temperature Anomaly Analysis

```julia
using SHTnsKit
using Statistics

cfg = create_gauss_config(64, 64)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Simulate monthly temperature anomalies
n_months = 120  # 10 years
anomalies = []

for month in 1:n_months
    # Seasonal cycle + trend + random variations
    seasonal = @. 5 * cos(2π * month / 12) * cos(θ)
    trend = @. 0.01 * month * ones(size(θ))
    random = @. 2 * randn(size(θ)...) * exp(-3 * (θ - π/2)^2)
    
    temp_anomaly = seasonal + trend + random
    push!(anomalies, temp_anomaly)
end

# Analyze each month
monthly_spectra = []
for anomaly in anomalies
    T_lm = analyze(cfg, anomaly)
    push!(monthly_spectra, T_lm)
end

# Compute time-averaged power spectrum
avg_power = mean([power_spectrum(cfg, spectrum) for spectrum in monthly_spectra])

# Find dominant modes
max_power_idx = argmax(avg_power[2:end]) + 1  # Skip l=0
println("Dominant mode: l = $(max_power_idx-1)")
println("Power: $(avg_power[max_power_idx])")

# Trend analysis - extract l=0,m=0 component (global mean)
global_means = [spectrum[1] for spectrum in monthly_spectra]
using Plots
plot(1:n_months, global_means, 
     xlabel="Month", ylabel="Global Mean Anomaly",
     title="Global Temperature Trend")

destroy_config(cfg)
```

### Precipitation Pattern Analysis

```julia
using SHTnsKit

cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Seasonal precipitation patterns
# Summer: ITCZ near equator, winter: shifted south
precip_summer = @. max(0, 10 * exp(-5 * (θ - π/2)^2) * 
                      (1 + 0.3 * cos(2φ)))
precip_winter = @. max(0, 8 * exp(-5 * (θ - π/2 - 0.2)^2) * 
                      (1 + 0.2 * cos(3φ)))

# Transform to spectral domain
P_summer_lm = analyze(cfg, precip_summer)
P_winter_lm = analyze(cfg, precip_winter)

# Compute seasonal difference
seasonal_diff_lm = P_summer_lm - P_winter_lm
seasonal_diff = synthesize(cfg, seasonal_diff_lm)

# Power spectrum of seasonal difference
diff_power = power_spectrum(cfg, seasonal_diff_lm)

println("Seasonal precipitation analysis:")
println("Summer total: ", sum(precip_summer))
println("Winter total: ", sum(precip_winter))
println("Max seasonal difference: ", maximum(abs.(seasonal_diff)))

# Find regions of maximum seasonal variation
max_diff_locations = findall(abs.(seasonal_diff) .> 0.8 * maximum(abs.(seasonal_diff)))
println("High variability regions: $(length(max_diff_locations)) grid points")

destroy_config(cfg)
```

## Advanced Applications

### Multiscale Analysis

```julia
using SHTnsKit

# Create different resolution configurations
cfgs = [create_gauss_config(l, l) for l in [16, 32, 64, 128]]

# Create test field with multiple scales
θ, φ = SHTnsKit.create_coordinate_matrices(cfgs[end])  # Use highest resolution grid
field = @. (sin(2θ) * cos(φ) +           # Large scale
           0.3 * sin(8θ) * cos(4φ) +     # Medium scale
           0.1 * sin(16θ) * cos(8φ))     # Small scale

# Analyze at different resolutions
powers = []
for (i, cfg) in enumerate(cfgs)
    # Interpolate field to current grid if needed
    θ_i, φ_i = SHTnsKit.create_coordinate_matrices(cfg)
    field_i = field[1:get_nlat(cfg), 1:get_nphi(cfg)]  # Simple subsampling
    
    # Analyze and compute power spectrum
    f_lm = analyze(cfg, field_i)
    power_i = power_spectrum(cfg, f_lm)
    push!(powers, power_i)
    
    println("Resolution $(get_lmax(cfg)): $(length(power_i)) modes")
end

# Compare power spectra
using Plots
p = plot(xlabel="Spherical Harmonic Degree", ylabel="Power", yscale=:log10)
for (i, power) in enumerate(powers)
    plot!(p, 0:length(power)-1, power, 
          label="lmax = $(get_lmax(cfgs[i]))", linewidth=2)
end
display(p)

# Cleanup
for cfg in cfgs
    destroy_config(cfg)
end
```

### Field Rotation and Coordinate Transformations

```julia
using SHTnsKit

cfg = create_gauss_config(32, 32)

# Create field in one coordinate system
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
original_field = @. sin(3θ) * cos(2φ)

# Rotate coordinates (simulate different observation viewpoint)
α, β, γ = π/4, π/6, π/8  # Euler angles

f_lm = analyze(cfg, original_field)
f_rot = copy(f_lm)
rotate_real!(cfg, f_rot; alpha=α, beta=β, gamma=γ)
rotated_field = synthesize(cfg, f_rot)

destroy_config(cfg)
```

## High-Performance Examples

### Multi-threaded Batch Processing

```julia
using SHTnsKit
using Base.Threads

cfg = create_gauss_config(64, 64)
set_optimal_threads!()

# Large batch of fields to process
n_batch = 1000
input_fields = [rand(get_nlat(cfg), get_nphi(cfg)) for _ in 1:n_batch]

# Process with threading
println("Processing $n_batch fields with $(nthreads()) Julia threads...")
results = Vector{Float64}(undef, n_batch)

@time @threads for i in 1:n_batch
    # Each thread gets its own work
    field = input_fields[i]
    
    # Transform and compute some property
    sh = analyze(cfg, field)
    power = power_spectrum(cfg, sh)
    
    # Store result
    results[i] = sum(power)  # Total energy
end

println("Mean energy per field: ", mean(results))
println("Energy std dev: ", std(results))

destroy_config(cfg)
```

## Validation and Testing Examples

### Analytical Test Cases

```julia
using SHTnsKit

cfg = create_gauss_config(24, 24)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Test Case 1: Pure spherical harmonics
test_cases = [
    (l=0, m=0, Y=(θ,φ) -> 1/sqrt(4π)),
    (l=1, m=-1, Y=(θ,φ) -> sqrt(3/(8π)) * sin.(θ) .* sin.(φ)),  
    (l=1, m=0, Y=(θ,φ) -> sqrt(3/(4π)) * cos.(θ)),
    (l=1, m=1, Y=(θ,φ) -> -sqrt(3/(8π)) * sin.(θ) .* cos.(φ)),
    (l=2, m=0, Y=(θ,φ) -> sqrt(5/(16π)) * (3*cos.(θ).^2 .- 1))
]

println("Analytical validation tests:")
for (i, case) in enumerate(test_cases)
    # Create analytical field
    Y_analytical = case.Y(θ, φ)
    
    # Transform to spectral
    sh = analyze(cfg, Y_analytical)
    
    # Check that only the correct coefficient is non-zero
expected_idx = lmidx(cfg, case.l, case.m)
    
    # Find largest coefficient
    max_idx = argmax(abs.(sh))
    max_val = sh[max_idx]
    
    println("Test $i: l=$(case.l), m=$(case.m)")
    println("  Expected index: $expected_idx, Found: $max_idx")
    println("  Coefficient value: $max_val")
    
    if max_idx == expected_idx
        println("  ✓ PASS")
    else
        println("  ✗ FAIL")
    end
end

destroy_config(cfg)
```

### Numerical Accuracy Tests

```julia
using SHTnsKit

# Test different resolutions and grid types
resolutions = [16, 32, 64]
grid_types = [:gauss, :regular]

println("Accuracy vs Resolution Test:")
for grid_type in grid_types
    println("\n$grid_type Grid:")
    
    for lmax in resolutions
        cfg = grid_type == :gauss ? 
              create_gauss_config(lmax, lmax) : 
              create_regular_config(lmax, lmax)
        
        # Random test field
        sh_original = rand(get_nlm(cfg))
        
        # Round-trip transform
        spatial = synthesize(cfg, sh_original)
        sh_recovered = analyze(cfg, spatial)
        
        # Measure error
        error = norm(sh_original - sh_recovered) / norm(sh_original)
        
        println("  lmax=$lmax: error = $error")
        
        destroy_config(cfg)
    end
end
```

These examples demonstrate the full range of SHTnsKit.jl capabilities from basic transforms to advanced scientific applications. Each example can serve as a starting point for your specific research needs.
