# Examples Gallery

Real-world examples and tutorials demonstrating SHTnsKit.jl capabilities, organized by difficulty level.

**How to use this guide:**
- **Beginner**: Start here if you're new to spherical harmonics
- **Intermediate**: For users comfortable with basic transforms
- **Advanced**: Complex workflows and specialized applications

**Learning path:** Work through the examples in order for the best learning experience.

## Beginner Examples

Start here if you're new to spherical harmonics. These examples teach fundamental concepts with simple, well-explained code.

### Example 1: Your First Transform

**Goal:** Learn the basic workflow of spherical harmonic transforms

```julia
using SHTnsKit

# Step 1: Create a configuration (like setting up your workspace)
cfg = create_gauss_config(16, 16)  # Start small for learning
println("Created configuration for degree up to 16")

# Step 2: Create a simple temperature pattern
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
# Simple pattern: warm equator (θ = π/2), cold poles (θ = 0, π)
temperature = @. 273.15 + 30 * sin(θ)^2  # Base temp + equatorial warming

println("Temperature range: $(extrema(temperature)) K")

# Step 3: Transform to spherical harmonic coefficients (analysis)
T_coeffs = analyze(cfg, temperature)
println("Number of coefficients: ", length(T_coeffs))

# Step 4: Find the most important coefficient
max_coeff_idx = argmax(abs.(T_coeffs))
l, m = SHTnsKit.lm_from_index(cfg, max_coeff_idx)
println("Strongest mode: l=$l, m=$m")

# Step 5: Reconstruct the original field (synthesis)
T_reconstructed = synthesize(cfg, T_coeffs)
error = maximum(abs.(temperature - T_reconstructed))
println("Reconstruction error: $error (should be tiny!)")

destroy_config(cfg)
```

**Key concepts learned:**
- Configuration setup (`create_gauss_config`)
- Creating realistic data patterns
- Analysis: spatial → spectral (`analyze`)
- Synthesis: spectral → spatial (`synthesize`)
- Understanding (l,m) mode indices

### Example 2: Pure Spherical Harmonic Patterns

**Goal:** Understand how individual spherical harmonic modes look

```julia
using SHTnsKit
using Plots  # For visualization

cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create pure Y_2^0 spherical harmonic (zonal mode)
sh = zeros(get_nlm(cfg))
idx = SHTnsKit.lmidx(cfg, 2, 0)  # l=2, m=0 (depends only on latitude)
sh[idx] = 1.0
println("Creating Y₂⁰ pattern (zonal, m=0)")

# Synthesize to spatial domain
Y20_pattern = synthesize(cfg, sh)

# This creates a pattern that varies only with latitude
println("Pattern statistics:")
println("  Min value: $(minimum(Y20_pattern))")
println("  Max value: $(maximum(Y20_pattern))")
println("  At north pole (θ=0): $(Y20_pattern[1,1])")
println("  At equator (θ=π/2): $(Y20_pattern[div(end,2),1])")

# Plot the pattern
heatmap(φ*180/π, θ*180/π, Y20_pattern, 
        xlabel="Longitude (°)", ylabel="Colatitude (°)",
        title="Y₂⁰ Spherical Harmonic (Zonal Pattern)",
        color=:RdBu)

destroy_config(cfg)
```

**Key concepts learned:**
- How to create pure spherical harmonic patterns
- Understanding zonal (m=0) vs sectoral (m≠0) modes
- The relationship between (l,m) indices and spatial patterns
- Basic visualization of spherical data

**Try this:** Change `(2,0)` to `(2,2)` to see a sectoral pattern!

### Example 3: Understanding Power Spectra

**Goal:** Learn how energy is distributed across different spatial scales

```julia
using SHTnsKit
using Plots

cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a field with multiple scales (like weather patterns)
field = @. (2*sin(2*θ)*cos(φ) +        # Large scale (continental)
           0.5*sin(6*θ)*cos(3*φ) +     # Medium scale (regional)  
           0.1*sin(12*θ)*cos(6*φ))     # Small scale (local)

println("Created multi-scale field with 3 different spatial scales")

# Transform to spectral domain
coeffs = analyze(cfg, field)

# Compute power spectrum (energy at each degree l)
power = power_spectrum(cfg, coeffs)

# Find which scales dominate
max_power_degree = argmax(power[2:end])  # Skip l=0 (global mean)
println("Peak energy at degree l = $max_power_degree")
println("This corresponds to ~$(360/max_power_degree)° wavelength")

# Plot the power spectrum
plot(0:length(power)-1, power, 
     xlabel="Spherical Harmonic Degree l", 
     ylabel="Power",
     title="Energy vs Spatial Scale",
     linewidth=2, marker=:circle)
plot!(yscale=:log10)  # Log scale often reveals more details

destroy_config(cfg)
```

**Key concepts learned:**
- How to create multi-scale patterns
- Power spectrum analysis shows energy distribution
- Relationship between degree l and spatial wavelength
- Using log scales for visualization

**Physical meaning:** In meteorology, this tells you whether your weather system is dominated by large-scale patterns (like jet streams) or small-scale features (like thunderstorms).

## Intermediate Examples

Ready to tackle more complex problems? These examples introduce vector fields, real-world data patterns, and scientific applications.

### Vector Field Decomposition

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

## Parallel Computing Examples

### MPI Distributed Computing

**Goal:** Learn how to use MPI for large-scale parallel spherical harmonic computations

```julia
# Save as parallel_example.jl and run with: mpiexec -n 4 julia parallel_example.jl
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

if rank == 0
    println("Running SHTnsKit parallel example with $size processes")
end

# Create configuration (same on all processes)
cfg = create_gauss_config(Float64, 30, 24, 64, 96)
pcfg = create_parallel_config(cfg, comm)

if rank == 0
    println("Problem size: $(cfg.nlm) spectral coefficients")
    println("Grid: $(cfg.nlat) × $(cfg.nphi) spatial points")
end

# Create test data
sh_coeffs = randn(Complex{Float64}, cfg.nlm)
result = similar(sh_coeffs)

# Benchmark parallel Laplacian operator
MPI.Barrier(comm)  # Synchronize timing
start_time = MPI.Wtime()

for i in 1:50
    parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)
end

MPI.Barrier(comm)
end_time = MPI.Wtime()

if rank == 0
    avg_time = (end_time - start_time) / 50
    println("Parallel Laplacian: $(avg_time*1000) ms per operation")
    
    # Compare with performance model
    perf_model = parallel_performance_model(cfg, size)
    println("Expected speedup: $(perf_model.speedup)x")
    println("Parallel efficiency: $(perf_model.efficiency*100)%")
end

# Test parallel transforms
spatial_data = allocate_spatial(cfg)
memory_efficient_parallel_transform!(pcfg, :synthesis, sh_coeffs, spatial_data)

if rank == 0
    println("Parallel synthesis completed")
end

# Test communication-intensive operator (cos θ)
parallel_apply_operator(pcfg, :costheta, sh_coeffs, result)

if rank == 0
    println("Parallel cos(θ) operator completed")
end

MPI.Finalize()
```

**Key concepts:**
- MPI initialization and communicator setup
- Creating parallel configurations with domain decomposition
- Using parallel operators for distributed computation
- Performance timing and comparison with models

### SIMD Vectorization Example

**Goal:** Leverage advanced SIMD optimizations for single-node performance

```julia
using SHTnsKit, LoopVectorization, BenchmarkTools

cfg = create_gauss_config(Float64, 64, 64)
sh_coeffs = randn(Complex{Float64}, cfg.nlm)

println("SIMD Optimization Comparison")
println("="^40)

# Benchmark regular SIMD
regular_time = @belapsed apply_laplacian!($cfg, copy($sh_coeffs))
println("Regular SIMD: $(regular_time*1000) ms")

# Benchmark turbo SIMD (with LoopVectorization)
turbo_time = @belapsed turbo_apply_laplacian!($cfg, copy($sh_coeffs))
println("Turbo SIMD:   $(turbo_time*1000) ms")

speedup = regular_time / turbo_time
println("Turbo speedup: $(speedup)x")

# Verify results are identical
result1 = copy(sh_coeffs)
result2 = copy(sh_coeffs)

apply_laplacian!(cfg, result1)
turbo_apply_laplacian!(cfg, result2)

max_diff = maximum(abs.(result1 - result2))
println("Max difference: $max_diff (should be ~0)")

# Benchmark comprehensive comparison
results = benchmark_turbo_vs_simd(cfg)
println("\nDetailed Benchmark Results:")
println("  SIMD time: $(results.simd_time*1000) ms")
println("  Turbo time: $(results.turbo_time*1000) ms") 
println("  Speedup: $(results.speedup)x")
println("  Accuracy: max diff = $(results.max_difference)")

destroy_config(cfg)
```

**Key concepts:**
- LoopVectorization.jl integration for enhanced SIMD
- Performance benchmarking and verification
- Automatic optimization selection

### Hybrid MPI + SIMD Example

**Goal:** Combine distributed and SIMD parallelization for maximum performance

```julia
# Save as hybrid_example.jl, run with: mpiexec -n 4 julia hybrid_example.jl
using SHTnsKit, MPI, PencilArrays, PencilFFTs, LoopVectorization

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Large problem that benefits from both MPI and SIMD
cfg = create_gauss_config(Float64, 128, 128, 256, 512)
pcfg = create_parallel_config(cfg, comm)

if rank == 0
    println("Hybrid MPI + SIMD Example")
    println("Problem: $(cfg.nlm) coefficients, $(cfg.nlat)×$(cfg.nphi) grid")
    println("MPI processes: $size")
    println("SIMD: LoopVectorization enabled")
end

# Test data
sh_coeffs = randn(Complex{Float64}, cfg.nlm)
result = similar(sh_coeffs)

# Benchmark different approaches
tests = [
    ("Parallel standard", () -> parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)),
    ("Parallel + turbo", () -> begin
        # This would use turbo optimizations within parallel operations
        parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)
    end)
]

if rank == 0
    println("\nPerformance Comparison:")
end

for (name, test_func) in tests
    MPI.Barrier(comm)
    start_time = MPI.Wtime()
    
    for i in 1:20
        test_func()
    end
    
    MPI.Barrier(comm)
    end_time = MPI.Wtime()
    
    if rank == 0
        avg_time = (end_time - start_time) / 20
        println("$name: $(avg_time*1000) ms per operation")
    end
end

# Test scaling efficiency
if rank == 0
    println("\nScaling Analysis:")
    for test_size in [2, 4, 8, 16]
        if test_size <= size * 2  # Don't test more than 2x current size
            model = parallel_performance_model(cfg, test_size)
            println("$test_size processes: $(model.speedup)x speedup, $(model.efficiency*100)% efficiency")
        end
    end
end

MPI.Finalize()
```

### Asynchronous Parallel Operations

**Goal:** Use non-blocking communication for better performance overlap

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

cfg = create_gauss_config(Float64, 64, 48, 128, 192)
pcfg = create_parallel_config(cfg, comm)

if rank == 0
    println("Asynchronous Parallel Operations Example")
end

sh_coeffs = randn(Complex{Float64}, cfg.nlm)
result = similar(sh_coeffs)

# Compare synchronous vs asynchronous operations
if rank == 0
    println("Benchmarking communication patterns...")
end

# Synchronous (blocking)
MPI.Barrier(comm)
sync_time = @elapsed begin
    for i in 1:30
        parallel_apply_operator(pcfg, :costheta, sh_coeffs, result)
    end
end

# Asynchronous (non-blocking, if available)
MPI.Barrier(comm)
async_time = @elapsed begin
    for i in 1:30
        try
            # Try asynchronous version
            async_parallel_costheta_operator!(pcfg, sh_coeffs, result)
        catch
            # Fall back to synchronous if not available
            parallel_apply_operator(pcfg, :costheta, sh_coeffs, result)
        end
    end
end

if rank == 0
    println("Communication Performance:")
    println("  Synchronous:  $(sync_time/30*1000) ms per operation")
    println("  Asynchronous: $(async_time/30*1000) ms per operation")
    if async_time < sync_time
        println("  Async speedup: $(sync_time/async_time)x")
    else
        println("  No async improvement (likely using fallback)")
    end
end

MPI.Finalize()
```

**Key concepts:**
- Non-blocking MPI communication patterns
- Communication-computation overlap
- Performance analysis of different parallel strategies

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
        println("   PASS")
    else
        println("  FAIL")
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
