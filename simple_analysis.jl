#!/usr/bin/env julia

"""
Simple analysis script for SHTnsKit.jl performance and type stability
"""

# Basic analysis without heavy dependencies
println("=== SHTnsKit.jl Analysis ===")
println("Julia version: ", VERSION)
println()

# Load the module directly from source
push!(LOAD_PATH, "src")

# Load source files directly to analyze
include("src/types.jl")
include("src/gauss_legendre.jl")
include("src/fft_utils.jl") 
include("src/core_transforms.jl")
include("src/utilities.jl")
include("src/threading.jl")

# Simple type stability test
println("1. Basic type analysis...")
cfg = SHTnsConfig{Float64}()
cfg.lmax = 15
cfg.mmax = 15
cfg.mres = 1
cfg.nlm = 256
cfg.nlat = 32
cfg.nphi = 64
cfg.grid_type = SHT_GAUSS
cfg.norm = SHT_ORTHONORMAL

println("Config type: ", typeof(cfg))
println("Field types:")
for field in fieldnames(typeof(cfg))
    println("  $field: ", typeof(getfield(cfg, field)))
end
println()

# Memory allocation test
println("2. Memory allocation test...")
test_array = Matrix{Float64}(undef, 32, 64)
test_vector = Vector{Float64}(undef, 256)

println("Matrix allocation: ", @allocated Matrix{Float64}(undef, 32, 64))
println("Vector allocation: ", @allocated Vector{Float64}(undef, 256))
println()

# Function analysis
println("3. Function performance test...")
function test_simple_operation(arr::Matrix{Float64})
    return sum(arr .^ 2)
end

println("Simple operation allocation: ", @allocated test_simple_operation(test_array))

# Test threading functions
println("4. Threading analysis...")
println("Threading enabled: ", SHTnsKit.get_threading())
println()

println("Analysis complete!")