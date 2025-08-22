#!/usr/bin/env julia
# Script to set up documentation environment for SHTnsKit.jl

println("Setting up documentation environment for SHTnsKit.jl...")

# Change to docs directory
cd(@__DIR__)

# Activate the docs environment
using Pkg
Pkg.activate(".")

println("Installing documentation dependencies...")

# Add core documentation packages
Pkg.add([
    "Documenter",
    "Literate", 
    "Plots",
    "BenchmarkTools"
])

# Add the parent package (SHTnsKit) from local path
println("Adding SHTnsKit package from parent directory...")
Pkg.develop(path="..")

# Resolve and instantiate
println("Resolving dependencies...")
Pkg.resolve()
Pkg.instantiate()

# Precompile everything
println("Precompiling packages...")
Pkg.precompile()

println(" Documentation environment setup complete!")
println()
println("To build the documentation:")
println("  cd docs/")
println("  julia --project=. make.jl")
println()
println("To serve locally (requires LiveServer.jl):")
println("  julia --project=. -e 'using LiveServer; serve(dir=\"build\")'")