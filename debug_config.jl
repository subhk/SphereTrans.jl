#!/usr/bin/env julia --startup-file=no --compile=no

# Load SHTnsKit directly 
println("Loading SHTnsKit...")
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Creating config...")
cfg = create_gauss_config(4, 4)
println("Config created:")
println(cfg)
println("nlat = $(get_nlat(cfg))")
println("nphi = $(get_nphi(cfg))")

# Check what the configuration should be based on the grid type validation
# From the validation: nlat > lmax, nphi >= 2*mmax + 1
println("Required: nlat > lmax ($(cfg.lmax)), so nlat > 4")
println("Required: nphi >= 2*mmax + 1 = 2*$(cfg.mmax) + 1 = $(2*cfg.mmax + 1)")
