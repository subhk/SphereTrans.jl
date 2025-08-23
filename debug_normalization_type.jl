#!/usr/bin/env julia --startup-file=no --compile=no

# Debug what normalization type we're using
include("src/SHTnsKit.jl")
using .SHTnsKit

println("=== Normalization Type Analysis ===")

cfg = create_gauss_config(4, 4)
println("Config normalization: $(cfg.norm)")

# Check if we have SHT_REAL_NORM or SHT_COMPLEX_NORM
if hasfield(typeof(SHTnsKit), :SHT_REAL_NORM)
    println("SHT_REAL_NORM available")
else
    println("SHT_REAL_NORM not defined")
end

# Check available norm types
for name in names(SHTnsKit)
    if occursin("SHT_", string(name)) && occursin("NORM", string(name))
        println("Available: $name = $(getfield(SHTnsKit, name))")
    end
end

destroy_config(cfg)