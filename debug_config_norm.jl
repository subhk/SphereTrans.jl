#!/usr/bin/env julia --startup-file=no --compile=no

# Check what normalization the config uses
include("src/SHTnsKit.jl")
using .SHTnsKit

cfg = create_gauss_config(4, 4)
println("Config normalization: $(cfg.norm)")
println("SHT_SCHMIDT = $(SHTnsKit.SHT_SCHMIDT)")
println("SHT_ORTHONORMAL = $(SHTnsKit.SHT_ORTHONORMAL)")
println("SHT_FOURPI = $(SHTnsKit.SHT_FOURPI)")

destroy_config(cfg)