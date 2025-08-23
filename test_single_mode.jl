# Load SHTnsKit directly to avoid compilation issues
include("src/SHTnsKit.jl")
using .SHTnsKit

# Test single (l=0, m=0) mode
println("Testing single mode (l=0, m=0)...")
config = create_gauss_config(2,2)

# For complex transforms, we need to use complex indexing
n_cplx = SHTnsKit._cplx_nlm(config)
lm_indices_cplx = SHTnsKit._cplx_lm_indices(config)

println("Complex nlm: $n_cplx")
println("Complex indices: $lm_indices_cplx")

shc = zeros(Complex{Float64}, n_cplx)
# Find index for (l=0, m=0) in complex indexing - should be index 1
idx_00 = 1  # Based on the output showing (0,0) is first
shc[idx_00] = 1.0 + 0.0im

spatial = zeros(Complex{Float64}, config.nlat, config.nphi) 
SHTnsKit.cplx_sh_to_spat!(config, shc, spatial)
shc_reconstructed = zeros(Complex{Float64}, n_cplx)
SHTnsKit.cplx_spat_to_sh!(config, spatial, shc_reconstructed)

original = shc[idx_00]
reconstructed = shc_reconstructed[idx_00]
ratio = reconstructed / original

println("Original coefficient: $original")
println("Reconstructed coefficient: $reconstructed") 
println("Reconstructed/Original ratio: $ratio")

# Test single (l=1, m=0) mode
println("\nTesting single mode (l=1, m=0)...")
shc = zeros(Complex{Float64}, n_cplx)
# From the output: (1,0) is at index 2
idx_10 = 2
shc[idx_10] = 1.0 + 0.0im

spatial = zeros(Complex{Float64}, config.nlat, config.nphi) 
SHTnsKit.cplx_sh_to_spat!(config, shc, spatial)
shc_reconstructed = zeros(Complex{Float64}, n_cplx)
SHTnsKit.cplx_spat_to_sh!(config, spatial, shc_reconstructed)

original = shc[idx_10]
reconstructed = shc_reconstructed[idx_10]
ratio = reconstructed / original

println("Original coefficient: $original")
println("Reconstructed coefficient: $reconstructed") 
println("Reconstructed/Original ratio: $ratio")
