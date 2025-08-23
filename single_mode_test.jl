#!/usr/bin/env julia --startup-file=no --compile=no

# Load SHTnsKit directly 
println("Loading SHTnsKit...")
include("src/SHTnsKit.jl")
using .SHTnsKit

println("Creating config...")
cfg = create_gauss_config(4, 4)

# Test a single mode (l=1, m=0) for complex transforms
println("Creating single mode test...")
sh_coeffs = zeros(Complex{Float64}, SHTnsKit._cplx_nlm(cfg))

# Find index for (l=3, m=0)
idx_target = 0
for (idx, (l, m)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
    if l == 3 && m == 0
        global idx_target = idx
        break
    end
end

if idx_target > 0
    # Set coefficient for (3,0) mode
    sh_coeffs[idx_target] = 1.0 + 0.0im
    
    println("Original coefficient for (l=3,m=0): ", sh_coeffs[idx_target])
    
    # Synthesize
    spatial = SHTnsKit.cplx_sh_to_spat(cfg, sh_coeffs)
    println("Synthesized spatial range: [$(minimum(abs.(spatial))), $(maximum(abs.(spatial)))]")
    
    # Analyze without the calibration scaling
    sh_reconstructed = zeros(Complex{Float64}, length(sh_coeffs))
    SHTnsKit._cplx_spat_to_sh_impl!(cfg, spatial, sh_reconstructed)
    
    println("Reconstructed coefficient (before calibration): ", sh_reconstructed[idx_target])
    println("Ratio (reconstructed/original): ", sh_reconstructed[idx_target] / sh_coeffs[idx_target])
    
    # Now apply full analysis with calibration
    sh_full = SHTnsKit.cplx_spat_to_sh(cfg, spatial)
    println("Reconstructed coefficient (after calibration): ", sh_full[idx_target])
    println("Final ratio (full/original): ", sh_full[idx_target] / sh_coeffs[idx_target])
    
    # Check what the calibration scale factor is
    scale = SHTnsKit._get_complex_scale!(cfg)
    println("Calibration scale factor: ", scale)
else
    println("Could not find (l=3, m=0) mode")
end
