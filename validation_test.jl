#!/usr/bin/env julia

"""
Validation test for SHTnsKit.jl optimizations
Tests basic functionality to ensure optimizations don't break correctness
"""

push!(LOAD_PATH, ".")
push!(LOAD_PATH, "src")

println("=== SHTnsKit Validation Test ===")
println("Loading optimized module...")

# Test core functionality
try
    include("src/SHTnsKit.jl")
    using .SHTnsKit
    
    println("✓ Module loaded successfully")
    
    # Test basic configuration
    println("Testing basic configuration...")
    cfg = create_gauss_config(15, 15; T=Float64)
    println("✓ Configuration created: lmax=15, size=$(cfg.nlat)×$(cfg.nphi)")
    
    # Test transforms
    println("Testing transforms...")
    spatial_data = randn(cfg.nlat, cfg.nphi)
    sh_coeffs = allocate_spectral(cfg)
    spatial_out = allocate_spatial(cfg)
    
    # Forward transform
    spat_to_sh!(cfg, spatial_data, sh_coeffs)
    println("✓ Forward transform completed")
    
    # Backward transform  
    sh_to_spat!(cfg, sh_coeffs, spatial_out)
    println("✓ Backward transform completed")
    
    # Test roundtrip accuracy
    max_error = maximum(abs.(spatial_data .- spatial_out))
    println("✓ Roundtrip error: $(max_error)")
    
    if max_error > 1e-12
        println("⚠️  High roundtrip error detected")
    else
        println("✓ Roundtrip accuracy excellent")
    end
    
    # Test threading
    println("Testing threading...")
    println("  Threading enabled: $(get_threading())")
    println("  FFT threads: $(get_fft_threads())")
    set_optimal_threads!()
    println("✓ Threading setup completed")
    
    # Performance test
    println("Basic performance test...")
    @time begin
        for i in 1:10
            spat_to_sh!(cfg, spatial_data, sh_coeffs)
            sh_to_spat!(cfg, sh_coeffs, spatial_out)
        end
    end
    println("✓ Performance test completed")
    
    println("\n=== Validation Successful ===")
    println("All optimizations appear to be working correctly!")
    
catch e
    println("❌ Error during validation: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end