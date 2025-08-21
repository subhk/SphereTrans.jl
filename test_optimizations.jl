#!/usr/bin/env julia

using LinearAlgebra, FFTW, BenchmarkTools
include("src/SHTnsKit.jl")
using .SHTnsKit

function test_optimizations()
    println("=== Testing Optimizations ===")
    
    # Test configuration creation and basic operations
    cfg = create_gauss_config(8, 8)
    
    # Test memory allocations
    println("\n1. Memory allocation test:")
    coeffs = allocate_spectral(cfg)
    coeffs[1] = sqrt(4π)  # Y_0^0 harmonic
    
    spatial_out = allocate_spatial(cfg)
    
    # Measure allocations in optimized version
    allocs_before = @allocated sh_to_spat!(cfg, coeffs, spatial_out)
    println("  sh_to_spat! allocations: $allocs_before bytes")
    
    # Test correctness
    spatial_data = sh_to_spat(cfg, coeffs)
    expected = 1.0
    actual = spatial_data[1,1]
    error = abs(actual - expected)
    println("  Synthesis accuracy: error = $error")
    
    if error > 1e-12
        @warn "Accuracy degraded: expected $expected, got $actual"
    else
        println("  ✓ Accuracy maintained")
    end
    
    # Test roundtrip
    coeffs_out = spat_to_sh(cfg, spatial_data)
    roundtrip_error = abs(coeffs_out[1] - coeffs[1])
    println("  Roundtrip error: $roundtrip_error")
    
    if roundtrip_error > 1e-12
        @warn "Roundtrip accuracy degraded"
    else
        println("  ✓ Roundtrip accuracy maintained")
    end
    
    # Benchmark performance
    println("\n2. Performance benchmark:")
    
    # Benchmark synthesis
    println("  Synthesis benchmark:")
    coeffs_bench = randn(get_nlm(cfg))
    spatial_bench = allocate_spatial(cfg)
    
    bench_syn = @benchmark sh_to_spat!($cfg, $coeffs_bench, $spatial_bench)
    println("    Time: $(minimum(bench_syn.times)/1e6) ms")
    println("    Allocs: $(bench_syn.allocs)")
    
    # Benchmark analysis
    println("  Analysis benchmark:")
    spatial_bench_data = randn(get_nlat(cfg), get_nphi(cfg))
    coeffs_bench_out = allocate_spectral(cfg)
    
    bench_ana = @benchmark spat_to_sh!($cfg, $spatial_bench_data, $coeffs_bench_out)
    println("    Time: $(minimum(bench_ana.times)/1e6) ms") 
    println("    Allocs: $(bench_ana.allocs)")
    
    # Test optimized nlm_calc function
    println("\n3. nlm_calc optimization test:")
    lmax_test = 50
    nlm_old = 0
    for l in 0:lmax_test
        nlm_old += l + 1
    end
    nlm_new = nlm_calc(lmax_test, lmax_test, 1)
    
    if nlm_old == nlm_new
        println("  ✓ nlm_calc optimization correct: $nlm_new coefficients")
    else
        @warn "nlm_calc optimization failed: expected $nlm_old, got $nlm_new"
    end
    
    destroy_config(cfg)
    println("\n✓ All optimization tests completed")
end

test_optimizations()