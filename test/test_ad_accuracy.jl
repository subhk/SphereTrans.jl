"""
Comprehensive accuracy tests for automatic differentiation in SHTnsKit.jl

Tests gradient accuracy against finite differences and analytical solutions
to ensure the AD implementations are mathematically correct.
"""

using Test
using SHTnsKit
using LinearAlgebra

# Try to load AD packages - skip tests if not available
try
    using ForwardDiff, Zygote
    AD_AVAILABLE = true
catch
    AD_AVAILABLE = false
end

@testset "AD Accuracy Tests" begin

if AD_AVAILABLE

@testset "Finite Difference Validation" begin
    cfg = create_gauss_config(4, 4)
    nlm = get_nlm(cfg)
    
    # Test function: synthesis followed by norm
    function test_function(sh_coeffs)
        spatial = synthesize(cfg, sh_coeffs)
        return sum(abs2, spatial)  # ||spatial||²
    end
    
    # Random test point
    sh_coeffs = 0.1 * randn(nlm)
    
    @testset "ForwardDiff vs Finite Differences" begin
        # ForwardDiff gradient
        grad_fd = ForwardDiff.gradient(test_function, sh_coeffs)
        
        # Central difference approximation
        h = 1e-8
        grad_numeric = zeros(nlm)
        
        for i in 1:nlm
            sh_plus = copy(sh_coeffs)
            sh_minus = copy(sh_coeffs)
            sh_plus[i] += h
            sh_minus[i] -= h
            
            grad_numeric[i] = (test_function(sh_plus) - test_function(sh_minus)) / (2h)
        end
        
        # Check agreement
        relative_error = norm(grad_fd - grad_numeric) / max(norm(grad_fd), norm(grad_numeric))
        @test relative_error < 1e-6
        
        println("ForwardDiff vs Finite Diff relative error: $relative_error")
    end
    
    @testset "Zygote vs Finite Differences" begin
        # Zygote gradient
        value, grad_zy = Zygote.withgradient(test_function, sh_coeffs)
        grad_zy = grad_zy[1]
        
        # Central difference approximation
        h = 1e-8
        grad_numeric = zeros(nlm)
        
        for i in 1:nlm
            sh_plus = copy(sh_coeffs)
            sh_minus = copy(sh_coeffs)
            sh_plus[i] += h
            sh_minus[i] -= h
            
            grad_numeric[i] = (test_function(sh_plus) - test_function(sh_minus)) / (2h)
        end
        
        # Check agreement
        relative_error = norm(grad_zy - grad_numeric) / max(norm(grad_zy), norm(grad_numeric))
        @test relative_error < 1e-6
        
        println("Zygote vs Finite Diff relative error: $relative_error")
    end
    
    @testset "ForwardDiff vs Zygote Consistency" begin
        grad_fd = ForwardDiff.gradient(test_function, sh_coeffs)
        value, grad_zy = Zygote.withgradient(test_function, sh_coeffs)
        grad_zy = grad_zy[1]
        
        relative_error = norm(grad_fd - grad_zy) / max(norm(grad_fd), norm(grad_zy))
        @test relative_error < 1e-12
        
        println("ForwardDiff vs Zygote relative error: $relative_error")
    end
end

@testset "Power Spectrum Derivative Accuracy" begin
    cfg = create_gauss_config(6, 6)
    sh_coeffs = 0.1 * randn(get_nlm(cfg))
    
    function power_sum(sh)
        power = power_spectrum(cfg, sh)
        return sum(power)
    end
    
    @testset "Power Spectrum - Finite Difference Check" begin
        # Analytical: ∂(∑P_l)/∂c_lm = ∂(∑|c_lm|²)/∂c_lm = 2*c_lm
        grad_analytical = 2 * sh_coeffs
        
        # ForwardDiff
        grad_fd = ForwardDiff.gradient(power_sum, sh_coeffs)
        
        # Check agreement with analytical gradient
        relative_error = norm(grad_fd - grad_analytical) / norm(grad_analytical)
        @test relative_error < 1e-10
        
        println("Power spectrum ForwardDiff error: $relative_error")
        
        # Zygote
        value, grad_zy = Zygote.withgradient(power_sum, sh_coeffs)
        grad_zy = grad_zy[1]
        
        relative_error_zy = norm(grad_zy - grad_analytical) / norm(grad_analytical)
        @test relative_error_zy < 1e-10
        
        println("Power spectrum Zygote error: $relative_error_zy")
    end
end

@testset "Point Evaluation Gradient Accuracy" begin
    cfg = create_gauss_config(4, 4)
    θ, φ = π/3, π/4
    
    function point_eval(sh)
        return evaluate_at_point(cfg, sh, θ, φ)
    end
    
    sh_coeffs = 0.1 * randn(get_nlm(cfg))
    
    @testset "Point Evaluation - Finite Difference" begin
        # ForwardDiff gradient
        grad_fd = ForwardDiff.gradient(point_eval, sh_coeffs)
        
        # Finite difference
        h = 1e-8
        grad_numeric = zeros(length(sh_coeffs))
        
        for i in 1:length(sh_coeffs)
            sh_plus = copy(sh_coeffs)
            sh_minus = copy(sh_coeffs)
            sh_plus[i] += h
            sh_minus[i] -= h
            
            grad_numeric[i] = (point_eval(sh_plus) - point_eval(sh_minus)) / (2h)
        end
        
        relative_error = norm(grad_fd - grad_numeric) / max(norm(grad_fd), norm(grad_numeric), 1e-15)
        @test relative_error < 1e-5  # Slightly relaxed for point evaluation
        
        println("Point evaluation ForwardDiff error: $relative_error")
        
        # Zygote
        value, grad_zy = Zygote.withgradient(point_eval, sh_coeffs)
        grad_zy = grad_zy[1]
        
        relative_error_zy = norm(grad_zy - grad_numeric) / max(norm(grad_zy), norm(grad_numeric), 1e-15)
        @test relative_error_zy < 1e-5
        
        println("Point evaluation Zygote error: $relative_error_zy")
    end
end

@testset "Round-trip Transform Gradient" begin
    cfg = create_gauss_config(4, 4)
    
    function roundtrip_error(sh)
        spatial = synthesize(cfg, sh)
        sh_recovered = analyze(cfg, spatial)
        return sum(abs2, sh - sh_recovered)  # Should be ~machine precision
    end
    
    sh_coeffs = 0.1 * randn(get_nlm(cfg))
    
    # The roundtrip error should be very small
    error_val = roundtrip_error(sh_coeffs)
    @test error_val < 1e-20
    
    @testset "Round-trip Gradient Accuracy" begin
        # ForwardDiff gradient  
        grad_fd = ForwardDiff.gradient(roundtrip_error, sh_coeffs)
        
        # Zygote gradient
        value, grad_zy = Zygote.withgradient(roundtrip_error, sh_coeffs)
        grad_zy = grad_zy[1]
        
        # Since the roundtrip error is essentially zero, the gradient should also be very small
        @test norm(grad_fd) < 1e-12
        @test norm(grad_zy) < 1e-12
        
        # And they should agree with each other
        relative_error = norm(grad_fd - grad_zy) / max(norm(grad_fd), norm(grad_zy), 1e-15)
        @test relative_error < 1e-10
        
        println("Round-trip gradient norms - FD: $(norm(grad_fd)), Zygote: $(norm(grad_zy))")
    end
end

@testset "Vector Field Gradient Accuracy" begin
    cfg = create_gauss_config(4, 4)
    nlm = get_nlm(cfg)
    
    function vector_energy(sph_coeffs, tor_coeffs)
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        return 0.5 * (sum(abs2, u_theta) + sum(abs2, u_phi))
    end
    
    sph_coeffs = 0.1 * randn(nlm)
    tor_coeffs = 0.1 * randn(nlm)
    
    @testset "Vector Energy Gradients" begin
        # Separate functions for each component
        energy_sph(s) = vector_energy(s, tor_coeffs)
        energy_tor(t) = vector_energy(sph_coeffs, t)
        
        # ForwardDiff
        grad_sph_fd = ForwardDiff.gradient(energy_sph, sph_coeffs)
        grad_tor_fd = ForwardDiff.gradient(energy_tor, tor_coeffs)
        
        # Zygote
        value, grads_zy = Zygote.withgradient(vector_energy, sph_coeffs, tor_coeffs)
        grad_sph_zy = grads_zy[1]
        grad_tor_zy = grads_zy[2]
        
        # Check consistency
        rel_err_sph = norm(grad_sph_fd - grad_sph_zy) / max(norm(grad_sph_fd), norm(grad_sph_zy))
        rel_err_tor = norm(grad_tor_fd - grad_tor_zy) / max(norm(grad_tor_fd), norm(grad_tor_zy))
        
        @test rel_err_sph < 1e-12
        @test rel_err_tor < 1e-12
        
        println("Vector energy gradient errors - Sph: $rel_err_sph, Tor: $rel_err_tor")
    end
end

@testset "Spatial Integration Gradient" begin
    cfg = create_gauss_config(4, 4)
    
    # Create a test spatial field
    spatial_data = 0.1 * randn(get_nlat(cfg), get_nphi(cfg))
    
    function integration_test(spatial)
        return spatial_integral(cfg, spatial)^2  # Square for more interesting gradient
    end
    
    @testset "Spatial Integration Accuracy" begin
        # Zygote gradient
        value, grad_zy = Zygote.withgradient(integration_test, spatial_data)
        grad_zy = grad_zy[1]
        
        # Finite difference check
        h = 1e-8
        grad_numeric = zeros(size(spatial_data))
        
        for i in eachindex(spatial_data)
            spatial_plus = copy(spatial_data)
            spatial_minus = copy(spatial_data)
            spatial_plus[i] += h
            spatial_minus[i] -= h
            
            grad_numeric[i] = (integration_test(spatial_plus) - integration_test(spatial_minus)) / (2h)
        end
        
        relative_error = norm(grad_zy - grad_numeric) / max(norm(grad_zy), norm(grad_numeric))
        @test relative_error < 1e-6
        
        println("Spatial integration gradient error: $relative_error")
    end
end

else
    @test_skip "Automatic differentiation packages not available"
end

end # main testset

"""
Helper function to run accuracy benchmark
"""
function benchmark_ad_accuracy(lmax=8)
    if !AD_AVAILABLE
        println("AD packages not available for benchmarking")
        return
    end
    
    cfg = create_gauss_config(lmax, lmax)
    nlm = get_nlm(cfg)
    sh_coeffs = randn(nlm)
    
    println("\\n=== AD Accuracy Benchmark (lmax=$lmax) ===")
    
    # Test function
    function test_func(sh)
        spatial = synthesize(cfg, sh)
        power = sum(abs2, spatial)
        return power + 0.1 * sum(abs2, analyze(cfg, spatial))
    end
    
    # ForwardDiff
    println("ForwardDiff gradient...")
    @time grad_fd = ForwardDiff.gradient(test_func, sh_coeffs)
    
    # Zygote  
    println("Zygote gradient...")
    @time value, grad_zy = Zygote.withgradient(test_func, sh_coeffs)
    
    # Accuracy
    rel_error = norm(grad_fd - grad_zy[1]) / norm(grad_fd)
    println("Relative error between methods: $rel_error")
    
    # Finite difference spot check (first 5 components)
    println("Finite difference validation (first 5 components)...")
    h = 1e-8
    for i in 1:min(5, nlm)
        sh_plus = copy(sh_coeffs)
        sh_minus = copy(sh_coeffs)
        sh_plus[i] += h
        sh_minus[i] -= h
        
        fd_grad = (test_func(sh_plus) - test_func(sh_minus)) / (2h)
        
        println("Component $i: FD=$(fd_grad), ForwardDiff=$(grad_fd[i]), Zygote=$(grad_zy[1][i])")
    end
    
    println("=== Benchmark Complete ===\\n")
end

# Export for external use
export benchmark_ad_accuracy