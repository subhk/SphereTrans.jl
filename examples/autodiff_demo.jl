#!/usr/bin/env julia

"""
Automatic Differentiation Demo for SHTnsKit.jl

This example demonstrates how to use ForwardDiff.jl and Zygote.jl with
SHTnsKit.jl for automatic differentiation through spherical harmonic transforms.

Applications include:
- Inverse problems (parameter estimation)
- Optimization on the sphere
- Neural differential equations with spherical geometry
- Gradient-based optimization for climate/weather models
"""

using SHTnsKit
using LinearAlgebra
using Printf

println("SHTnsKit.jl Automatic Differentiation Demo")
println("=" ^ 60)

# Test basic functionality first (even without AD packages)
function basic_functionality_demo()
    println("\n Basic Functionality Test")
    
    cfg = create_gauss_config(16, 16)
    
    # Test helper functions
    println("Testing AD helper functions...")
    @assert get_lm_from_index(cfg, 1) == (0, 0)
    @assert get_index_from_lm(cfg, 0, 0) == 1
    @assert get_lm_from_index(cfg, 4) == (1, 1)
    @assert get_index_from_lm(cfg, 1, 1) == 4
    println("✓ Helper functions working correctly")
    
    free_config(cfg)
end

basic_functionality_demo()

# ForwardDiff demo
function forwarddiff_demo()
    println("\n ForwardDiff.jl Integration Demo")
    
    try
        using ForwardDiff
        
        cfg = create_gauss_config(16, 16)
        θ, φ = get_coordinates(cfg)
        
        # Example 1: Parameter estimation for a known field
        println("Example 1: Parameter estimation")
        
        # True parameters for a simple field: amplitude * Y_2^1
        true_amplitude = 2.5
        true_frequency = 1.0
        
        # Generate "observed" field
        function generate_field(amplitude, frequency)
            sh = zeros(get_nlm(cfg))
            idx = get_index_from_lm(cfg, 2, 1)  # Y_2^1 mode
            sh[idx] = amplitude * frequency
            return synthesize(cfg, sh)
        end
        
        observed_field = generate_field(true_amplitude, true_frequency)
        
        # Define loss function for parameter estimation
        function loss_function(params)
            amplitude, frequency = params
            predicted_field = generate_field(amplitude, frequency)
            return sum((observed_field - predicted_field).^2)
        end
        
        # Initial guess
        initial_params = [1.0, 0.5]
        
        # Compute gradient using ForwardDiff
        grad = ForwardDiff.gradient(loss_function, initial_params)
        println(@sprintf("   Initial params: [%.3f, %.3f]", initial_params...))
        println(@sprintf("   True params:    [%.3f, %.3f]", true_amplitude, true_frequency))
        println(@sprintf("   Gradient:       [%.3f, %.3f]", grad...))
        
        # Simple gradient descent step
        learning_rate = 0.01
        updated_params = initial_params - learning_rate * grad
        println(@sprintf("   Updated params: [%.3f, %.3f]", updated_params...))
        
        # Example 2: Optimization on the sphere
        println("\nExample 2: Spherical optimization")
        
        # Find spherical harmonic coefficients that minimize a functional
        function spherical_functional(sh)
            spatial = synthesize(cfg, sh)
            
            # Example functional: minimize departure from target pattern
            target = @. cos(2θ) * sin(3φ)
            mismatch = sum((spatial - target).^2)
            
            # Add regularization term
            regularization = 0.01 * sum(sh.^2)
            
            return mismatch + regularization
        end
        
        # Random initial coefficients
        sh0 = 0.1 * randn(get_nlm(cfg))
        
        # Compute gradient
        grad_sh = ForwardDiff.gradient(spherical_functional, sh0)
        
        println(@sprintf("   Initial functional value: %.6f", spherical_functional(sh0)))
        println(@sprintf("   Gradient norm: %.6f", norm(grad_sh)))
        
        # Gradient descent step
        sh1 = sh0 - 0.1 * grad_sh
        new_value = spherical_functional(sh1)
        println(@sprintf("   After gradient step: %.6f", new_value))
        
        if new_value < spherical_functional(sh0)
            println("   ✓ Gradient descent step successful!")
        end
        
        # Example 3: Vector field optimization
        println("\nExample 3: Vector field optimization")
        
        function vector_functional(params)
            n = length(params) ÷ 2
            S_lm = params[1:n]
            T_lm = params[n+1:end]
            
            # Synthesize vector field
            Vt, Vp = synthesize_vector(cfg, S_lm, T_lm)
            
            # Minimize kinetic energy while matching a target pattern
            kinetic_energy = sum(Vt.^2 + Vp.^2)
            
            # Target: simple rotation pattern
            target_Vt = @. -sin(φ) * sin(θ)
            target_Vp = @. cos(φ) * ones(size(φ))
            
            mismatch = sum((Vt - target_Vt).^2 + (Vp - target_Vp).^2)
            
            return 0.1 * kinetic_energy + mismatch
        end
        
        nlm = get_nlm(cfg)
        vector_params = 0.1 * randn(2 * nlm)
        
        grad_vector = ForwardDiff.gradient(vector_functional, vector_params)
        println(@sprintf("   Vector functional value: %.6f", vector_functional(vector_params)))
        println(@sprintf("   Vector gradient norm: %.6f", norm(grad_vector)))
        
        free_config(cfg)
        println("✓ ForwardDiff integration successful!")
        
    catch LoadError
        println("⚠ ForwardDiff.jl not available - install with: using Pkg; Pkg.add(\"ForwardDiff\")")
    catch e
        println("❌ ForwardDiff demo failed: $e")
    end
end

forwarddiff_demo()

# Zygote demo
function zygote_demo()
    println("\n Zygote.jl Integration Demo")
    
    try
        using Zygote
        
        cfg = create_gauss_config(16, 16)
        
        # Example 1: Reverse-mode AD for analysis transform
        println("Example 1: Reverse-mode AD through analysis")
        
        function analysis_functional(spatial)
            sh = analyze(cfg, spatial)
            # Focus on low-degree modes (smooth component)
            low_degree_energy = sum(sh[1:10].^2)
            return low_degree_energy
        end
        
        # Generate test spatial field
        spatial_test = randn(get_nlat(cfg), get_nphi(cfg))
        
        # Compute gradient using Zygote
        grad_spatial = Zygote.gradient(analysis_functional, spatial_test)[1]
        
        println(@sprintf("   Functional value: %.6f", analysis_functional(spatial_test)))
        println(@sprintf("   Gradient shape: %s", size(grad_spatial)))
        println(@sprintf("   Gradient norm: %.6f", norm(grad_spatial)))
        
        # Example 2: Complex field gradients
        println("\nExample 2: Complex field optimization")
        
        function complex_functional(sh_real, sh_imag)
            sh_complex = complex.(sh_real, sh_imag)
            spatial_complex = synthesize_complex(cfg, sh_complex)
            
            # Minimize magnitude while preserving phase structure
            magnitude = abs.(spatial_complex)
            phase_smoothness = sum(abs2, diff(angle.(spatial_complex), dims=1))
            
            return sum(magnitude.^2) + 0.1 * phase_smoothness
        end
        
        nlm = get_nlm(cfg)
        sh_real = 0.1 * randn(nlm)
        sh_imag = 0.1 * randn(nlm)
        
        grad_real, grad_imag = Zygote.gradient(complex_functional, sh_real, sh_imag)
        
        println(@sprintf("   Complex functional value: %.6f", complex_functional(sh_real, sh_imag)))
        println(@sprintf("   Real part gradient norm: %.6f", norm(grad_real)))
        println(@sprintf("   Imag part gradient norm: %.6f", norm(grad_imag)))
        
        # Example 3: Nested transforms
        println("\nExample 3: Nested transform gradients")
        
        function nested_functional(sh)
            # Transform to spatial, modify, transform back
            spatial = synthesize(cfg, sh)
            modified_spatial = spatial.^2  # Nonlinear operation
            sh_modified = analyze(cfg, modified_spatial)
            
            # Minimize high-frequency content
            return sum(sh_modified[end-10:end].^2)
        end
        
        sh_test = 0.1 * randn(get_nlm(cfg))
        grad_nested = Zygote.gradient(nested_functional, sh_test)[1]
        
        println(@sprintf("   Nested functional value: %.6f", nested_functional(sh_test)))
        println(@sprintf("   Nested gradient norm: %.6f", norm(grad_nested)))
        
        free_config(cfg)
        println("✓ Zygote integration successful!")
        
    catch LoadError
        println("⚠ Zygote.jl not available - install with: using Pkg; Pkg.add(\"Zygote\")")
    catch e
        println("❌ Zygote demo failed: $e")
    end
end

zygote_demo()

# Comparison demo
function comparison_demo()
    println("\n  ForwardDiff vs Zygote Comparison")
    
    try
        using ForwardDiff, Zygote, BenchmarkTools
        
        cfg = create_gauss_config(12, 12)
        
        # Simple test function for comparison
        function test_function(sh)
            spatial = synthesize(cfg, sh)
            return sum(spatial.^2)
        end
        
        sh_test = randn(get_nlm(cfg))
        
        # Compute gradients with both methods
        grad_forward = ForwardDiff.gradient(test_function, sh_test)
        grad_reverse = Zygote.gradient(test_function, sh_test)[1]
        
        # Compare accuracy
        difference = norm(grad_forward - grad_reverse)
        println(@sprintf("   Gradient difference norm: %.2e", difference))
        
        if difference < 1e-10
            println(" Both methods agree to machine precision!")
        else
            println("  Methods disagree - check implementation")
        end
        
        # Performance comparison (if BenchmarkTools available)
        try
            println("\n   Performance comparison:")
            
            forward_time = @belapsed ForwardDiff.gradient($test_function, $sh_test)
            reverse_time = @belapsed Zygote.gradient($test_function, $sh_test)
            
            println(@sprintf("   ForwardDiff time: %.2f ms", forward_time * 1000))
            println(@sprintf("   Zygote time:      %.2f ms", reverse_time * 1000))
            
            if forward_time < reverse_time
                println(@sprintf("   ForwardDiff is %.1fx faster", reverse_time / forward_time))
            else
                println(@sprintf("   Zygote is %.1fx faster", forward_time / reverse_time))
            end
            
        catch
            println("   (BenchmarkTools not available for timing)")
        end
        
        free_config(cfg)
        
    catch LoadError
        println("⚠ Comparison requires both ForwardDiff and Zygote")
    catch e
        println("❌ Comparison demo failed: $e")
    end
end

comparison_demo()

# Application example: Simple inverse problem
function inverse_problem_demo()
    println("\n Application: Simple Inverse Problem")
    
    try
        using ForwardDiff
        
        cfg = create_gauss_config(20, 20)
        θ, φ = get_coordinates(cfg)
        
        println("Scenario: Estimate source parameters from observed field")
        
        # True source parameters (unknown in real problem)
        true_center_θ = π/3
        true_center_φ = π/2
        true_strength = 5.0
        true_width = 0.3
        
        # Generate synthetic observations
        function generate_source_field(center_θ, center_φ, strength, width)
            # Gaussian-like source on the sphere
            distances = @. acos(sin(center_θ) * sin(θ) * cos(φ - center_φ) + 
                               cos(center_θ) * cos(θ))
            source_spatial = @. strength * exp(-distances^2 / width^2)
            
            # Transform to spectral domain and back for SHT filtering
            sh = analyze(cfg, source_spatial)
            return synthesize(cfg, sh)
        end
        
        observations = generate_source_field(true_center_θ, true_center_φ, 
                                           true_strength, true_width)
        
        # Define inverse problem objective
        function objective(params)
            center_θ, center_φ, strength, width = params
            
            # Ensure valid parameter ranges
            center_θ = clamp(center_θ, 0.1, π - 0.1)
            center_φ = clamp(center_φ, 0.1, 2π - 0.1) 
            strength = max(strength, 0.1)
            width = clamp(width, 0.1, 1.0)
            
            predicted = generate_source_field(center_θ, center_φ, strength, width)
            mismatch = sum((observations - predicted).^2)
            
            # Add regularization
            regularization = 0.01 * (strength^2 + (width - 0.5)^2)
            
            return mismatch + regularization
        end
        
        # Initial guess
        initial_guess = [π/2, π, 3.0, 0.5]
        
        println(@sprintf("True parameters:    [%.3f, %.3f, %.3f, %.3f]", 
                        true_center_θ, true_center_φ, true_strength, true_width))
        println(@sprintf("Initial guess:      [%.3f, %.3f, %.3f, %.3f]", initial_guess...))
        println(@sprintf("Initial objective:  %.6f", objective(initial_guess)))
        
        # Simple gradient descent
        params = copy(initial_guess)
        learning_rate = 0.01
        
        println("\nGradient descent iterations:")
        for iter in 1:10
            grad = ForwardDiff.gradient(objective, params)
            params -= learning_rate * grad
            
            obj_val = objective(params)
            grad_norm = norm(grad)
            
            println(@sprintf("  Iter %2d: obj=%.6f, |grad|=%.4f, params=[%.3f,%.3f,%.3f,%.3f]",
                           iter, obj_val, grad_norm, params...))
            
            # Stop if converged
            if grad_norm < 1e-4
                break
            end
        end
        
        # Final comparison
        println(@sprintf("\nFinal parameters:   [%.3f, %.3f, %.3f, %.3f]", params...))
        println(@sprintf("True parameters:    [%.3f, %.3f, %.3f, %.3f]", 
                        true_center_θ, true_center_φ, true_strength, true_width))
        
        parameter_error = norm(params - [true_center_θ, true_center_φ, true_strength, true_width])
        println(@sprintf("Parameter error:    %.6f", parameter_error))
        
        if parameter_error < 0.2
            println("✓ Parameter estimation successful!")
        else
            println("⚠ Parameter estimation needs more iterations or tuning")
        end
        
        free_config(cfg)
        
    catch LoadError
        println("⚠ Inverse problem demo requires ForwardDiff")
    catch e
        println("❌ Inverse problem demo failed: $e")
    end
end

inverse_problem_demo()

println("\n" * "=" ^ 60)
println(" Automatic Differentiation Demo Complete!")
println()
println("Key takeaways:")
println("• SHTnsKit.jl supports both ForwardDiff.jl and Zygote.jl")
println("• AD works through all transform types (scalar, vector, complex)")
println("• Useful for inverse problems and optimization on the sphere")
println("• Both forward-mode and reverse-mode AD are efficient")
println()
println("Next steps:")
println("• Try with your own spherical harmonic applications")
println("• Combine with optimization packages (Optim.jl, Flux.jl)")
println("• Use for neural differential equations with spherical geometry")