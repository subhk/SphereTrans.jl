"""
Direct test of AD functionality without precompilation
"""

println("Testing AD fixes directly from source...")

# Load source directly
try
    # Add current directory to path for loading local modules
    push!(LOAD_PATH, ".")
    push!(LOAD_PATH, "./src")
    
    # Load ForwardDiff and Zygote first
    println("Loading AD packages...")
    using ForwardDiff, Zygote
    println(" AD packages loaded")
    
    # Now load SHTnsKit source
    println("Loading SHTnsKit from source...")
    include("src/SHTnsKit.jl")
    using .SHTnsKit
    println(" SHTnsKit loaded from source")
    
    # Create test config
    cfg = create_gauss_config(4, 4)
    nlm = get_nlm(cfg)
    sh_coeffs = 0.1 * randn(nlm)
    
    println(" Test configuration: lmax=4, nlm=$nlm")
    
    # Test basic functionality
    spatial = synthesize(cfg, sh_coeffs)
    println(" Synthesis works: spatial size = $(size(spatial))")
    
    sh_back = analyze(cfg, spatial)
    error = norm(sh_coeffs - sh_back) / norm(sh_coeffs)
    println(" Analysis works: relative error = $error")
    
    # Test extensions are loaded
    println("\\nChecking extensions...")
    
    # Check if ForwardDiff extension methods exist
    test_func(sh) = sum(abs2, synthesize(cfg, sh))
    
    try
        grad_fd = ForwardDiff.gradient(test_func, sh_coeffs)
        println(" ForwardDiff extension working! Gradient norm: $(norm(grad_fd))")
        
        # Numerical check
        h = 1e-8
        num_grad = (test_func(sh_coeffs .+ h) - test_func(sh_coeffs .- h)) / (2h)
        error = abs(grad_fd[1] - num_grad) / max(abs(grad_fd[1]), num_grad, 1e-15)
        println("  First component accuracy: AD=$(grad_fd[1]), FD=$num_grad, error=$error")
        
    catch e
        println(" ForwardDiff extension issue: ", typeof(e), " - ", e)
    end
    
    try
        val, grad_zy = Zygote.withgradient(test_func, sh_coeffs)
        println(" Zygote extension working! Value: $(val[1]), gradient norm: $(norm(grad_zy[1]))")
        
    catch e
        println(" Zygote extension issue: ", typeof(e), " - ", e)
    end
    
    # Test power spectrum accuracy
    println("\\n--- Power Spectrum Accuracy Test ---")
    
    power = power_spectrum(cfg, sh_coeffs)
    total_power_val = sum(power)
    
    # For orthonormal spherical harmonics, total power should equal sum(coeffs^2)
    expected_power = sum(abs2, sh_coeffs)
    power_error = abs(total_power_val - expected_power) / expected_power
    
    println("Expected total power: $expected_power")
    println("Computed total power: $total_power_val") 
    println("Power calculation error: $power_error")
    
    if power_error < 1e-10
        println(" Power spectrum calculation EXACT")
    elseif power_error < 1e-6
        println(" Power spectrum calculation good")
    else
        println(" Power spectrum calculation may have issues")
    end
    
    # Test power spectrum derivative
    power_func(sh) = sum(power_spectrum(cfg, sh))
    
    try
        grad_power_fd = ForwardDiff.gradient(power_func, sh_coeffs)
        analytical_grad = 2 .* sh_coeffs
        
        rel_error = norm(grad_power_fd - analytical_grad) / norm(analytical_grad)
        
        println("\\nPower spectrum derivative test:")
        println("  Analytical gradient norm: $(norm(analytical_grad))")
        println("  ForwardDiff gradient norm: $(norm(grad_power_fd))")
        println("  Relative error: $rel_error")
        
        if rel_error < 1e-12
            println(" EXCELLENT: Power spectrum derivative fixes working perfectly!")
        elseif rel_error < 1e-6
            println(" GOOD: Power spectrum derivative accuracy acceptable")
        else
            println(" ISSUE: Power spectrum derivative has accuracy problems")
            
            # Show first few components for debugging
            println("  First 5 components comparison:")
            for i in 1:min(5, nlm)
                println("    $i: Analytical=$(analytical_grad[i]), AD=$(grad_power_fd[i]), Diff=$(grad_power_fd[i] - analytical_grad[i])")
            end
        end
        
    catch e
        println(" Power spectrum derivative test failed: $e")
    end
    
    println("\\n AD accuracy test completed!")
    
catch e
    println(" Test failed with error: $e")
    println("\\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        Base.showerror(stdout, exc, bt)
        println()
    end
end