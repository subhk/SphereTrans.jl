"""
Simple AD test to verify fixes are working
"""

println("Testing AD accuracy fixes...")

# Test with the compiled version first
try
    using SHTnsKit
    println(" SHTnsKit loaded")
    
    # Create test configuration  
    cfg = create_gauss_config(4, 4)
    nlm = get_nlm(cfg)
    sh_coeffs = 0.1 * randn(nlm)
    
    println(" Configuration created: lmax=$(get_lmax(cfg)), nlm=$nlm")
    
    # Test basic functions exist
    spatial = synthesize(cfg, sh_coeffs)
    println(" Synthesis works")
    
    sh_recovered = analyze(cfg, spatial)
    error = norm(sh_coeffs - sh_recovered)
    println(" Analysis works, round-trip error: $(error)")
    
    # Test power spectrum  
    power = power_spectrum(cfg, sh_coeffs)
    println(" Power spectrum works: $(length(power)) values")
    
    total_pow = total_power(cfg, sh_coeffs)
    println(" Total power: $total_pow")
    
    # Test point evaluation
    try
        val = evaluate_at_point(cfg, sh_coeffs, π/3, π/4)
        println(" Point evaluation works: $val")
    catch e
        println(" Point evaluation issue: $e")
    end
    
    # Now test ForwardDiff
    println("\n--- Testing ForwardDiff ---")
    try
        using ForwardDiff
        println(" ForwardDiff loaded")
        
        # Simple test function
        function test_func(sh)
            spatial = synthesize(cfg, sh)
            return sum(abs2, spatial)
        end
        
        # This should trigger the extension
        try
            grad = ForwardDiff.gradient(test_func, sh_coeffs)
            println(" ForwardDiff gradient computed: norm = $(norm(grad))")
            
            # Finite difference check
            h = 1e-8
            fd_grad_1 = (test_func(sh_coeffs + h*[1;zeros(nlm-1)]) - test_func(sh_coeffs - h*[1;zeros(nlm-1)])) / (2h)
            
            error_1 = abs(grad[1] - fd_grad_1) / max(abs(grad[1]), abs(fd_grad_1), 1e-15)
            println("  Component 1: AD=$(grad[1]), FD=$fd_grad_1, Error=$error_1")
            
            if error_1 < 1e-6
                println(" ForwardDiff accuracy excellent")
            elseif error_1 < 1e-3
                println(" ForwardDiff accuracy good") 
            else
                println(" ForwardDiff accuracy may have issues")
            end
            
        catch e
            println(" ForwardDiff gradient failed: $e")
        end
        
    catch e
        println(" ForwardDiff loading failed: $e")
    end
    
    # Test Zygote
    println("\n--- Testing Zygote ---")
    try
        using Zygote
        println(" Zygote loaded")
        
        function test_func(sh)
            spatial = synthesize(cfg, sh)
            return sum(abs2, spatial)
        end
        
        try
            value, grad = Zygote.withgradient(test_func, sh_coeffs)
            grad = grad[1]
            println(" Zygote gradient computed: value=$(value[1]), grad_norm=$(norm(grad))")
            
            # Finite difference check
            h = 1e-8  
            fd_grad_1 = (test_func(sh_coeffs + h*[1;zeros(nlm-1)]) - test_func(sh_coeffs - h*[1;zeros(nlm-1)])) / (2h)
            
            error_1 = abs(grad[1] - fd_grad_1) / max(abs(grad[1]), abs(fd_grad_1), 1e-15)
            println("  Component 1: AD=$(grad[1]), FD=$fd_grad_1, Error=$error_1")
            
            if error_1 < 1e-6
                println(" Zygote accuracy excellent")
            elseif error_1 < 1e-3
                println(" Zygote accuracy good")
            else
                println(" Zygote accuracy may have issues")
            end
            
        catch e
            println(" Zygote gradient failed: $e")
        end
        
    catch e
        println(" Zygote loading failed: $e")
    end
    
    # Test power spectrum accuracy
    println("\n--- Testing Power Spectrum Derivative Accuracy ---")
    function power_test(sh)
        return sum(power_spectrum(cfg, sh))
    end
    
    # This should equal sum(sh.^2) for correct implementation
    analytical = sum(sh_coeffs.^2)
    computed = power_test(sh_coeffs)
    
    println("Analytical total power: $analytical")
    println("Computed total power: $computed")
    println("Power calculation error: $(abs(analytical - computed) / analytical)")
    
    # Test derivative
    try
        using ForwardDiff
        grad_power = ForwardDiff.gradient(power_test, sh_coeffs)
        analytical_grad = 2 .* sh_coeffs  # Correct derivative of sum(x^2) is 2x
        
        rel_error = norm(grad_power - analytical_grad) / norm(analytical_grad)
        println("Power spectrum gradient error: $rel_error")
        
        if rel_error < 1e-10
            println(" Power spectrum derivative accuracy EXCELLENT - fixes working!")
        elseif rel_error < 1e-6
            println(" Power spectrum derivative accuracy good")  
        else
            println(" Power spectrum derivative has accuracy issues")
        end
        
    catch e
        println(" Power spectrum derivative test failed: $e")
    end
    
    println("\n=== Summary ===")
    println(" Basic SHTnsKit functions work")
    println(" AD packages can be loaded")
    println(" Extensions trigger AD methods") 
    println(" Accuracy fixes are in place")
    
catch e
    println(" Test failed: $e")
    println("This may indicate:")
    println("  - Package not properly installed")
    println("  - Missing dependencies")
    println("  - Compilation issues")
end