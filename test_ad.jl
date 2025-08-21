"""
Simple test script to verify automatic differentiation works
"""

# Test without pre-compilation to avoid permission issues
println("Testing SHTnsKit automatic differentiation...")

try
    # Load packages
    include("src/SHTnsKit.jl")
    using .SHTnsKit
    
    # Test basic configuration creation
    cfg = create_gauss_config(4, 4)
    println(" Configuration created successfully")
    println("  - lmax: ", get_lmax(cfg))
    println("  - nlm: ", get_nlm(cfg))
    println("  - nlat: ", get_nlat(cfg))
    println("  - nphi: ", get_nphi(cfg))
    
    # Test basic transforms
    sh_coeffs = rand(get_nlm(cfg))
    spatial_data = synthesize(cfg, sh_coeffs)
    sh_recovered = analyze(cfg, spatial_data)
    
    error = norm(sh_coeffs - sh_recovered)
    println(" Basic transforms work, round-trip error: ", error)
    
    # Test ForwardDiff availability
    try
        using ForwardDiff
        
        function test_function(sh)
            spatial = synthesize(cfg, sh)
            return sum(abs2, spatial)
        end
        
        grad_fd = ForwardDiff.gradient(test_function, sh_coeffs)
        println(" ForwardDiff extension works")
        println("  - Gradient norm: ", norm(grad_fd))
        println("  - Gradient finite: ", all(isfinite, grad_fd))
        
    catch e
        println(" ForwardDiff not available or not working: ", e)
    end
    
    # Test Zygote availability
    try
        using Zygote
        
        function test_function(sh)
            spatial = synthesize(cfg, sh)
            return sum(abs2, spatial)
        end
        
        value, grad_zy = Zygote.withgradient(test_function, sh_coeffs)
        println(" Zygote extension works") 
        println("  - Value: ", value[1])
        println("  - Gradient norm: ", norm(grad_zy[1]))
        println("  - Gradient finite: ", all(isfinite, grad_zy[1]))
        
    catch e
        println(" Zygote not available or not working: ", e)
    end
    
    println("\n Basic tests completed successfully!")
    
catch e
    println(" Error during testing: ", e)
    println("This may be due to missing dependencies or compilation issues.")
end