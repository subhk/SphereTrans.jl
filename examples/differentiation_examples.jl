"""
SHTnsKit.jl Automatic Differentiation Examples

This script demonstrates how to use ForwardDiff.jl and Zygote.jl with SHTnsKit
for forward and backward automatic differentiation of spherical harmonic transforms.

Run with: julia --project=. examples/differentiation_examples.jl
"""

using SHTnsKit
using ForwardDiff
using Zygote
using LinearAlgebra
using Printf

# Helper function to display results nicely
function display_results(name, value, gradient)
    println("="^60)
    println(name)
    println("="^60)
    println("Function value: ", value)
    println("Gradient norm: ", norm(gradient))
    println("Gradient sample (first 5 elements): ", gradient[1:min(5, length(gradient))])
    println()
end

# Example 1: Forward-mode differentiation with ForwardDiff
function example_forwarddiff()
    println("\n ForwardDiff Examples (Forward-mode AD)")
    println("="^60)
    
    # Create SHT configuration
    cfg = create_gauss_config(8, 8)
    nlm = get_nlm(cfg)
    
    # Example 1a: Gradient of total power w.r.t. spectral coefficients
    function total_power_loss(sh_coeffs)
        spatial_field = synthesize(cfg, sh_coeffs)
        return sum(abs2, spatial_field)  # Total power in spatial domain
    end
    
    # Random spectral coefficients
    sh_coeffs = rand(nlm)
    
    # Compute gradient using ForwardDiff
    value = total_power_loss(sh_coeffs)
    grad_fd = ForwardDiff.gradient(total_power_loss, sh_coeffs)
    
    display_results("ForwardDiff: Total Power Gradient", value, grad_fd)
    
    # Example 1b: Gradient of point evaluation
    θ, φ = π/4, π/2  # Evaluation point
    function point_eval_loss(sh_coeffs)
        return evaluate_at_point(cfg, sh_coeffs, θ, φ)^2
    end
    
    value = point_eval_loss(sh_coeffs)
    grad_point = ForwardDiff.gradient(point_eval_loss, sh_coeffs)
    
    display_results("ForwardDiff: Point Evaluation Gradient", value, grad_point)
    
    # Example 1c: Gradient of power spectrum regularization
    function power_spectrum_regularization(sh_coeffs)
        power = power_spectrum(cfg, sh_coeffs)
        # L2 regularization with higher penalty for higher degrees
        weights = [(l+1)^2 for l in 0:get_lmax(cfg)]
        return sum(weights .* power)
    end
    
    value = power_spectrum_regularization(sh_coeffs)
    grad_reg = ForwardDiff.gradient(power_spectrum_regularization, sh_coeffs)
    
    display_results("ForwardDiff: Power Spectrum Regularization", value, grad_reg)
    
    return cfg, sh_coeffs
end

# Example 2: Reverse-mode differentiation with Zygote
function example_zygote(cfg, sh_coeffs)
    println("\n Zygote Examples (Reverse-mode AD)")
    println("="^60)
    
    # Example 2a: Same total power function using Zygote
    function total_power_loss(sh_coeffs)
        spatial_field = synthesize(cfg, sh_coeffs)
        return sum(abs2, spatial_field)
    end
    
    value, grad_zy = Zygote.withgradient(total_power_loss, sh_coeffs)
    
    display_results("Zygote: Total Power Gradient", value[1], grad_zy[1])
    
    # Example 2b: Gradient of analysis operation
    spatial_data = rand(get_nlat(cfg), get_nphi(cfg))
    function analysis_loss(spatial_data)
        sh_result = analyze(cfg, spatial_data)
        return sum(abs2, sh_result[1:10])  # Only first 10 coefficients
    end
    
    value, grad_analysis = Zygote.withgradient(analysis_loss, spatial_data)
    
    display_results("Zygote: Analysis Operation Gradient", value[1], vec(grad_analysis[1]))
    
    # Example 2c: Round-trip loss (synthesis → analysis)
    function roundtrip_loss(sh_coeffs)
        spatial = synthesize(cfg, sh_coeffs)
        sh_recovered = analyze(cfg, spatial)
        return sum(abs2, sh_coeffs - sh_recovered)  # Reconstruction error
    end
    
    value, grad_roundtrip = Zygote.withgradient(roundtrip_loss, sh_coeffs)
    
    display_results("Zygote: Round-trip Loss Gradient", value[1], grad_roundtrip[1])
end

# Example 3: Vector field differentiation
function example_vector_differentiation()
    println("\n Vector Field Differentiation")
    println("="^60)
    
    cfg = create_gauss_config(6, 6)
    nlm = get_nlm(cfg)
    
    # Example: Gradient of kinetic energy in vector fields
    function kinetic_energy(sph_coeffs, tor_coeffs)
        u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
        return 0.5 * (sum(abs2, u_theta) + sum(abs2, u_phi))
    end
    
    sph_coeffs = rand(nlm)
    tor_coeffs = rand(nlm)
    
    # Forward-mode differentiation
    function ke_wrt_sph(s)
        return kinetic_energy(s, tor_coeffs)
    end
    function ke_wrt_tor(t)
        return kinetic_energy(sph_coeffs, t)
    end
    
    value = kinetic_energy(sph_coeffs, tor_coeffs)
    grad_sph_fd = ForwardDiff.gradient(ke_wrt_sph, sph_coeffs)
    grad_tor_fd = ForwardDiff.gradient(ke_wrt_tor, tor_coeffs)
    
    display_results("ForwardDiff: Kinetic Energy w.r.t. Spheroidal", value, grad_sph_fd)
    display_results("ForwardDiff: Kinetic Energy w.r.t. Toroidal", value, grad_tor_fd)
    
    # Reverse-mode differentiation
    val_zy, grads_zy = Zygote.withgradient(kinetic_energy, sph_coeffs, tor_coeffs)
    
    display_results("Zygote: Kinetic Energy w.r.t. Spheroidal", val_zy[1], grads_zy[1])
    display_results("Zygote: Kinetic Energy w.r.t. Toroidal", val_zy[1], grads_zy[2])
end

# Example 4: Comparison between ForwardDiff and Zygote
function example_comparison()
    println("\n ForwardDiff vs Zygote Comparison")
    println("="^60)
    
    cfg = create_gauss_config(6, 6)
    sh_coeffs = rand(get_nlm(cfg))
    
    function test_function(sh)
        spatial = synthesize(cfg, sh)
        return sum(spatial .* spatial)  # L2 norm squared
    end
    
    # ForwardDiff
    @time begin
        value_fd = test_function(sh_coeffs)
        grad_fd = ForwardDiff.gradient(test_function, sh_coeffs)
    end
    
    # Zygote  
    @time begin
        value_zy, grad_zy = Zygote.withgradient(test_function, sh_coeffs)
    end
    
    println("ForwardDiff value: ", value_fd)
    println("Zygote value: ", value_zy[1])
    println("Values match: ", isapprox(value_fd, value_zy[1]))
    
    println("Gradient difference (norm): ", norm(grad_fd - grad_zy[1]))
    println("Gradients match: ", isapprox(grad_fd, grad_zy[1], rtol=1e-10))
end

# Example 5: Machine Learning-style optimization
function example_optimization()
    println("\n Optimization Example")
    println("="^60)
    
    cfg = create_gauss_config(4, 4)
    nlm = get_nlm(cfg)
    
    # Create target spatial field (e.g., a Gaussian)
    theta_mat, phi_mat = create_coordinate_matrices(cfg)
    x, y, z = create_cartesian_coordinates(cfg)
    
    # Target: Gaussian centered at north pole
    target_field = exp.(-((z .- 1).^2) ./ 0.1)
    
    # Loss function: mean squared error between synthesis and target
    function loss_function(sh_coeffs)
        spatial_pred = synthesize(cfg, sh_coeffs)
        return sum((spatial_pred - target_field).^2) / length(target_field)
    end
    
    # Initialize random coefficients
    sh_coeffs = 0.1 * randn(nlm)
    
    # Optimization using gradient descent with Zygote
    learning_rate = 0.01
    n_iterations = 100
    
    println("Starting optimization...")
    println("Initial loss: ", loss_function(sh_coeffs))
    
    for i in 1:n_iterations
        loss_val, grad = Zygote.withgradient(loss_function, sh_coeffs)
        
        # Gradient descent step
        sh_coeffs .-= learning_rate .* grad[1]
        
        if i % 20 == 0
            println("Iteration $i: loss = ", loss_val[1])
        end
    end
    
    final_loss = loss_function(sh_coeffs)
    println("Final loss: ", final_loss)
    
    # Show improvement
    println("Loss reduction: ", (loss_function(0.1 * randn(nlm)) - final_loss))
end

# Main execution
function main()
    println(" SHTnsKit.jl Automatic Differentiation Examples")
    println("="^60)
    println("This demonstrates forward-mode (ForwardDiff) and reverse-mode (Zygote)")
    println("automatic differentiation with spherical harmonic transforms.")
    println()
    
    try
        # Run examples
        cfg, sh_coeffs = example_forwarddiff()
        example_zygote(cfg, sh_coeffs)
        example_vector_differentiation()
        example_comparison()
        example_optimization()
        
        println("\n All examples completed successfully!")
        println("="^60)
        println("Key takeaways:")
        println("• ForwardDiff is excellent for functions with few inputs, many outputs")
        println("• Zygote is excellent for functions with many inputs, few outputs (ML/optimization)")
        println("• Both support all SHTnsKit transform operations")
        println("• Vector transforms work seamlessly with both AD systems")
        println("• Round-trip transforms preserve gradients correctly")
        
    catch e
        println(" Error running examples: ", e)
        println("Make sure ForwardDiff and Zygote are installed and SHTnsKit is properly loaded.")
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end