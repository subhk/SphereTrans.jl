# Automatic Differentiation with SHTnsKit.jl

SHTnsKit.jl provides comprehensive automatic differentiation support through extensions for both **ForwardDiff.jl** (forward-mode AD) and **Zygote.jl** (reverse-mode AD). This enables efficient gradient computation for optimization, machine learning, and sensitivity analysis applications involving spherical harmonic transforms.

## Overview

### Forward-mode AD (ForwardDiff.jl)
- **Best for**: Functions with few inputs and many outputs
- **Use cases**: Sensitivity analysis, parameter estimation with few parameters
- **Performance**: Scales with number of input parameters

### Reverse-mode AD (Zygote.jl) 
- **Best for**: Functions with many inputs and few outputs
- **Use cases**: Machine learning, optimization with many parameters
- **Performance**: Scales with number of output parameters

## Installation and Setup

The differentiation capabilities are automatically available when you load the respective packages:

```julia
using SHTnsKit
using ForwardDiff  # Enables ForwardDiff extension
using Zygote       # Enables Zygote extension
```

## Supported Functions

Both ForwardDiff and Zygote support differentiation of all major SHTnsKit functions:

### Core Transforms
- `synthesize(cfg, sh_coeffs)` - Spectral to spatial transform
- `analyze(cfg, spatial_data)` - Spatial to spectral transform
- `sh_to_spat!(cfg, sh_coeffs, spatial_data)` - In-place synthesis
- `spat_to_sh!(cfg, spatial_data, sh_coeffs)` - In-place analysis

### Vector Transforms
- `synthesize_vector(cfg, sph_coeffs, tor_coeffs)` - Vector synthesis
- `analyze_vector(cfg, u_theta, u_phi)` - Vector analysis

### Complex Transforms
- `synthesize_complex(cfg, sh_coeffs)` - Complex synthesis
- `analyze_complex(cfg, spatial_data)` - Complex analysis

### Analysis Functions
- `evaluate_at_point(cfg, sh_coeffs, theta, phi)` - Point evaluation
- `power_spectrum(cfg, sh_coeffs)` - Power spectrum computation
- `total_power(cfg, sh_coeffs)` - Total power
- `spatial_integral(cfg, spatial_data)` - Spatial integration
- `spatial_mean(cfg, spatial_data)` - Spatial mean

## ForwardDiff Examples

### Basic Gradient Computation

```julia
using SHTnsKit, ForwardDiff

# Create configuration
cfg = create_gauss_config(8, 8)
sh_coeffs = rand(get_nlm(cfg))

# Define objective function
function total_power_loss(sh)
    spatial = synthesize(cfg, sh)
    return sum(abs2, spatial)  # Total power
end

# Compute gradient
gradient = ForwardDiff.gradient(total_power_loss, sh_coeffs)
```

### Point Evaluation Gradients

```julia
# Gradient of field value at specific point
θ, φ = π/4, π/2
function point_eval_loss(sh)
    return evaluate_at_point(cfg, sh, θ, φ)^2
end

gradient = ForwardDiff.gradient(point_eval_loss, sh_coeffs)
```

### Power Spectrum Regularization

```julia
function power_regularization(sh)
    power = power_spectrum(cfg, sh)
    weights = [(l+1)^2 for l in 0:get_lmax(cfg)]  # Higher penalty for higher degrees
    return sum(weights .* power)
end

gradient = ForwardDiff.gradient(power_regularization, sh_coeffs)
```

### Higher-order Derivatives

```julia
# Hessian computation
hessian_matrix = ForwardDiff.hessian(total_power_loss, sh_coeffs)

# Jacobian for vector-valued functions
function multi_point_eval(sh)
    points = [(π/4, π/2), (π/3, π), (π/2, 3π/2)]
    return [evaluate_at_point(cfg, sh, θ, φ) for (θ, φ) in points]
end

jacobian_matrix = ForwardDiff.jacobian(multi_point_eval, sh_coeffs)
```

## Zygote Examples

### Basic Reverse-mode Differentiation

```julia
using SHTnsKit, Zygote

cfg = create_gauss_config(8, 8)
sh_coeffs = rand(get_nlm(cfg))

function loss_function(sh)
    spatial = synthesize(cfg, sh)
    return sum(abs2, spatial)
end

# Get both value and gradient
value, gradient = Zygote.withgradient(loss_function, sh_coeffs)
```

### Analysis Operation Gradients

```julia
spatial_data = rand(get_nlat(cfg), get_nphi(cfg))

function analysis_loss(spatial)
    sh_result = analyze(cfg, spatial)
    return sum(abs2, sh_result[1:10])  # Loss on first 10 coefficients
end

value, gradient = Zygote.withgradient(analysis_loss, spatial_data)
```

### Vector Field Differentiation

```julia
function kinetic_energy(sph_coeffs, tor_coeffs)
    u_theta, u_phi = synthesize_vector(cfg, sph_coeffs, tor_coeffs)
    return 0.5 * (sum(abs2, u_theta) + sum(abs2, u_phi))
end

sph_coeffs = rand(get_nlm(cfg))
tor_coeffs = rand(get_nlm(cfg))

# Gradient w.r.t. both coefficient types
value, gradients = Zygote.withgradient(kinetic_energy, sph_coeffs, tor_coeffs)
grad_sph = gradients[1]  # Gradient w.r.t. spheroidal coefficients
grad_tor = gradients[2]  # Gradient w.r.t. toroidal coefficients
```

## Optimization Examples

### Gradient Descent with Zygote

```julia
# Target fitting example
cfg = create_gauss_config(4, 4)
theta_mat, phi_mat = create_coordinate_matrices(cfg)
x, y, z = create_cartesian_coordinates(cfg)

# Create target field (Gaussian at north pole)
target_field = exp.(-((z .- 1).^2) ./ 0.1)

function mse_loss(sh_coeffs)
    predicted = synthesize(cfg, sh_coeffs)
    return sum((predicted - target_field).^2) / length(target_field)
end

# Initialize parameters
sh_coeffs = 0.1 * randn(get_nlm(cfg))
learning_rate = 0.01

# Training loop
for i in 1:100
    loss_val, grad = Zygote.withgradient(mse_loss, sh_coeffs)
    
    # Gradient descent update
    sh_coeffs .-= learning_rate .* grad[1]
    
    if i % 20 == 0
        println("Iteration $i: loss = $(loss_val[1])")
    end
end
```

### Adam Optimizer

```julia
# Using the built-in Adam optimizer from the Zygote extension
function adam_optimization()
    cfg = create_gauss_config(6, 6)
    sh_coeffs = randn(get_nlm(cfg))
    
    # Adam state variables
    m = zeros(length(sh_coeffs))  # First moment
    v = zeros(length(sh_coeffs))  # Second moment
    
    for t in 1:200
        loss_val = adam_step!(cfg, sh_coeffs, mse_loss, m, v, 0.9, 0.999, 0.001, t)
        
        if t % 50 == 0
            println("Iteration $t: loss = $loss_val")
        end
    end
end
```

## Advanced Usage

### Custom Chain Rules

For specialized functions, you can define custom differentiation rules:

```julia
using ChainRulesCore

# Example: Custom rule for a specialized function
function my_special_function(cfg, sh_coeffs)
    # Some complex operation involving SHT
    spatial = synthesize(cfg, sh_coeffs)
    return sum(spatial .* log.(abs.(spatial) .+ 1e-8))
end

# Define custom adjoint rule
function ChainRulesCore.rrule(::typeof(my_special_function), cfg, sh_coeffs)
    spatial = synthesize(cfg, sh_coeffs)
    result = sum(spatial .* log.(abs.(spatial) .+ 1e-8))
    
    function my_special_pullback(∂result)
        ∂spatial = ∂result .* (log.(abs.(spatial) .+ 1e-8) .+ spatial ./ (abs.(spatial) .+ 1e-8))
        ∂sh_coeffs = analyze(cfg, ∂spatial)  # Use SHT adjoint
        return (NoTangent(), NoTangent(), ∂sh_coeffs)
    end
    
    return result, my_special_pullback
end
```

### Neural Network Integration

```julia
# Using SHT layers in neural networks (conceptual)
using Flux

struct SHTLayer{T}
    cfg::SHTnsConfig{T}
end

function (layer::SHTLayer)(sh_coeffs)
    return synthesize(layer.cfg, sh_coeffs)
end

# Example network
cfg = create_gauss_config(8, 8)
model = Chain(
    Dense(get_nlm(cfg), get_nlm(cfg), tanh),
    SHTLayer(cfg),
    x -> reshape(x, :),
    Dense(get_nlat(cfg) * get_nphi(cfg), 1)
)

# Training would work automatically with Zygote
```

## Performance Tips

### ForwardDiff Performance
- Best for problems with few parameters (< 100)
- Use `ForwardDiff.GradientConfig` for repeated gradient computations:
  ```julia
  cfg_fd = ForwardDiff.GradientConfig(loss_function, sh_coeffs)
  gradient = ForwardDiff.gradient(loss_function, sh_coeffs, cfg_fd)
  ```

### Zygote Performance  
- Best for problems with many parameters
- Avoid scalar indexing in GPU contexts
- Use in-place operations where possible:
  ```julia
  function efficient_loss(sh_coeffs, spatial_buffer)
      synthesize!(cfg, sh_coeffs, spatial_buffer)  # In-place
      return sum(abs2, spatial_buffer)
  end
  ```

### Memory Management
- Pre-allocate arrays for repeated computations
- Use `similar` for type-stable allocations
- Consider using `StaticArrays` for small, fixed-size problems

## Troubleshooting

### Common Issues

1. **Type instabilities**: Ensure consistent floating-point types
   ```julia
   sh_coeffs = Vector{Float64}(undef, nlm)  # Good
   sh_coeffs = Vector{Any}(undef, nlm)      # Bad
   ```

2. **Scalar indexing errors**: Use broadcasting or vectorized operations
   ```julia
   result = sum(abs2.(spatial))  # Good
   result = sum([abs2(x) for x in spatial])  # Less efficient
   ```

3. **Missing gradients**: Check that all operations support AD
   ```julia
   # If you get "no method" errors, the function may need manual rules
   ```

### Performance Debugging

Use `@time` and `@allocated` to profile gradient computations:

```julia
# Profile forward pass
@time value = loss_function(sh_coeffs)

# Profile gradient computation
@time gradient = ForwardDiff.gradient(loss_function, sh_coeffs)

# Check allocations
@allocated ForwardDiff.gradient(loss_function, sh_coeffs)
```

## Mathematical Background

### Linear Operations
SHT operations are linear, so their adjoints are well-defined:
- Adjoint of synthesis is analysis: `∂L/∂sh = analyze(cfg, ∂L/∂spatial)`
- Adjoint of analysis is synthesis: `∂L/∂spatial = synthesize(cfg, ∂L/∂sh)`

### Power Spectrum Derivatives
For power spectrum `P_l = Σ_m |c_{l,m}|²`:
- `∂P_l/∂c_{l,m} = 2 * c_{l,m}` (for m > 0)
- `∂P_l/∂c_{l,0} = 2 * c_{l,0}` (for m = 0)

### Vector Transform Adjoints  
Vector synthesis and analysis are also adjoint pairs, preserving the relationship between spheroidal/toroidal coefficients and velocity components.

## References

- [ForwardDiff.jl Documentation](https://juliadiff.org/ForwardDiff.jl/)
- [Zygote.jl Documentation](https://fluxml.ai/Zygote.jl/)
- [ChainRules.jl Documentation](https://juliadiff.org/ChainRulesCore.jl/)
- [Automatic Differentiation Review](https://arxiv.org/abs/1502.05767)

## Examples Directory

See `examples/differentiation_examples.jl` for complete, runnable examples demonstrating all the features described in this guide.