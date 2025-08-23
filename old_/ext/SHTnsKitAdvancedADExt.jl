"""
Advanced automatic differentiation support for SHTnsKit.jl

This extension provides comprehensive AD support including:
- Matrix operators (Laplacian, cos(θ), etc.)
- Performance-optimized AD paths
- Advanced transform operations  
- Parallel/distributed AD operations
"""

using SHTnsKit
using ChainRulesCore
using LinearAlgebra

# Matrix operator AD rules
"""
    rrule(::typeof(apply_laplacian!), cfg, qlm_in, qlm_out)

Efficient reverse-mode AD for Laplacian operator.
The adjoint of the Laplacian is itself (self-adjoint operator).
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.apply_laplacian!), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    # Forward pass
    y = SHTnsKit.apply_laplacian!(cfg, qlm_in, qlm_out)
    
    function pullback_laplacian(ȳ)
        # Laplacian is self-adjoint: L* = L
        # So adjoint is just applying Laplacian to pullback signal
        qlm_in_bar = similar(qlm_in)
        SHTnsKit.apply_laplacian!(cfg, ȳ, qlm_in_bar)
        qlm_out_bar = @thunk(zero(qlm_out))  # Output pullback is zero
        
        return (NoTangent(), NoTangent(), qlm_in_bar, qlm_out_bar)
    end
    
    return y, pullback_laplacian
end

"""
    rrule(::typeof(apply_costheta_operator!), cfg, qlm_in, qlm_out)

Reverse-mode AD for cos(θ) coupling operator.
Uses transpose of coupling matrix for adjoint computation.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.apply_costheta_operator!), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    # Forward pass
    y = SHTnsKit.apply_costheta_operator!(cfg, qlm_in, qlm_out)
    
    function pullback_costheta(ȳ)
        # For matrix operator A: x̄ = A^T * ȳ
        # cos(θ) coupling matrix is symmetric, so A^T = A
        qlm_in_bar = similar(qlm_in)
        SHTnsKit.apply_costheta_operator!(cfg, ȳ, qlm_in_bar)
        qlm_out_bar = @thunk(zero(qlm_out))
        
        return (NoTangent(), NoTangent(), qlm_in_bar, qlm_out_bar)
    end
    
    return y, pullback_costheta
end

"""
    rrule(::typeof(apply_sintdtheta_operator!), cfg, qlm_in, qlm_out)

Reverse-mode AD for sin(θ)d/dθ operator.
Uses adjoint operator relationship.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.apply_sintdtheta_operator!), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    y = SHTnsKit.apply_sintdtheta_operator!(cfg, qlm_in, qlm_out)
    
    function pullback_sintdtheta(ȳ)
        # For derivative operators, adjoint is typically -transpose
        # This needs the actual adjoint operator implementation
        qlm_in_bar = similar(qlm_in)
        # Note: This would need proper adjoint implementation
        SHTnsKit.apply_sintdtheta_operator_adjoint!(cfg, ȳ, qlm_in_bar)
        qlm_out_bar = @thunk(zero(qlm_out))
        
        return (NoTangent(), NoTangent(), qlm_in_bar, qlm_out_bar)
    end
    
    return y, pullback_sintdtheta
end

# Advanced transforms AD rules

"""
    rrule(::typeof(sh_to_spat_l), cfg, qlm, l)

AD rule for single-l transform.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.sh_to_spat_l), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm::AbstractVector{Complex{T}}, 
                              l::Int) where T
    
    y = SHTnsKit.sh_to_spat_l(cfg, qlm, l)
    
    function pullback_sh_to_spat_l(ȳ)
        # Adjoint is analysis for same l
        qlm_bar = SHTnsKit.spat_to_sh_l(cfg, ȳ, l)
        return (NoTangent(), NoTangent(), qlm_bar, NoTangent())
    end
    
    return y, pullback_sh_to_spat_l
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.spat_to_sh_l), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              vr::AbstractVector{T}, 
                              l::Int) where T
    
    y = SHTnsKit.spat_to_sh_l(cfg, vr, l)
    
    function pullback_spat_to_sh_l(ȳ)
        # Adjoint is synthesis for same l
        vr_bar = SHTnsKit.sh_to_spat_l(cfg, ȳ, l)
        return (NoTangent(), NoTangent(), vr_bar, NoTangent())
    end
    
    return y, pullback_spat_to_sh_l
end

# Point evaluation AD rules

"""
    rrule(::typeof(sh_to_point), cfg, qlm, theta, phi)

AD rule for point evaluation - critical for neural PDE solvers.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.sh_to_point), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm::AbstractVector{T}, 
                              theta::Real, phi::Real) where T
    
    y = SHTnsKit.sh_to_point(cfg, qlm, theta, phi)
    
    function pullback_sh_to_point(ȳ)
        # Point evaluation adjoint: distribute point gradient to all modes
        qlm_bar = similar(qlm)
        
        # Compute gradient w.r.t each coefficient
        for (idx, (l, m)) in enumerate(cfg.lm_indices)
            # Evaluate spherical harmonic basis function at point
            ylm_value = SHTnsKit._evaluate_spherical_harmonic(cfg, l, m, theta, phi)
            qlm_bar[idx] = ȳ * conj(ylm_value)
        end
        
        return (NoTangent(), NoTangent(), qlm_bar, NoTangent(), NoTangent())
    end
    
    return y, pullback_sh_to_point
end

# Performance-optimized AD rules

"""
    rrule(::typeof(turbo_apply_laplacian!), cfg, qlm)

AD rule for turbo-optimized Laplacian.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.turbo_apply_laplacian!), 
                              cfg::SHTnsKit.SHTnsConfig{T}, 
                              qlm::AbstractVector{Complex{T}}) where T
    
    # Store original for potential restoration
    qlm_original = copy(qlm)
    y = SHTnsKit.turbo_apply_laplacian!(cfg, qlm)
    
    function pullback_turbo_laplacian(ȳ)
        # Laplacian is self-adjoint, use turbo version for efficiency
        qlm_bar = copy(ȳ)
        SHTnsKit.turbo_apply_laplacian!(cfg, qlm_bar)
        
        return (NoTangent(), NoTangent(), qlm_bar)
    end
    
    return y, pullback_turbo_laplacian
end

"""
    rrule(::typeof(turbo_auto_dispatch), cfg, op, qlm_in, qlm_out)

AD rule for automatic optimization dispatch.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.turbo_auto_dispatch), 
                              cfg::SHTnsKit.SHTnsConfig{T}, 
                              op::Symbol,
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    y = SHTnsKit.turbo_auto_dispatch(cfg, op, qlm_in, qlm_out)
    
    function pullback_turbo_auto_dispatch(ȳ)
        # Route to appropriate adjoint based on operator
        qlm_in_bar = similar(qlm_in)
        
        if op === :laplacian
            SHTnsKit.turbo_apply_laplacian!(cfg, copy(ȳ))
            qlm_in_bar .= ȳ
        elseif op === :costheta
            SHTnsKit.turbo_auto_dispatch(cfg, :costheta, ȳ, qlm_in_bar)
        else
            error("Unsupported operator for AD: $op")
        end
        
        qlm_out_bar = @thunk(zero(qlm_out))
        
        return (NoTangent(), NoTangent(), NoTangent(), qlm_in_bar, qlm_out_bar)
    end
    
    return y, pullback_turbo_auto_dispatch
end

# Parallel AD support

"""
    rrule(::typeof(parallel_apply_operator), op, pcfg, qlm_in, qlm_out)

AD rule for parallel matrix operations.
"""
function ChainRulesCore.rrule(::typeof(SHTnsKit.parallel_apply_operator), 
                              op::Symbol,
                              pcfg::SHTnsKit.ParallelSHTConfig{T}, 
                              qlm_in, qlm_out) where T
    
    y = SHTnsKit.parallel_apply_operator(op, pcfg, qlm_in, qlm_out)
    
    function pullback_parallel_operator(ȳ)
        # Use parallel adjoint computation
        qlm_in_bar = similar(qlm_in)
        
        if op === :laplacian
            # Laplacian is self-adjoint
            SHTnsKit.parallel_apply_operator(:laplacian, pcfg, ȳ, qlm_in_bar)
        elseif op === :costheta
            # cos(θ) is symmetric
            SHTnsKit.parallel_apply_operator(:costheta, pcfg, ȳ, qlm_in_bar)
        else
            error("Unsupported parallel operator for AD: $op")
        end
        
        qlm_out_bar = @thunk(zero(qlm_out))
        
        return (NoTangent(), NoTangent(), NoTangent(), qlm_in_bar, qlm_out_bar)
    end
    
    return y, pullback_parallel_operator
end

# Memory-efficient AD utilities

"""
    efficient_pullback_buffer(cfg::SHTnsConfig{T}) where T

Get thread-local buffer for AD computations to avoid allocations.
"""
function efficient_pullback_buffer(cfg::SHTnsKit.SHTnsConfig{T}) where T
    # Use advanced memory pool for zero-allocation AD
    pool = SHTnsKit.get_advanced_pool(cfg, :ad_pullback)
    return pool.temp_coeffs
end

# Gradient computation utilities for optimization

"""
    compute_spectral_gradient(cfg, loss_fn, qlm; method=:zygote)

Compute gradient of loss function with respect to spectral coefficients.
"""
function compute_spectral_gradient(cfg::SHTnsKit.SHTnsConfig, 
                                  loss_fn::Function, 
                                  qlm::AbstractVector{Complex{T}};
                                  method::Symbol=:zygote) where T
    
    if method === :zygote
        try
            using Zygote
            return Zygote.gradient(loss_fn, qlm)[1]
        catch
            @warn "Zygote not available, falling back to ForwardDiff"
            method = :forwarddiff
        end
    end
    
    if method === :forwarddiff
        try
            using ForwardDiff
            # For complex numbers, differentiate real and imaginary parts separately
            qlm_real = real.(qlm)
            qlm_imag = imag.(qlm)
            
            function real_loss(x_real, x_imag)
                x_complex = complex.(x_real, x_imag)
                return real(loss_fn(x_complex))
            end
            
            grad_real = ForwardDiff.gradient(x -> real_loss(x, qlm_imag), qlm_real)
            grad_imag = ForwardDiff.gradient(x -> real_loss(qlm_real, x), qlm_imag)
            
            return complex.(grad_real, grad_imag)
        catch e
            error("Both Zygote and ForwardDiff failed: $e")
        end
    end
    
    error("Unknown differentiation method: $method")
end

# High-level AD-aware optimization interface

"""
    optimize_spectral_coefficients(cfg, loss_fn, qlm0; 
                                 optimizer=:gradient_descent, 
                                 learning_rate=1e-3, 
                                 max_iterations=100)

Optimize spectral coefficients using automatic differentiation.
"""
function optimize_spectral_coefficients(cfg::SHTnsKit.SHTnsConfig, 
                                       loss_fn::Function, 
                                       qlm0::AbstractVector{Complex{T}};
                                       optimizer::Symbol=:gradient_descent,
                                       learning_rate::T=T(1e-3),
                                       max_iterations::Int=100) where T
    
    qlm = copy(qlm0)
    loss_history = T[]
    
    for iter in 1:max_iterations
        # Compute loss and gradient
        current_loss = loss_fn(qlm)
        push!(loss_history, current_loss)
        
        grad = compute_spectral_gradient(cfg, loss_fn, qlm)
        
        # Simple gradient descent update
        if optimizer === :gradient_descent
            qlm .-= learning_rate .* grad
        elseif optimizer === :momentum
            # Could implement momentum, Adam, etc.
            error("Momentum optimizer not yet implemented")
        else
            error("Unknown optimizer: $optimizer")
        end
        
        # Check convergence
        if iter > 1 && abs(loss_history[end] - loss_history[end-1]) < T(1e-8)
            @info "Converged after $iter iterations"
            break
        end
        
        if iter % 10 == 0
            @info "Iteration $iter: loss = $(current_loss)"
        end
    end
    
    return qlm, loss_history
end

# Export advanced AD functionality
export compute_spectral_gradient,
       optimize_spectral_coefficients,
       efficient_pullback_buffer

end # module