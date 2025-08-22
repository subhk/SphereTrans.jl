module SHTnsKitZygoteExt

using SHTnsKit
using ChainRulesCore

# Scalar real transforms with optimized memory usage
function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesize), cfg::SHTnsKit.SHTnsConfig, sh::AbstractVector)
    y = SHTnsKit.synthesize(cfg, sh)
    function pullback(ȳ)
        # Use pre-allocated workspace if available in config for adjoint computation
        # adjoint: sh̄ = analyze(cfg, ȳ)
        result = SHTnsKit.analyze(cfg, ȳ)
        return (NoTangent(), NoTangent(), result)
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.analyze), cfg::SHTnsKit.SHTnsConfig, spat::AbstractMatrix)
    y = SHTnsKit.analyze(cfg, spat)
    function pullback(ȳ)
        # Use pre-allocated workspace if available in config for adjoint computation
        # adjoint: spat̄ = synthesize(cfg, ȳ)
        result = SHTnsKit.synthesize(cfg, ȳ)
        return (NoTangent(), NoTangent(), result)
    end
    return y, pullback
end

# Scalar complex transforms with type-stable operations
function ChainRulesCore.rrule(::typeof(SHTnsKit.cplx_sh_to_spat), cfg::SHTnsKit.SHTnsConfig, sh::AbstractVector{<:Complex})
    y = SHTnsKit.cplx_sh_to_spat(cfg, sh)
    function pullback(ȳ)
        result = SHTnsKit.cplx_spat_to_sh(cfg, ȳ)
        return (NoTangent(), NoTangent(), result)
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.cplx_spat_to_sh), cfg::SHTnsKit.SHTnsConfig, spat::AbstractMatrix{<:Complex})
    y = SHTnsKit.cplx_spat_to_sh(cfg, spat)
    function pullback(ȳ)
        result = SHTnsKit.cplx_sh_to_spat(cfg, ȳ)
        return (NoTangent(), NoTangent(), result)
    end
    return y, pullback
end

# Complex vector transforms
function ChainRulesCore.rrule(::typeof(SHTnsKit.cplx_synthesize_vector), cfg::SHTnsKit.SHTnsConfig,
                              S::AbstractVector{<:Complex}, T::AbstractVector{<:Complex})
    uθ, uφ = SHTnsKit.cplx_synthesize_vector(cfg, S, T)
    function pullback(ȳ)
        ȳθ, ȳφ = ȳ
        S̄, T̄ = SHTnsKit.cplx_analyze_vector(cfg, ȳθ, ȳφ)
        return (NoTangent(), NoTangent(), S̄, T̄)
    end
    return (uθ, uφ), pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.cplx_analyze_vector), cfg::SHTnsKit.SHTnsConfig,
                              uθ::AbstractMatrix{<:Complex}, uφ::AbstractMatrix{<:Complex})
    S, T = SHTnsKit.cplx_analyze_vector(cfg, uθ, uφ)
    function pullback(ȳ)
        S̄, T̄ = ȳ
        uθ̄, uφ̄ = SHTnsKit.cplx_synthesize_vector(cfg, S̄, T̄)
        return (NoTangent(), NoTangent(), uθ̄, uφ̄)
    end
    return (S, T), pullback
end

# Matrix operator AD rules
function ChainRulesCore.rrule(::typeof(SHTnsKit.apply_laplacian!), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    y = SHTnsKit.apply_laplacian!(cfg, qlm_in, qlm_out)
    
    function pullback_laplacian(ȳ)
        # Laplacian is self-adjoint
        qlm_in_bar = similar(qlm_in)
        SHTnsKit.apply_laplacian!(cfg, ȳ, qlm_in_bar)
        return (NoTangent(), NoTangent(), qlm_in_bar, @thunk(zero(qlm_out)))
    end
    
    return y, pullback_laplacian
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.apply_costheta_operator!), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm_in::AbstractVector{Complex{T}}, 
                              qlm_out::AbstractVector{Complex{T}}) where T
    
    y = SHTnsKit.apply_costheta_operator!(cfg, qlm_in, qlm_out)
    
    function pullback_costheta(ȳ)
        # cos(θ) operator is symmetric (self-adjoint)
        qlm_in_bar = similar(qlm_in)
        SHTnsKit.apply_costheta_operator!(cfg, ȳ, qlm_in_bar)
        return (NoTangent(), NoTangent(), qlm_in_bar, @thunk(zero(qlm_out)))
    end
    
    return y, pullback_costheta
end

# Advanced transform AD rules
function ChainRulesCore.rrule(::typeof(SHTnsKit.sh_to_point), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm::AbstractVector{T}, 
                              theta::Real, phi::Real) where T
    
    y = SHTnsKit.sh_to_point(cfg, qlm, theta, phi)
    
    function pullback_sh_to_point(ȳ)
        qlm_bar = similar(qlm)
        lm_indices = cfg.lm_indices
        
        # Distribute point gradient to all spectral modes
        @inbounds for (idx, (l, m)) in enumerate(lm_indices)
            ylm_value = SHTnsKit._evaluate_spherical_harmonic(cfg, l, m, theta, phi)
            qlm_bar[idx] = ȳ * conj(ylm_value)
        end
        
        return (NoTangent(), NoTangent(), qlm_bar, NoTangent(), NoTangent())
    end
    
    return y, pullback_sh_to_point
end

# Single-l transform AD rules
function ChainRulesCore.rrule(::typeof(SHTnsKit.sh_to_spat_l), 
                              cfg::SHTnsKit.SHTnsConfig, 
                              qlm::AbstractVector{Complex{T}}, 
                              l::Int) where T
    
    y = SHTnsKit.sh_to_spat_l(cfg, qlm, l)
    
    function pullback_sh_to_spat_l(ȳ)
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
        vr_bar = SHTnsKit.sh_to_spat_l(cfg, ȳ, l)
        return (NoTangent(), NoTangent(), vr_bar, NoTangent())
    end
    
    return y, pullback_spat_to_sh_l
end

# Performance-optimized AD rules
function ChainRulesCore.rrule(::typeof(SHTnsKit.turbo_apply_laplacian!), 
                              cfg::SHTnsKit.SHTnsConfig{T}, 
                              qlm::AbstractVector{Complex{T}}) where T
    
    qlm_original = copy(qlm)
    y = SHTnsKit.turbo_apply_laplacian!(cfg, qlm)
    
    function pullback_turbo_laplacian(ȳ)
        # Use turbo version for efficiency in adjoint too
        qlm_bar = copy(ȳ)
        SHTnsKit.turbo_apply_laplacian!(cfg, qlm_bar)
        return (NoTangent(), NoTangent(), qlm_bar)
    end
    
    return y, pullback_turbo_laplacian
end

# Parallel operation AD rules
function ChainRulesCore.rrule(::typeof(SHTnsKit.parallel_apply_operator), 
                              op::Symbol,
                              pcfg::SHTnsKit.ParallelSHTConfig{T}, 
                              qlm_in, qlm_out) where T
    
    y = SHTnsKit.parallel_apply_operator(op, pcfg, qlm_in, qlm_out)
    
    function pullback_parallel_operator(ȳ)
        qlm_in_bar = similar(qlm_in)
        
        if op === :laplacian
            SHTnsKit.parallel_apply_operator(:laplacian, pcfg, ȳ, qlm_in_bar)
        elseif op === :costheta
            SHTnsKit.parallel_apply_operator(:costheta, pcfg, ȳ, qlm_in_bar)
        else
            error("Unsupported parallel operator for AD: $op")
        end
        
        return (NoTangent(), NoTangent(), NoTangent(), qlm_in_bar, @thunk(zero(qlm_out)))
    end
    
    return y, pullback_parallel_operator
end

# Memory-efficient pullback operations
function ChainRulesCore.rrule(::typeof(SHTnsKit.memory_efficient_parallel_transform!), 
                              pcfg::SHTnsKit.ParallelSHTConfig{T}, 
                              operators::Vector{Symbol},
                              qlm_in, qlm_out) where T
    
    y = SHTnsKit.memory_efficient_parallel_transform!(pcfg, operators, qlm_in, qlm_out)
    
    function pullback_memory_efficient_transform(ȳ)
        # Apply adjoint operators in reverse order
        qlm_in_bar = similar(qlm_in)
        current_pullback = ȳ
        
        # Create temporary for intermediate results
        temp_pullback = similar(qlm_in)
        
        for op in reverse(operators)
            if op === :laplacian
                SHTnsKit.parallel_apply_operator(:laplacian, pcfg, current_pullback, temp_pullback)
            elseif op === :costheta  
                SHTnsKit.parallel_apply_operator(:costheta, pcfg, current_pullback, temp_pullback)
            else
                error("Unsupported operator in chain for AD: $op")
            end
            current_pullback, temp_pullback = temp_pullback, current_pullback
        end
        
        copyto!(qlm_in_bar, current_pullback)
        
        return (NoTangent(), NoTangent(), NoTangent(), qlm_in_bar, @thunk(zero(qlm_out)))
    end
    
    return y, pullback_memory_efficient_transform
end

end # module

