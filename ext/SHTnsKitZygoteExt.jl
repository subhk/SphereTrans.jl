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

end # module

