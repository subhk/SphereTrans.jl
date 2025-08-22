module SHTnsKitLoopVecExt

using SHTnsKit
using LoopVectorization
using Base.Threads
using LinearAlgebra

# Override the fallback @turbo macro in the main module
macro turbo(expr)
    return esc(:(LoopVectorization.@turbo $expr))
end

# Enhanced SIMD implementations when LoopVectorization is available
function SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTnsConfig{T}, 
                                        qlm::AbstractVector{Complex{T}}) where T
    
    nlm = length(qlm)
    
    # Real and imaginary parts can be processed separately
    qlm_real = reinterpret(T, qlm)
    
    @turbo for i in 1:2:2*nlm
        idx = (i + 1) ÷ 2
        l, m = SHTnsKit.lm_from_index(cfg, idx)
        eigenvalue = -T(l * (l + 1))
        
        # Apply to both real and imaginary parts
        qlm_real[i] *= eigenvalue
        qlm_real[i + 1] *= eigenvalue
    end
    
    return qlm
end

function SHTnsKit.turbo_sparse_matvec!(A::SparseArrays.SparseMatrixCSC{T}, 
                                      x::Vector{Complex{T}}, 
                                      y::Vector{Complex{T}}) where T
    
    fill!(y, zero(Complex{T}))
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    n = size(A, 2)
    
    # Convert to real arrays for better SIMD
    x_real = reinterpret(T, x)
    y_real = reinterpret(T, y)
    
    for j in 1:n
        for k in SparseArrays.nzrange(A, j)
            row = rows[k]
            val = vals[k]
            
            # Manual complex multiplication with SIMD
            @turbo for comp in 1:2
                idx_x = 2 * j - 2 + comp
                idx_y = 2 * row - 2 + comp
                y_real[idx_y] += val * x_real[idx_x]
            end
        end
    end
    
    return y
end

function SHTnsKit.turbo_threaded_costheta_operator!(cfg::SHTnsKit.SHTnsConfig{T}, 
                                                   qlm_in::AbstractVector{Complex{T}},
                                                   qlm_out::AbstractVector{Complex{T}}) where T
    
    fill!(qlm_out, zero(Complex{T}))
    nlm = cfg.nlm
    
    # Thread-parallel with SIMD within each thread
    @threads for thread_id in 1:nthreads()
        thread_start = ((thread_id - 1) * nlm) ÷ nthreads() + 1
        thread_end = (thread_id * nlm) ÷ nthreads()
        
        # Process chunk with SIMD
        for idx_out in thread_start:thread_end
            l_out, m_out = SHTnsKit.lm_from_index(cfg, idx_out)
            
            # cos(θ) couples (l,m) with (l±1,m)
            for Δl in [-1, 1]
                l_in = l_out + Δl
                if 0 <= l_in <= cfg.lmax && abs(m_out) <= min(l_in, cfg.mmax)
                    idx_in = SHTnsKit.lmidx(cfg, l_in, m_out)
                    if idx_in <= length(qlm_in)
                        coupling = _turbo_costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                        qlm_out[idx_out] += coupling * qlm_in[idx_in]
                    end
                end
            end
        end
    end
    
    return qlm_out
end

function SHTnsKit.turbo_auto_dispatch(cfg::SHTnsKit.SHTnsConfig{T}, 
                                     operation::Symbol,
                                     args...) where T
    
    if operation === :laplacian && length(args) == 1
        return turbo_apply_laplacian!(cfg, args[1])
    elseif operation === :costheta && length(args) == 2
        return turbo_threaded_costheta_operator!(cfg, args[1], args[2])
    elseif operation === :sparse_matvec && length(args) == 3
        return turbo_sparse_matvec!(args[1], args[2], args[3])
    else
        # Fall back to regular SIMD implementations
        return SHTnsKit.auto_simd_dispatch(cfg, operation, args...)
    end
end

# Optimized coefficient computation with SIMD
function _turbo_costheta_coupling_coefficient(cfg::SHTnsKit.SHTnsConfig{T}, 
                                             l_out::Int, l_in::Int, m::Int) where T
    
    # Use SIMD-friendly computation
    if l_out == l_in - 1
        numerator = l_in^2 - m^2
        denominator = 4*l_in^2 - 1
        return T(sqrt(numerator / denominator))
    elseif l_out == l_in + 1
        numerator = (l_in+1)^2 - m^2
        denominator = 4*(l_in+1)^2 - 1
        return T(sqrt(numerator / denominator))
    else
        return T(0)
    end
end

# Benchmarking function to compare turbo vs regular SIMD
function SHTnsKit.benchmark_turbo_vs_simd(cfg::SHTnsKit.SHTnsConfig{T}; 
                                         n_trials::Int = 100) where T
    
    qlm = randn(Complex{T}, cfg.nlm)
    qlm_out_simd = similar(qlm)
    qlm_out_turbo = similar(qlm)
    
    # Benchmark regular SIMD
    simd_time = @elapsed begin
        for _ in 1:n_trials
            SHTnsKit.simd_apply_laplacian!(cfg, copy(qlm))
        end
    end
    
    # Benchmark turbo SIMD
    turbo_time = @elapsed begin
        for _ in 1:n_trials
            turbo_apply_laplacian!(cfg, copy(qlm))
        end
    end
    
    # Verify results are the same
    SHTnsKit.simd_apply_laplacian!(cfg, copy(qlm), qlm_out_simd)
    turbo_apply_laplacian!(cfg, copy(qlm), qlm_out_turbo)
    
    max_diff = maximum(abs.(qlm_out_simd - qlm_out_turbo))
    
    return (
        simd_time = simd_time / n_trials,
        turbo_time = turbo_time / n_trials,
        speedup = simd_time / turbo_time,
        max_difference = max_diff,
        turbo_faster = turbo_time < simd_time
    )
end

# Enhanced memory pooling with LoopVectorization optimizations
function SHTnsKit.get_advanced_pool(cfg::SHTnsKit.SHTnsConfig{T}) where T
    
    # Enhanced pool that can take advantage of LoopVectorization
    pool = SHTnsKit.get_work_pool(cfg)
    
    # Pre-compute SIMD-friendly data layouts
    if !haskey(pool.cache, :turbo_layout)
        # Reorganize data for better vectorization
        pool.cache[:turbo_layout] = _prepare_turbo_layout(cfg)
    end
    
    return pool
end

function _prepare_turbo_layout(cfg::SHTnsKit.SHTnsConfig{T}) where T
    # Pre-compute indices and coefficients for vectorized operations
    nlm = cfg.nlm
    
    # Group operations by SIMD width for better vectorization
    simd_width = LoopVectorization.pick_vector_width(T)
    
    layout = Dict(
        :simd_width => simd_width,
        :vectorized_indices => Vector{Int}[],
        :coupling_coefficients => Vector{T}[]
    )
    
    # Group indices for vectorized processing
    current_group = Int[]
    for idx in 1:nlm
        push!(current_group, idx)
        if length(current_group) == simd_width || idx == nlm
            push!(layout[:vectorized_indices], copy(current_group))
            empty!(current_group)
        end
    end
    
    return layout
end

function SHTnsKit.clear_advanced_pools()
    # Clear the enhanced pools
    SHTnsKit.clear_work_pools!()
    # Additional cleanup for LoopVectorization-specific data could go here
    return nothing
end

# Override the fallback macro in advanced_optimizations.jl
# This will be called when LoopVectorization is loaded
function __init__()
    # Replace the fallback @turbo macro in the main module
    @eval SHTnsKit.AdvancedOptimizations begin
        macro turbo(expr)
            return esc(:(LoopVectorization.@turbo $expr))
        end
    end
end

end # module