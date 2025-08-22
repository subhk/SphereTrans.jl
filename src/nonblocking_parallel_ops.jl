"""
Non-blocking parallel matrix operations with computation/communication overlap.

This module implements advanced MPI communication patterns to hide latency
and maximize parallel efficiency through overlapped computation and communication.

Note: Requires MPI, PencilArrays, and PencilFFTs packages for full functionality.
"""

# Placeholder async communication handle - actual implementation in extension
mutable struct AsyncCommHandle{T}
    stub::Bool
    AsyncCommHandle{T}() where T = new(true)
end

# Stub functions - implementations in SHTnsKitParallelExt
function create_async_comm_pattern(pcfg::ParallelSHTConfig{T}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function async_start_boundary_exchange!(handle::AsyncCommHandle{T}, 
                                       boundary_data::AbstractVector{Complex{T}}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function async_compute_internal!(cfg::SHTnsConfig{T}, op::Symbol,
                                qlm_internal::AbstractVector{Complex{T}},
                                qlm_result::AbstractVector{Complex{T}}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function async_finalize_boundary_exchange!(handle::AsyncCommHandle{T}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function async_parallel_costheta_operator!(pcfg::ParallelSHTConfig{T},
                                         qlm_in::AbstractVector{Complex{T}},
                                         qlm_out::AbstractVector{Complex{T}}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function pipeline_parallel_operators!(pcfg::ParallelSHTConfig{T},
                                     operations::Vector{Symbol},
                                     qlm_data::Vector{AbstractVector{Complex{T}}}) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

function benchmark_async_vs_sync_parallel(cfg::SHTnsConfig{T}, nprocs::Int) where T
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

# All exports handled by main module SHTnsKit.jl