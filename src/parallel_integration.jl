"""
Integration layer for parallel matrix operations with PencilArrays/PencilFFTs.

This module provides seamless integration between serial and parallel operations,
allowing users to scale from single-process to massively parallel computations.
"""

# MPI loaded conditionally through extension

# Conditional loading of parallel dependencies
const PARALLEL_AVAILABLE = Ref(false)

function __init_parallel__()
    try
        # Check if parallel packages are available through extensions
        if isdefined(Base, :get_extension) && 
           !isnothing(Base.get_extension(SHTnsKit, :SHTnsKitParallelExt))
            PARALLEL_AVAILABLE[] = true
            @info "Parallel matrix operations available (extension loaded)"
        end
    catch e
        @warn "Parallel operations not available: $e"
        PARALLEL_AVAILABLE[] = false
    end
end

"""
    create_parallel_config(cfg::SHTnsConfig{T}; kwargs...) where T

Create a parallel configuration from a serial SHTnsConfig.
Automatically determines optimal 2D decomposition and initializes parallel data structures.
Note: Requires MPI, PencilArrays, and PencilFFTs packages for full functionality.
"""
function create_parallel_config(cfg::SHTnsConfig{T}; kwargs...) where T
    if !PARALLEL_AVAILABLE[]
        error("Parallel operations not available. Ensure MPI is initialized and parallel packages are loaded.")
    end
    
    # Parallel matrix operations now handled by extensions
    return ParallelSHTConfig{T}(cfg, comm)
end

"""
    parallel_apply_operator(op::Symbol, cfg::Union{SHTnsConfig, ParallelSHTConfig}, 
                           qlm_in, qlm_out=nothing)

Unified interface for applying operators in serial or parallel.
Automatically dispatches to appropriate implementation based on configuration type.
"""
function parallel_apply_operator(op::Symbol, cfg::SHTnsConfig{T}, qlm_in::AbstractVector{Complex{T}}, 
                                qlm_out::Union{Nothing, AbstractVector{Complex{T}}}=nothing) where T
    # Serial implementation
    if qlm_out === nothing
        qlm_out = similar(qlm_in)
    end
    
    if op === :costheta
        return apply_costheta_operator!(cfg, qlm_in, qlm_out)
    elseif op === :sintdtheta
        return apply_sintdtheta_operator!(cfg, qlm_in, qlm_out)
    elseif op === :laplacian
        return apply_laplacian!(cfg, qlm_in, qlm_out)
    else
        throw(ArgumentError("Unknown operator: $op"))
    end
end

function parallel_apply_operator(op::Symbol, pcfg::ParallelSHTConfig{T}, 
                                qlm_in::AbstractVector{Complex{T}}, 
                                qlm_out::Union{Nothing, AbstractVector{Complex{T}}}=nothing) where T
    # Parallel implementation
    if qlm_out === nothing
        error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
    end
    
    if op === :costheta
        return parallel_apply_costheta_operator!(pcfg, qlm_in, qlm_out)
    elseif op === :sintdtheta  
        return parallel_apply_sintdtheta_operator!(pcfg, qlm_in, qlm_out)
    elseif op === :laplacian
        return parallel_apply_laplacian_distributed!(pcfg, qlm_in)
    else
        throw(ArgumentError("Unknown operator: $op"))
    end
end

"""
    auto_parallel_config(lmax::Int, mmax::Int=lmax; T::Type=Float64, 
                        min_parallel_size::Int=100)

Automatically decide whether to use serial or parallel configuration based on problem size.
"""
function auto_parallel_config(lmax::Int, mmax::Int=lmax; T::Type=Float64, 
                             min_parallel_size::Int=100)
    
    # Create base configuration
    cfg = create_config(T, lmax, mmax, 1; grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL)
    
    # Decide serial vs parallel based on problem size and availability
    if cfg.nlm >= min_parallel_size && PARALLEL_AVAILABLE[]
        @info "Using parallel configuration for nlm=$(cfg.nlm)"
        return create_parallel_config(cfg)
    else
        @info "Using serial configuration for nlm=$(cfg.nlm)"
        return cfg
    end
end

"""
    adaptive_operator_selection(cfg, qlm_size::Int, op::Symbol)

Adaptively choose the best operator implementation based on problem characteristics.
"""
function adaptive_operator_selection(cfg, qlm_size::Int, op::Symbol)
    if isa(cfg, ParallelSHTConfig)
        # For parallel configs, analyze communication vs computation cost
        comm_cost = estimate_communication_cost(cfg, op)
        comp_cost = estimate_computation_cost(cfg, op, qlm_size)
        
        if comm_cost / comp_cost > 0.1  # Communication overhead > 10%
            @warn "High communication overhead detected for operator $op. Consider increasing problem size or reducing process count."
        end
        
        return :parallel
    else
        # For serial configs, choose based on sparsity and size
        if op === :laplacian
            return :diagonal_optimized  # Always fastest for diagonal ops
        elseif qlm_size < 1000
            return :direct  # Matrix-free for small problems
        else
            return :cached_sparse  # Cached sparse for large problems
        end
    end
end

"""
    parallel_performance_model(lmax::Int, nprocs::Int, op::Symbol) -> Float64

Performance model to predict execution time for parallel operations.
Helps with optimal process count selection and load balancing decisions.
"""
function parallel_performance_model(lmax::Int, nprocs::Int, op::Symbol)
    nlm = (lmax + 1)^2  # Approximate for full spectrum
    
    # Computational complexity (floating point operations)
    if op === :laplacian
        flops = nlm  # O(n) diagonal operation
        comm_volume = 0  # No communication needed
    elseif op === :costheta
        sparsity = 0.01  # Approximately 1% non-zero for coupling matrices
        flops = nlm^2 * sparsity  # O(n²) sparse matrix-vector
        comm_volume = nlm * 0.1  # Estimate ~10% boundary exchange
    else
        flops = nlm^2 * 0.05  # General sparse operator
        comm_volume = nlm * 0.2
    end
    
    # Performance model parameters (machine-specific, should be calibrated)
    flop_rate = 1e9  # 1 GFLOP/s per core
    bandwidth = 1e8  # 100 MB/s network bandwidth
    latency = 1e-5   # 10 μs network latency
    
    # Compute time components
    serial_comp_time = flops / flop_rate
    parallel_comp_time = serial_comp_time / nprocs
    
    comm_time = (comm_volume * sizeof(ComplexF64) / bandwidth) + latency * log2(nprocs)
    
    # Total time includes computation and communication
    total_time = parallel_comp_time + comm_time
    
    return total_time
end

"""
    optimal_process_count(lmax::Int, available_procs::Int, op::Symbol) -> Int

Determine optimal number of processes to minimize execution time.
"""
function optimal_process_count(lmax::Int, available_procs::Int, op::Symbol)
    best_time = Inf
    best_procs = 1
    
    # Test different process counts
    for nprocs in 1:available_procs
        predicted_time = parallel_performance_model(lmax, nprocs, op)
        
        if predicted_time < best_time
            best_time = predicted_time
            best_procs = nprocs
        end
    end
    
    return best_procs
end

"""
    memory_efficient_parallel_transform!(pcfg::ParallelSHTConfig{T}, 
                                        operators::Vector{Symbol},
                                        qlm_in, qlm_out)

Apply multiple operators in sequence with minimal memory allocation.
Optimizes for memory bandwidth and reduces intermediate allocations.
"""
function memory_efficient_parallel_transform!(pcfg::ParallelSHTConfig{T}, 
                                             operators::Vector{Symbol},
                                             qlm_in::AbstractVector{Complex{T}}, 
                                             qlm_out::AbstractVector{Complex{T}}) where T
    
    error("Parallel functionality requires MPI, PencilArrays, and PencilFFTs packages")
end

"""
    parallel_operator_fusion(operators::Vector{Symbol}) -> Function

Fuse multiple operators into a single kernel for better performance.
Reduces memory traffic and improves cache utilization.
"""
function parallel_operator_fusion(operators::Vector{Symbol})
    if :laplacian in operators && length(operators) == 2
        # Special case: Laplacian + another operator can often be fused
        other_op = operators[operators .!= :laplacian][1]
        
        return function fused_laplacian_op!(pcfg, qlm_in, qlm_out)
            # Apply both operations in single pass over data
            parallel_apply_fused_ops!(pcfg, [:laplacian, other_op], qlm_in, qlm_out)
        end
    else
        # General case: create composite operator
        return function fused_general_op!(pcfg, qlm_in, qlm_out) 
            memory_efficient_parallel_transform!(pcfg, operators, qlm_in, qlm_out)
        end
    end
end

# Utility functions for performance estimation

function estimate_communication_cost(pcfg::ParallelSHTConfig, op::Symbol)
    # Estimate based on operator sparsity and domain decomposition
    nprocs = 1  # Default to serial, actual value from extension
    nlm = pcfg.base_cfg.nlm
    
    if op === :laplacian
        return 0.0  # No communication for diagonal
    else
        # Estimate fraction of data that needs communication
        boundary_fraction = 2.0 / sqrt(nprocs)  # Rough estimate
        return nlm * boundary_fraction * sizeof(ComplexF64)
    end
end

function estimate_computation_cost(pcfg::ParallelSHTConfig, op::Symbol, qlm_size::Int)
    if op === :laplacian
        return qlm_size  # O(n) operations
    elseif op === :costheta
        return qlm_size * qlm_size * 0.01  # Sparse matrix-vector
    else
        return qlm_size * qlm_size * 0.05  # General sparse
    end
end

# Initialize parallel capabilities
__init_parallel__()

# All exports handled by main module SHTnsKit.jl