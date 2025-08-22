"""
Integration layer for parallel matrix operations with PencilArrays/PencilFFTs.

This module provides seamless integration between serial and parallel operations,
allowing users to scale from single-process to massively parallel computations.
"""

using MPI

# Conditional loading of parallel dependencies
const PARALLEL_AVAILABLE = Ref(false)

function __init_parallel__()
    try
        # Check if MPI is initialized and parallel packages are available
        if MPI.Initialized() && !MPI.Finalized()
            PARALLEL_AVAILABLE[] = true
            @info "Parallel matrix operations enabled with $(MPI.Comm_size(MPI.COMM_WORLD)) processes"
        end
    catch e
        @warn "Parallel operations not available: $e"
        PARALLEL_AVAILABLE[] = false
    end
end

"""
    create_parallel_config(cfg::SHTnsConfig{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T

Create a parallel configuration from a serial SHTnsConfig.
Automatically determines optimal 2D decomposition and initializes parallel data structures.
"""
function create_parallel_config(cfg::SHTnsConfig{T}, comm::MPI.Comm=MPI.COMM_WORLD) where T
    if !PARALLEL_AVAILABLE[]
        error("Parallel operations not available. Ensure MPI is initialized and parallel packages are loaded.")
    end
    
    include("parallel_matrix_ops.jl")
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
        error("Unknown operator: $op")
    end
end

function parallel_apply_operator(op::Symbol, pcfg::ParallelSHTConfig{T}, 
                                qlm_in::PencilArray{Complex{T}}, 
                                qlm_out::Union{Nothing, PencilArray{Complex{T}}}=nothing) where T
    # Parallel implementation
    if qlm_out === nothing
        qlm_out = allocate_array(pcfg.spectral_pencil, Complex{T})
    end
    
    if op === :costheta
        return parallel_apply_costheta_operator!(pcfg, qlm_in, qlm_out)
    elseif op === :sintdtheta  
        return parallel_apply_sintdtheta_operator!(pcfg, qlm_in, qlm_out)
    elseif op === :laplacian
        return parallel_apply_laplacian_distributed!(pcfg, qlm_in)
    else
        error("Unknown operator: $op")
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
    if cfg.nlm >= min_parallel_size && PARALLEL_AVAILABLE[] && MPI.Comm_size(MPI.COMM_WORLD) > 1
        @info "Using parallel configuration for nlm=$(cfg.nlm) with $(MPI.Comm_size(MPI.COMM_WORLD)) processes"
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
                                             qlm_in::PencilArray{Complex{T}}, 
                                             qlm_out::PencilArray{Complex{T}}) where T
    
    # Pre-allocate single working array for all operations
    work_array = allocate_array(pcfg.spectral_pencil, Complex{T})
    
    current_in = qlm_in
    current_out = work_array
    
    for (i, op) in enumerate(operators)
        # Alternate between work_array and qlm_out to minimize copies
        if i == length(operators)
            current_out = qlm_out  # Final result goes to output
        elseif i > 1
            current_in, current_out = current_out, current_in  # Swap
        end
        
        # Apply operator
        parallel_apply_operator(op, pcfg, current_in, current_out)
        
        # Memory fence to ensure operation completion before next
        MPI.Barrier(pcfg.comm)
    end
    
    return qlm_out
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
    nprocs = MPI.Comm_size(pcfg.comm)
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