"""
Parallel matrix operations using PencilArrays and PencilFFTs for 2D parallelization.

This module implements distributed spherical harmonic matrix operations that can
scale across multiple processes with efficient 2D domain decomposition.
"""

using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays

# Global configuration for parallel operations
struct ParallelSHTConfig{T<:AbstractFloat}
    # Base SHT configuration
    base_cfg::SHTnsConfig{T}
    
    # MPI and pencil configuration
    comm::MPI.Comm
    pencil_decomp::PencilDecomposition
    
    # Distributed arrays for spectral coefficients
    spectral_pencil::Pencil{2}  # (l, m) decomposition
    spatial_pencil::Pencil{2}   # (θ, φ) decomposition
    
    # FFT transforms between pencils
    fft_plan::PencilFFTPlan
    
    # Local index ranges for this process
    local_l_range::UnitRange{Int}
    local_m_range::UnitRange{Int}
    
    # Parallel operator caches
    local_operators::Dict{Symbol, Any}
    
    function ParallelSHTConfig{T}(base_cfg::SHTnsConfig{T}, comm::MPI.Comm) where T
        # Set up 2D decomposition: one dimension for l, one for m
        comm_size = MPI.Comm_size(comm)
        
        # Choose optimal 2D grid factorization
        dims = optimal_2d_dims(comm_size)
        
        # Create pencil decomposition
        # First pencil: distribute across l (degrees)
        # Second pencil: distribute across m (orders)
        spectral_size = (base_cfg.lmax + 1, 2*base_cfg.mmax + 1)
        spatial_size = (base_cfg.nlat, base_cfg.nphi)
        
        pencil_decomp = PencilDecomposition(spectral_size, comm; dims=dims)
        
        # Create pencils for different data layouts
        spectral_pencil = Pencil(pencil_decomp, spectral_size, (1, 2))  # (l, m) distributed
        spatial_pencil = Pencil(pencil_decomp, spatial_size, (1, 2))    # (θ, φ) distributed
        
        # Set up FFT plan for transforms between spectral and spatial
        fft_plan = PencilFFTPlan(spatial_pencil, spectral_pencil, FFTW.FORWARD)
        
        # Determine local ranges for this MPI rank
        local_l_range = get_local_range(spectral_pencil, 1)
        local_m_range = get_local_range(spectral_pencil, 2)
        
        new{T}(base_cfg, comm, pencil_decomp, spectral_pencil, spatial_pencil, 
               fft_plan, local_l_range, local_m_range, Dict{Symbol, Any}())
    end
end

"""
    optimal_2d_dims(nprocs::Int) -> Tuple{Int,Int}

Find optimal 2D processor grid dimensions for load balancing.
Tries to make dimensions as close as possible for best communication.
"""
function optimal_2d_dims(nprocs::Int)
    # Find factors of nprocs
    factors = []
    for i in 1:Int(sqrt(nprocs))
        if nprocs % i == 0
            push!(factors, (i, nprocs ÷ i))
        end
    end
    
    # Choose factors closest to square
    if isempty(factors)
        return (1, nprocs)
    end
    
    # Minimize |p - q| for p × q = nprocs
    best_diff = typemax(Int)
    best_dims = (1, nprocs)
    
    for (p, q) in factors
        diff = abs(p - q)
        if diff < best_diff
            best_diff = diff
            best_dims = (p, q)
        end
    end
    
    return best_dims
end

"""
    parallel_mul_ct_matrix(pcfg::ParallelSHTConfig{T}) -> DistributedSparseMatrix

Compute cos(θ) coupling matrix using 2D parallelization.
Each process computes its local block of the matrix.
"""
function parallel_mul_ct_matrix(pcfg::ParallelSHTConfig{T}) where T
    base_cfg = pcfg.base_cfg
    
    # Get local ranges for this process
    l_range = pcfg.local_l_range
    m_range = pcfg.local_m_range
    
    # Local sparse matrix components
    local_I = Int[]
    local_J = Int[]
    local_vals = T[]
    
    # Compute local portion of coupling matrix
    for (local_i, l_out) in enumerate(l_range)
        for (local_j, m_out) in enumerate(m_range)
            
            # Convert to global indices
            global_idx_out = get_global_spectral_index(base_cfg, l_out, m_out)
            
            # Compute couplings for this (l_out, m_out)
            for l_in in max(0, l_out-1):min(base_cfg.lmax, l_out+1)
                if abs(l_in - l_out) == 1  # cos(θ) couples ±1 in l only
                    
                    global_idx_in = get_global_spectral_index(base_cfg, l_in, m_out)
                    coeff = _costheta_coupling_coefficient(base_cfg, l_out, l_in, m_out)
                    
                    if abs(coeff) > eps(T)
                        push!(local_I, global_idx_out)
                        push!(local_J, global_idx_in)
                        push!(local_vals, coeff)
                    end
                end
            end
        end
    end
    
    # Create local sparse matrix
    total_size = base_cfg.nlm
    local_matrix = sparse(local_I, local_J, local_vals, total_size, total_size)
    
    return DistributedSparseMatrix(local_matrix, pcfg.comm)
end

"""
    parallel_apply_costheta_operator!(pcfg::ParallelSHTConfig{T}, 
                                     qlm_in::PencilArray, qlm_out::PencilArray)

Apply cos(θ) operator using parallel matrix-vector multiplication.
Uses optimized distributed sparse matrix operations.
"""
function parallel_apply_costheta_operator!(pcfg::ParallelSHTConfig{T}, 
                                          qlm_in::PencilArray{Complex{T}}, 
                                          qlm_out::PencilArray{Complex{T}}) where T
    
    # Get or create cached parallel operator
    if !haskey(pcfg.local_operators, :costheta)
        pcfg.local_operators[:costheta] = parallel_mul_ct_matrix(pcfg)
    end
    
    op_matrix = pcfg.local_operators[:costheta]
    
    # Parallel matrix-vector multiplication
    parallel_sparse_matvec!(op_matrix, qlm_in, qlm_out, pcfg.comm)
    
    return qlm_out
end

"""
    parallel_sparse_matvec!(A::DistributedSparseMatrix, x::PencilArray, y::PencilArray, comm::MPI.Comm)

Optimized parallel sparse matrix-vector multiplication.
Uses non-blocking communication for overlap of computation and communication.
"""
function parallel_sparse_matvec!(A::DistributedSparseMatrix{T}, 
                                 x::PencilArray{Complex{T}}, 
                                 y::PencilArray{Complex{T}}, 
                                 comm::MPI.Comm) where T
    
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Step 1: Local matrix-vector multiply (no communication needed)
    local_A = A.local_matrix
    local_x = parent(x)  # Get local portion
    local_y = parent(y)
    
    # Apply local sparse matrix (real and imaginary separately for efficiency)
    n_local = length(local_x)
    x_real = Vector{T}(undef, n_local)
    x_imag = Vector{T}(undef, n_local)
    y_real = Vector{T}(undef, n_local) 
    y_imag = Vector{T}(undef, n_local)
    
    @inbounds @simd for i in 1:n_local
        x_real[i] = real(local_x[i])
        x_imag[i] = imag(local_x[i])
    end
    
    # Use optimized sparse BLAS
    mul!(y_real, local_A, x_real)
    mul!(y_imag, local_A, x_imag)
    
    @inbounds @simd for i in 1:n_local
        local_y[i] = complex(y_real[i], y_imag[i])
    end
    
    # Step 2: Communication phase for inter-process dependencies
    # Use non-blocking MPI for maximum overlap
    requests = MPI.Request[]
    
    # Exchange boundary data with neighboring processes
    for neighbor_rank in get_communication_neighbors(A, rank)
        send_buf, recv_buf = prepare_boundary_exchange(A, x, neighbor_rank)
        
        # Non-blocking send/receive
        send_req = MPI.Isend(send_buf, comm; dest=neighbor_rank, tag=1)
        recv_req = MPI.Irecv!(recv_buf, comm; source=neighbor_rank, tag=1)
        
        push!(requests, send_req)
        push!(requests, recv_req)
    end
    
    # Step 3: Overlap computation with communication
    # Compute contributions that don't need communication while waiting
    compute_local_contributions!(A, x, y)
    
    # Step 4: Complete communication and add remote contributions
    MPI.Waitall(requests)
    
    # Add contributions from other processes
    add_remote_contributions!(A, y, received_data)
    
    return y
end

"""
    parallel_apply_laplacian_distributed!(pcfg::ParallelSHTConfig{T}, 
                                         qlm::PencilArray{Complex{T}})

Apply Laplacian operator in parallel. This is embarrassingly parallel since 
it's diagonal: ∇²Y_l^m = -l(l+1) Y_l^m.
"""
function parallel_apply_laplacian_distributed!(pcfg::ParallelSHTConfig{T}, 
                                              qlm::PencilArray{Complex{T}}) where T
    
    base_cfg = pcfg.base_cfg
    local_data = parent(qlm)
    
    # Get local l,m ranges
    l_range = pcfg.local_l_range
    m_range = pcfg.local_m_range
    
    # Apply Laplacian locally (no communication needed!)
    @inbounds for (local_i, l) in enumerate(l_range)
        for (local_j, m) in enumerate(m_range)
            local_idx = linear_index_from_lm(local_i, local_j, length(m_range))
            eigenvalue = -T(l * (l + 1))
            local_data[local_idx] *= eigenvalue
        end
    end
    
    # No MPI communication needed for diagonal operator!
    return qlm
end

"""
    parallel_fft_synthesis!(pcfg::ParallelSHTConfig{T}, 
                           qlm::PencilArray{Complex{T}}, 
                           spatial_field::PencilArray{T})

Perform parallel spherical harmonic synthesis using distributed FFTs.
"""
function parallel_fft_synthesis!(pcfg::ParallelSHTConfig{T}, 
                                 qlm::PencilArray{Complex{T}}, 
                                 spatial_field::PencilArray{T}) where T
    
    # Step 1: Transform from (l,m) pencil to (θ,φ) pencil
    # This involves global transpose communication
    qlm_spatial_layout = allocate_array(pcfg.spatial_pencil, Complex{T})
    transpose!(qlm_spatial_layout, qlm, pcfg.pencil_decomp)
    
    # Step 2: Apply Legendre transform in θ direction (local operation)
    apply_legendre_transform_local!(pcfg, qlm_spatial_layout, spatial_field)
    
    # Step 3: Apply FFT in φ direction using PencilFFTs
    fft_plan = pcfg.fft_plan
    spatial_freq = allocate_array(pcfg.spatial_pencil, Complex{T})
    
    # Execute parallel FFT
    mul!(spatial_freq, fft_plan, spatial_field)
    
    # Step 4: Post-process and extract real part
    extract_real_spatial_field!(spatial_field, spatial_freq)
    
    return spatial_field
end

"""
    threaded_coupling_computation(pcfg::ParallelSHTConfig{T}, 
                                 l_range::UnitRange, m_range::UnitRange)

Use shared-memory threading within each MPI process for coupling computation.
This provides additional parallelism beyond the MPI decomposition.
"""
function threaded_coupling_computation(pcfg::ParallelSHTConfig{T}, 
                                      l_range::UnitRange, m_range::UnitRange) where T
    base_cfg = pcfg.base_cfg
    
    # Thread-local storage
    n_threads = Threads.nthreads()
    thread_results = [Vector{Tuple{Int,Int,T}}() for _ in 1:n_threads]
    
    # Parallel loop over l and m with threading
    total_work = length(l_range) * length(m_range)
    
    Threads.@threads for work_idx in 1:total_work
        # Convert linear work index to (l, m) coordinates
        l_idx = ((work_idx - 1) ÷ length(m_range)) + 1
        m_idx = ((work_idx - 1) % length(m_range)) + 1
        
        l = l_range[l_idx]
        m = m_range[m_idx]
        
        thread_id = Threads.threadid()
        
        # Compute coupling coefficients for this (l,m)
        for l_couple in max(0, l-1):min(base_cfg.lmax, l+1)
            if abs(l_couple - l) == 1
                coeff = _costheta_coupling_coefficient(base_cfg, l, l_couple, m)
                if abs(coeff) > eps(T)
                    global_out = get_global_spectral_index(base_cfg, l, m)
                    global_in = get_global_spectral_index(base_cfg, l_couple, m)
                    push!(thread_results[thread_id], (global_out, global_in, coeff))
                end
            end
        end
    end
    
    # Merge thread results
    all_results = Vector{Tuple{Int,Int,T}}()
    for thread_result in thread_results
        append!(all_results, thread_result)
    end
    
    return all_results
end

"""
    optimize_communication_pattern(pcfg::ParallelSHTConfig{T})

Analyze and optimize MPI communication patterns for this configuration.
Pre-computes communication schedules to minimize latency.
"""
function optimize_communication_pattern(pcfg::ParallelSHTConfig{T}) where T
    comm = pcfg.comm
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Analyze which processes need to communicate
    comm_graph = build_communication_graph(pcfg)
    
    # Optimize communication schedule
    # 1. Group small messages to reduce latency
    # 2. Pipeline long messages to overlap with computation
    # 3. Use non-blocking operations where possible
    
    optimized_schedule = CommSchedule(
        grouped_sends = group_small_messages(comm_graph, rank),
        pipelined_ops = plan_message_pipelining(comm_graph, rank),
        async_operations = identify_async_opportunities(comm_graph, rank)
    )
    
    return optimized_schedule
end

# Helper structures and functions

struct DistributedSparseMatrix{T}
    local_matrix::SparseMatrixCSC{T,Int}
    comm::MPI.Comm
    row_distribution::Vector{UnitRange{Int}}
    col_distribution::Vector{UnitRange{Int}}
end

struct CommSchedule
    grouped_sends::Vector{Vector{Int}}
    pipelined_ops::Vector{PipelineOp}
    async_operations::Vector{AsyncOp}
end

"""Helper functions for index management and communication"""

function get_global_spectral_index(cfg::SHTnsConfig, l::Int, m::Int)
    # Convert (l,m) to linear index in global spectral space
    idx = 1
    for (ll, mm) in cfg.lm_indices
        if ll == l && mm == m
            return idx
        end
        idx += 1
    end
    error("Invalid (l,m) = ($l,$m)")
end

function linear_index_from_lm(l_idx::Int, m_idx::Int, m_stride::Int)
    return (l_idx - 1) * m_stride + m_idx
end

# Memory-efficient array allocation for pencils
function allocate_array(pencil::Pencil, T::Type)
    return PencilArray{T}(undef, pencil)
end

"""
    benchmark_parallel_performance(pcfg::ParallelSHTConfig{T}, 
                                  problem_sizes::Vector{Int})

Benchmark parallel matrix operations across different problem sizes and 
process counts to guide optimal configuration choices.
"""
function benchmark_parallel_performance(pcfg::ParallelSHTConfig{T}, 
                                       problem_sizes::Vector{Int}) where T
    
    rank = MPI.Comm_rank(pcfg.comm)
    nprocs = MPI.Comm_size(pcfg.comm)
    
    results = Dict{String, Vector{Float64}}()
    
    for lmax in problem_sizes
        # Update configuration for this problem size
        test_cfg = create_test_config(T, lmax)
        test_pcfg = ParallelSHTConfig{T}(test_cfg, pcfg.comm)
        
        # Benchmark different operations
        if rank == 0
            println("Testing lmax = $lmax with $nprocs processes")
        end
        
        # Benchmark cos(θ) operator
        qlm_test = create_test_spectral_field(test_pcfg)
        qlm_out = similar(qlm_test)
        
        MPI.Barrier(pcfg.comm)
        t_costheta = @elapsed begin
            for _ in 1:10
                parallel_apply_costheta_operator!(test_pcfg, qlm_test, qlm_out)
            end
            MPI.Barrier(pcfg.comm)
        end
        
        # Benchmark Laplacian (should be fastest)
        MPI.Barrier(pcfg.comm)
        t_laplacian = @elapsed begin
            for _ in 1:100
                parallel_apply_laplacian_distributed!(test_pcfg, qlm_test)
            end
            MPI.Barrier(pcfg.comm)
        end
        
        # Store results
        key = "lmax_$(lmax)_procs_$(nprocs)"
        results["costheta_$key"] = [t_costheta / 10]
        results["laplacian_$key"] = [t_laplacian / 100]
        
        if rank == 0
            println("  cos(θ) operator: $(round(t_costheta/10*1000, digits=2)) ms")
            println("  Laplacian:       $(round(t_laplacian/100*1000, digits=2)) ms")
        end
    end
    
    return results
end

# All exports handled by main module SHTnsKit.jl