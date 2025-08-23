module SHTnsKitParallelExt

using SHTnsKit
using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays

# Override the main module's parallel configuration
mutable struct ParallelSHTConfig{T<:AbstractFloat}
    # Base SHT configuration
    base_cfg::SHTnsKit.SHTnsConfig{T}
    
    # MPI and pencil configuration
    comm::MPI.Comm
    rank::Int
    size::Int
    
    # Pencil decomposition for spectral and spatial domains
    spectral_decomp::PencilArrays.PencilDecomposition
    spatial_decomp::PencilArrays.PencilDecomposition
    
    # Pencils for different data layouts
    spectral_pencil::PencilArrays.Pencil{2}  # (l, m) decomposition
    spatial_pencil::PencilArrays.Pencil{2}   # (θ, φ) decomposition
    
    # FFT plans for transforms
    fft_plan::PencilFFTs.PencilFFTPlan
    ifft_plan::PencilFFTs.PencilFFTPlan
    
    # Local data ranges
    local_l_range::UnitRange{Int}
    local_m_range::UnitRange{Int}
    local_theta_range::UnitRange{Int}
    local_phi_range::UnitRange{Int}
    
    # Local spectral coefficient indices for load balancing
    local_spectral_indices::UnitRange{Int}
    
    # Communication buffers
    send_buffers::Dict{Int, Vector{Complex{T}}}
    recv_buffers::Dict{Int, Vector{Complex{T}}}
    
    # Performance options
    use_async_comm::Bool
    overlap_compute_comm::Bool
    
    # Constructor
    function ParallelSHTConfig{T}(base_cfg, comm, rank, size,
                                 spectral_decomp, spatial_decomp,
                                 spectral_pencil, spatial_pencil,
                                 fft_plan, ifft_plan,
                                 local_l_range, local_m_range, 
                                 local_theta_range, local_phi_range,
                                 send_buffers, recv_buffers,
                                 use_async_comm, overlap_compute_comm) where T
        new{T}(base_cfg, comm, rank, size,
               spectral_decomp, spatial_decomp,
               spectral_pencil, spatial_pencil,
               fft_plan, ifft_plan,
               local_l_range, local_m_range, local_theta_range, local_phi_range,
               1:0,  # Initialize empty range for local_spectral_indices
               send_buffers, recv_buffers,
               use_async_comm, overlap_compute_comm)
    end
end

# Implementation of create_parallel_config
function SHTnsKit.create_parallel_config(cfg::SHTnsKit.SHTnsConfig{T}, 
                                         comm::MPI.Comm = MPI.COMM_WORLD;
                                         dims::Tuple{Int,Int} = (0, 0),
                                         use_async_comm::Bool = true) where T
    
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # Optimize 2D decomposition if not specified
    if dims == (0, 0)
        dims = _optimal_2d_dims(size)
    end
    
    # Create decomposition for spectral domain (l, m)
    spectral_size = (cfg.lmax + 1, 2 * cfg.mmax + 1)  # Include negative m
    spectral_decomp = PencilArrays.PencilDecomposition(spectral_size, comm; dims=dims)
    spectral_pencil = PencilArrays.Pencil(spectral_decomp, spectral_size, (1, 2))
    
    # Create decomposition for spatial domain (θ, φ)
    spatial_size = (cfg.nlat, cfg.nphi)
    spatial_decomp = PencilArrays.PencilDecomposition(spatial_size, comm; dims=dims)
    spatial_pencil = PencilArrays.Pencil(spatial_decomp, spatial_size, (1, 2))
    
    # Create FFT plans for azimuthal transforms
    fft_plan = PencilFFTs.PencilFFTPlan(spatial_pencil, PencilFFTs.Transforms.FFT(), 2)
    ifft_plan = PencilFFTs.PencilFFTPlan(spatial_pencil, PencilFFTs.Transforms.IFFT(), 2)
    
    # Determine local ranges
    local_l_range = _get_local_range(spectral_pencil, 1)
    local_m_range = _get_local_range(spectral_pencil, 2)
    local_theta_range = _get_local_range(spatial_pencil, 1)
    local_phi_range = _get_local_range(spatial_pencil, 2)
    
    # Initialize communication buffers
    send_buffers = Dict{Int, Vector{Complex{T}}}()
    recv_buffers = Dict{Int, Vector{Complex{T}}}()
    
    for neighbor in 0:(size-1)
        if neighbor != rank
            buffer_size = min(cfg.nlm ÷ size, 1024)  # Reasonable buffer size
            send_buffers[neighbor] = Vector{Complex{T}}(undef, buffer_size)
            recv_buffers[neighbor] = Vector{Complex{T}}(undef, buffer_size)
        end
    end
    
    return ParallelSHTConfig{T}(
        cfg, comm, rank, size,
        spectral_decomp, spatial_decomp,
        spectral_pencil, spatial_pencil,
        fft_plan, ifft_plan,
        local_l_range, local_m_range, local_theta_range, local_phi_range,
        send_buffers, recv_buffers,
        use_async_comm, true
    )
end

# Implementation of parallel_apply_operator
function SHTnsKit.parallel_apply_operator(pcfg::ParallelSHTConfig{T}, op::Symbol,
                                         qlm_in::AbstractVector{Complex{T}},
                                         qlm_out::AbstractVector{Complex{T}}) where T
    
    if op === :laplacian
        return _parallel_apply_laplacian!(pcfg, qlm_in, qlm_out)
    elseif op === :costheta
        return _parallel_apply_costheta!(pcfg, qlm_in, qlm_out)
    elseif op === :sintdtheta
        return _parallel_apply_sintdtheta!(pcfg, qlm_in, qlm_out)
    else
        throw(ArgumentError("Unknown parallel operator: $op"))
    end
end

# Implementation of auto_parallel_config
function SHTnsKit.auto_parallel_config(cfg::SHTnsKit.SHTnsConfig{T}) where T
    if !MPI.Initialized()
        throw(ArgumentError("MPI must be initialized for parallel operations"))
    end
    
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    
    # Only use parallel if we have multiple processes and problem is large enough
    if size > 1 && cfg.nlm > 1000
        return create_parallel_config(cfg, comm)
    else
        # Return a "serial" parallel config that just wraps the original
        return _create_serial_parallel_config(cfg)
    end
end

# Parallel Laplacian operator
function _parallel_apply_laplacian!(pcfg::ParallelSHTConfig{T}, 
                                   qlm_in::AbstractVector{Complex{T}},
                                   qlm_out::AbstractVector{Complex{T}}) where T
    
    cfg = pcfg.base_cfg
    
    # Laplacian is diagonal, so no communication needed
    Threads.@threads for idx in eachindex(qlm_in)
        l, m = SHTnsKit.lm_from_index(cfg, idx)
        eigenvalue = -T(l * (l + 1))
        qlm_out[idx] = eigenvalue * qlm_in[idx]
    end
    
    return qlm_out
end

# Parallel cos(θ) operator (requires communication)
function _parallel_apply_costheta!(pcfg::ParallelSHTConfig{T},
                                  qlm_in::AbstractVector{Complex{T}},
                                  qlm_out::AbstractVector{Complex{T}}) where T
    
    cfg = pcfg.base_cfg
    fill!(qlm_out, zero(Complex{T}))
    
    # Local computation within each process
    for idx_out in eachindex(qlm_out)
        l_out, m_out = SHTnsKit.lm_from_index(cfg, idx_out)
        
        # cos(θ) couples (l,m) with (l±1,m)
        for Δl in [-1, 1]
            l_in = l_out + Δl
            if 0 <= l_in <= cfg.lmax && abs(m_out) <= min(l_in, cfg.mmax)
                idx_in = SHTnsKit.lmidx(cfg, l_in, m_out)
                if idx_in <= length(qlm_in)
                    coupling = _costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                    qlm_out[idx_out] += coupling * qlm_in[idx_in]
                end
            end
        end
    end
    
    # MPI reduction to combine results from all processes
    MPI.Allreduce!(qlm_out, +, pcfg.comm)
    
    return qlm_out
end

# Parallel sin(θ)d/dθ operator
function _parallel_apply_sintdtheta!(pcfg::ParallelSHTConfig{T},
                                    qlm_in::AbstractVector{Complex{T}},
                                    qlm_out::AbstractVector{Complex{T}}) where T
    
    cfg = pcfg.base_cfg
    fill!(qlm_out, zero(Complex{T}))
    
    # Similar to cos(θ) but with different coupling coefficients
    for idx_out in eachindex(qlm_out)
        l_out, m_out = SHTnsKit.lm_from_index(cfg, idx_out)
        
        for Δl in [-1, 1]
            l_in = l_out + Δl
            if 0 <= l_in <= cfg.lmax && abs(m_out) <= min(l_in, cfg.mmax)
                idx_in = SHTnsKit.lmidx(cfg, l_in, m_out)
                if idx_in <= length(qlm_in)
                    coupling = _sintdtheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                    qlm_out[idx_out] += coupling * qlm_in[idx_in]
                end
            end
        end
    end
    
    MPI.Allreduce!(qlm_out, +, pcfg.comm)
    
    return qlm_out
end

# Utility functions
function _optimal_2d_dims(nprocs::Int)
    # Find factors that are as close as possible
    factors = []
    for i in 1:Int(sqrt(nprocs))
        if nprocs % i == 0
            push!(factors, (i, nprocs ÷ i))
        end
    end
    
    # Choose the pair with minimal difference
    if isempty(factors)
        return (1, nprocs)
    end
    
    best_pair = factors[1]
    min_diff = abs(best_pair[1] - best_pair[2])
    
    for pair in factors
        diff = abs(pair[1] - pair[2])
        if diff < min_diff
            min_diff = diff
            best_pair = pair
        end
    end
    
    return best_pair
end

function _get_local_range(pencil::PencilArrays.Pencil, dim::Int)
    local_size = size(pencil.data, dim)
    global_size = pencil.pencil_info.size_global[dim]
    offset = pencil.pencil_info.range_local[dim][1] - 1
    
    return (offset + 1):(offset + local_size)
end

function _costheta_coupling_coefficient(cfg::SHTnsKit.SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T
    # Simplified coupling coefficient for cos(θ) operator
    # This would need proper normalization based on cfg.norm
    if l_out == l_in - 1
        return T(sqrt((l_in^2 - m^2) / (4*l_in^2 - 1)))
    elseif l_out == l_in + 1
        return T(sqrt(((l_in+1)^2 - m^2) / (4*(l_in+1)^2 - 1)))
    else
        return T(0)
    end
end

function _sintdtheta_coupling_coefficient(cfg::SHTnsKit.SHTnsConfig{T}, l_out::Int, l_in::Int, m::Int) where T
    # Simplified coupling coefficient for sin(θ)d/dθ operator
    if l_out == l_in - 1
        return T(sqrt((l_in^2 - m^2) / (4*l_in^2 - 1))) * T(l_in)
    elseif l_out == l_in + 1
        return -T(sqrt(((l_in+1)^2 - m^2) / (4*(l_in+1)^2 - 1))) * T(l_in + 1)
    else
        return T(0)
    end
end

function _create_serial_parallel_config(cfg::SHTnsKit.SHTnsConfig{T}) where T
    # Create a minimal parallel config for serial execution
    comm = MPI.COMM_SELF
    rank = 0
    size = 1
    
    # Create dummy decompositions
    spectral_size = (cfg.lmax + 1, 2 * cfg.mmax + 1)
    spatial_size = (cfg.nlat, cfg.nphi)
    
    spectral_decomp = PencilArrays.PencilDecomposition(spectral_size, comm)
    spatial_decomp = PencilArrays.PencilDecomposition(spatial_size, comm)
    
    spectral_pencil = PencilArrays.Pencil(spectral_decomp, spectral_size)
    spatial_pencil = PencilArrays.Pencil(spatial_decomp, spatial_size)
    
    fft_plan = PencilFFTs.PencilFFTPlan(spatial_pencil, PencilFFTs.Transforms.FFT(), 2)
    ifft_plan = PencilFFTs.PencilFFTPlan(spatial_pencil, PencilFFTs.Transforms.IFFT(), 2)
    
    return ParallelSHTConfig{T}(
        cfg, comm, rank, size,
        spectral_decomp, spatial_decomp,
        spectral_pencil, spatial_pencil,
        fft_plan, ifft_plan,
        1:(cfg.lmax+1), 1:(2*cfg.mmax+1), 1:cfg.nlat, 1:cfg.nphi,
        Dict{Int, Vector{Complex{T}}}(), Dict{Int, Vector{Complex{T}}}(),
        false, false
    )
end

# Implementation of memory_efficient_parallel_transform!
function SHTnsKit.memory_efficient_parallel_transform!(pcfg::ParallelSHTConfig{T}, 
                                                      operation::Symbol,
                                                      input_data::AbstractArray{T},
                                                      output_data::AbstractArray{T}) where T
    
    if operation === :synthesis
        return _parallel_synthesis!(pcfg, input_data, output_data)
    elseif operation === :analysis
        return _parallel_analysis!(pcfg, input_data, output_data)
    else
        throw(ArgumentError("Unknown operation: $operation"))
    end
end

function _parallel_synthesis!(pcfg::ParallelSHTConfig{T}, 
                             sh_coeffs::AbstractVector{Complex{T}},
                             spatial_data::AbstractMatrix{T}) where T
    
    cfg = pcfg.base_cfg
    rank, size = pcfg.rank, pcfg.size
    
    # Step 1: Distribute spectral coefficients across processes
    # Each process gets a subset of (l,m) modes
    local_coeffs = _distribute_spectral_coeffs(pcfg, sh_coeffs)
    
    # Step 2: Perform local Legendre transforms for owned modes
    nlat, nphi = cfg.nlat, cfg.nphi
    nphi_modes = nphi ÷ 2 + 1
    
    # Allocate local Fourier workspace
    local_fourier = zeros(Complex{T}, nlat, nphi_modes)
    
    # Process only locally owned spectral modes
    @inbounds for (local_idx, global_idx) in enumerate(pcfg.local_spectral_indices)
        if global_idx <= length(sh_coeffs)
            l, m = SHTnsKit.lm_from_index(cfg, global_idx)
            
            # Skip if m is outside our local range
            m ∈ pcfg.local_m_range || continue
            
            m_col = m + 1
            if m_col <= nphi_modes
                coeff_val = local_coeffs[local_idx]
                
                # Vectorized Legendre synthesis for this mode
                @inbounds @simd for i in 1:nlat
                    plm_val = _compute_legendre_value(cfg, i, l, m)
                    local_fourier[i, m_col] += coeff_val * plm_val
                end
            end
        end
    end
    
    # Step 3: Reduce Fourier coefficients across processes
    # Use segmented reduction to minimize communication
    global_fourier = similar(local_fourier)
    _segmented_allreduce!(local_fourier, global_fourier, pcfg)
    
    # Step 4: Distributed FFT using PencilFFTs
    # Transform from Fourier modes to spatial domain
    spatial_complex = similar(global_fourier, nlat, nphi)
    PencilFFTs.mul!(spatial_complex, pcfg.ifft_plan, global_fourier)
    
    # Step 5: Extract real part and apply scaling
    @inbounds @simd for i in 1:(nlat * nphi)
        spatial_data[i] = real(spatial_complex[i]) * nphi
    end
    
    return spatial_data
end

function _parallel_analysis!(pcfg::ParallelSHTConfig{T},
                            spatial_data::AbstractMatrix{T},
                            sh_coeffs::AbstractVector{Complex{T}}) where T
    
    cfg = pcfg.base_cfg
    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Step 1: Distributed FFT from spatial to Fourier domain
    spatial_complex = Complex{T}.(spatial_data)
    fourier_coeffs = similar(spatial_complex, nlat, nphi ÷ 2 + 1)
    PencilFFTs.mul!(fourier_coeffs, pcfg.fft_plan, spatial_complex)
    
    # Step 2: Parallel Legendre analysis
    fill!(sh_coeffs, zero(Complex{T}))
    
    # Each process works on its local spectral modes
    @inbounds for (local_idx, global_idx) in enumerate(pcfg.local_spectral_indices)
        if global_idx <= length(sh_coeffs)
            l, m = SHTnsKit.lm_from_index(cfg, global_idx)
            
            # Skip if m is outside our range
            m ∈ pcfg.local_m_range || continue
            
            m_col = m + 1
            if m_col <= size(fourier_coeffs, 2)
                # Vectorized integration over latitudes
                integral = zero(Complex{T})
                @inbounds @simd for i in 1:nlat
                    plm_val = _compute_legendre_value(cfg, i, l, m)
                    weight = cfg.gauss_weights[i]
                    integral += fourier_coeffs[i, m_col] * plm_val * weight
                end
                
                # Apply normalization
                phi_normalization = T(2π) / nphi
                integral *= phi_normalization
                
                # Store local result
                sh_coeffs[global_idx] = m == 0 ? integral : integral * T(2)
            end
        end
    end
    
    # Step 3: Reduce coefficients across all processes
    MPI.Allreduce!(sh_coeffs, +, pcfg.comm)
    
    return sh_coeffs
end

# Helper functions for improved parallel implementation

"""
Distribute spectral coefficients across MPI processes with load balancing.
"""
function _distribute_spectral_coeffs(pcfg::ParallelSHTConfig{T}, 
                                   sh_coeffs::AbstractVector{Complex{T}}) where T
    
    # Determine local coefficient indices for this process
    coeffs_per_proc = length(sh_coeffs) ÷ pcfg.size
    remainder = length(sh_coeffs) % pcfg.size
    
    # Calculate start and end indices for this process
    if pcfg.rank < remainder
        local_count = coeffs_per_proc + 1
        local_start = pcfg.rank * local_count + 1
    else
        local_count = coeffs_per_proc
        local_start = pcfg.rank * coeffs_per_proc + remainder + 1
    end
    local_end = local_start + local_count - 1
    
    # Store local indices for later use
    pcfg.local_spectral_indices = local_start:local_end
    
    # Extract local coefficients
    local_coeffs = Vector{Complex{T}}(undef, local_count)
    @inbounds for (i, global_idx) in enumerate(local_start:local_end)
        if global_idx <= length(sh_coeffs)
            local_coeffs[i] = sh_coeffs[global_idx]
        else
            local_coeffs[i] = zero(Complex{T})
        end
    end
    
    return local_coeffs
end

"""
Segmented all-reduce for better communication efficiency.
Reduces communication volume by processing data in chunks.
"""
function _segmented_allreduce!(local_data::Matrix{Complex{T}}, 
                             global_data::Matrix{Complex{T}},
                             pcfg::ParallelSHTConfig{T}) where T
    
    # Process in segments to reduce memory pressure
    segment_size = min(1024, length(local_data) ÷ 4)  # Tunable parameter
    
    @inbounds for start_idx in 1:segment_size:length(local_data)
        end_idx = min(start_idx + segment_size - 1, length(local_data))
        
        # Create views for this segment
        local_segment = @view local_data[start_idx:end_idx]
        global_segment = @view global_data[start_idx:end_idx]
        
        # Perform reduction on segment
        MPI.Allreduce!(local_segment, global_segment, +, pcfg.comm)
    end
    
    return global_data
end

"""
Fast computation of Legendre polynomial values using cached recurrence.
"""
function _compute_legendre_value(cfg::SHTnsKit.SHTnsConfig{T}, 
                               lat_idx::Int, l::Int, m::Int) where T
    
    # Use precomputed cache if available
    if haskey(cfg.plm_cache, (lat_idx, l, m))
        return cfg.plm_cache[(lat_idx, l, m)]
    end
    
    # Fall back to direct computation
    theta = cfg.theta[lat_idx]
    costheta = cos(theta)
    sintheta = sin(theta)
    
    # Compute using recurrence relations (simplified)
    if l == m
        # P_m^m case
        pmm = T(1)
        for i in 1:m
            pmm *= -sintheta * sqrt((2*i + 1) / (2*i))
        end
        return pmm
    elseif l == m + 1
        # P_{m+1}^m case
        return sqrt(2*m + 3) * costheta * _compute_legendre_value(cfg, lat_idx, m, m)
    else
        # General recurrence
        alpha = sqrt((4*l^2 - 1) / (l^2 - m^2))
        beta = -sqrt(((l-1)^2 - m^2) / (4*(l-1)^2 - 1))
        
        plm_curr = _compute_legendre_value(cfg, lat_idx, l-1, m)
        plm_prev = _compute_legendre_value(cfg, lat_idx, l-2, m)
        
        return alpha * costheta * plm_curr + beta * plm_prev
    end
end

# Override other parallel functions
function SHTnsKit.optimal_process_count(cfg::SHTnsKit.SHTnsConfig{T}) where T
    # Better heuristic based on problem characteristics
    spectral_work = cfg.nlm
    spatial_work = cfg.nlat * cfg.nphi
    total_work = spectral_work + spatial_work
    
    # Estimate based on work per process
    work_per_process = 50000  # Tunable parameter
    optimal_procs = max(1, total_work ÷ work_per_process)
    
    # Don't exceed reasonable limits
    return min(optimal_procs, 64, cfg.lmax ÷ 2)
end

function SHTnsKit.parallel_performance_model(cfg::SHTnsKit.SHTnsConfig{T}, nprocs::Int) where T
    # Simple performance model
    serial_time = cfg.nlm * cfg.nlat * 1e-8  # Rough estimate in seconds
    
    # Account for communication overhead
    comm_overhead = nprocs > 1 ? 0.1 + 0.01 * log(nprocs) : 0.0
    parallel_efficiency = 1.0 / (1.0 + comm_overhead)
    
    parallel_time = serial_time / (nprocs * parallel_efficiency)
    
    return (
        serial_time = serial_time,
        parallel_time = parallel_time,
        speedup = serial_time / parallel_time,
        efficiency = parallel_efficiency,
        comm_overhead = comm_overhead
    )
end

end # module