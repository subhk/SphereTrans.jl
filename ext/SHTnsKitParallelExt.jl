module SHTnsKitParallelExt

using SHTnsKit
using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays

# Override the main module's parallel configuration
struct ParallelSHTConfig{T<:AbstractFloat}
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
    
    # Communication buffers
    send_buffers::Dict{Int, Vector{Complex{T}}}
    recv_buffers::Dict{Int, Vector{Complex{T}}}
    
    # Performance options
    use_async_comm::Bool
    overlap_compute_comm::Bool
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
    
    # This is a simplified parallel synthesis
    # In practice, this would involve:
    # 1. Distribute spectral coefficients across processes
    # 2. Perform local Legendre transforms
    # 3. Redistribute for FFT transforms
    # 4. Perform distributed FFTs using PencilFFTs
    # 5. Gather results
    
    cfg = pcfg.base_cfg
    
    # For now, fall back to serial on each process for local data
    local_spatial = zeros(T, size(spatial_data))
    SHTnsKit.sh_to_spat!(cfg, sh_coeffs, local_spatial)
    
    # Combine results using MPI reduction
    MPI.Allreduce!(local_spatial, +, pcfg.comm)
    spatial_data .= local_spatial
    
    return spatial_data
end

function _parallel_analysis!(pcfg::ParallelSHTConfig{T},
                            spatial_data::AbstractMatrix{T},
                            sh_coeffs::AbstractVector{Complex{T}}) where T
    
    cfg = pcfg.base_cfg
    
    # Simplified parallel analysis
    local_coeffs = zeros(Complex{T}, length(sh_coeffs))
    SHTnsKit.spat_to_sh!(cfg, spatial_data, local_coeffs)
    
    # Combine results
    MPI.Allreduce!(local_coeffs, +, pcfg.comm)
    sh_coeffs .= local_coeffs
    
    return sh_coeffs
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