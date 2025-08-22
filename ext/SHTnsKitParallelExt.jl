module SHTnsKitParallelExt

using SHTnsKit
using MPI
using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays

# Re-export the parallel functionality
struct ParallelSHTConfig{T<:AbstractFloat}
    # Base SHT configuration
    base_cfg::SHTnsKit.SHTnsConfig{T}
    
    # MPI and pencil configuration
    comm::MPI.Comm
    pencil_decomp::PencilArrays.PencilDecomposition
    
    # Distributed arrays for spectral coefficients
    spectral_pencil::PencilArrays.Pencil{2}  # (l, m) decomposition
    spatial_pencil::PencilArrays.Pencil{2}   # (θ, φ) decomposition
    
    # FFT transforms between pencils
    fft_plan::PencilFFTs.PencilFFTPlan
    
    # Communication patterns
    boundary_exchange_pattern::Vector{Int}
    halo_width::Int
    
    # Performance options
    use_async_comm::Bool
    overlap_compute_comm::Bool
    memory_limit::Int  # MB
end

# Basic parallel configuration creation
function create_parallel_config(cfg::SHTnsKit.SHTnsConfig{T}, comm::MPI.Comm;
                               dims::Tuple{Int,Int} = (0, 0),
                               use_async_comm::Bool = true,
                               memory_limit::Int = 1024) where T
    
    # Create pencil decomposition for spectral coefficients
    spectral_size = (cfg.lmax + 1, cfg.mmax + 1)
    pencil_decomp = PencilArrays.PencilDecomposition(spectral_size, comm; dims=dims)
    
    # Create pencils for different data layouts
    spectral_pencil = PencilArrays.Pencil(pencil_decomp, spectral_size)
    
    # Create spatial pencil for grid points
    spatial_size = (cfg.nlat, cfg.nphi)
    spatial_pencil = PencilArrays.Pencil(pencil_decomp, spatial_size)
    
    # Create FFT plan between pencils
    fft_plan = PencilFFTs.PencilFFTPlan(spatial_pencil, PencilFFTs.Transforms.FFT())
    
    # Initialize communication pattern
    boundary_exchange_pattern = collect(0:MPI.Comm_size(comm)-1)
    halo_width = 2
    
    return ParallelSHTConfig{T}(
        cfg, comm, pencil_decomp, spectral_pencil, spatial_pencil,
        fft_plan, boundary_exchange_pattern, halo_width,
        use_async_comm, true, memory_limit
    )
end

# Stub implementations for parallel operations
function parallel_apply_operator(pcfg::ParallelSHTConfig{T}, op::Symbol,
                                qlm_in::AbstractVector{Complex{T}},
                                qlm_out::AbstractVector{Complex{T}}) where T
    error("Parallel operations require MPI, PencilArrays, and PencilFFTs packages to be loaded")
end

function auto_parallel_config(cfg::SHTnsKit.SHTnsConfig{T}) where T
    if !MPI.Initialized()
        error("MPI must be initialized for parallel operations")
    end
    comm = MPI.COMM_WORLD
    return create_parallel_config(cfg, comm)
end

function optimal_process_count(cfg::SHTnsKit.SHTnsConfig{T}) where T
    # Simple heuristic based on problem size
    total_size = cfg.nlm + cfg.nlat * cfg.nphi
    return min(8, max(1, total_size ÷ 10000))
end

end # module