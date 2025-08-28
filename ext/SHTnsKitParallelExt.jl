__precompile__(false)  # Disable precompilation due to MPI dependencies
module SHTnsKitParallelExt

"""
SHTnsKit Parallel Extension

This Julia package extension provides distributed/parallel spherical harmonic transform
capabilities using MPI for inter-process communication and PencilArrays for distributed
memory management. The extension is automatically loaded when both MPI.jl and 
PencilArrays.jl are available in the environment.

Key capabilities:
- Distributed spherical harmonic transforms across MPI processes
- Pencil decomposition for memory-distributed arrays (latitude/longitude/spectral)
- Parallel FFTs via PencilFFTs for longitude direction transforms
- Load balancing and data redistribution for optimal performance
- Caching of FFT plans for repeated operations (optional via environment variable)
"""

using MPI: Allreduce, Allreduce!, Allgather, Allgatherv, Comm_size, COMM_WORLD
using PencilArrays: Pencil, PencilArray  # Distributed array framework
using PencilFFTs                          # Distributed FFTs
using SHTnsKit                           # Core spherical harmonic functionality

# ===== FFT PLAN CACHING =====
# Optional plan caching to avoid repeated planning overhead in performance-critical code
# Enable via: ENV["SHTNSKIT_CACHE_PENCILFFTS"] = "1"
const _CACHE_PENCILFFTS = Ref{Bool}(get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "0") == "1")

# Cache storage for FFT plans indexed by array characteristics
const _pfft_cache = IdDict{Any,Any}()

# Generate cache key based on array characteristics for FFT plan reuse
_cache_key(kind::Symbol, A) = (kind, size(A,1), size(A,2), eltype(A), 
                               try Comm_size(communicator(A)) catch; 1 end,
                               try hash(A.pencil.decomposition) catch; 0 end)

function _get_or_plan(kind::Symbol, A)
    # If caching disabled, create plan directly without storing
    if !_CACHE_PENCILFFTS[]
        return kind === :fft  ? plan_fft(A; dims=2) :     # Forward FFT along longitude (dim 2)
               kind === :ifft ? plan_fft(A; dims=2) :     # Inverse FFT along longitude  
               kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :   # Real-to-complex FFT
               kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) : # Complex-to-real IFFT
               error("unknown plan kind")
    end
    
    # Check if we already have a cached plan for this array configuration
    key = _cache_key(kind, A)
    
    if haskey(_pfft_cache, key)
        return _pfft_cache[key]  # Return cached plan
    end

    # Create new plan and cache it for future use
    plan = kind === :fft  ? plan_fft(A; dims=2) :     # Forward FFT along longitude
           kind === :ifft ? plan_fft(A; dims=2) :     # Inverse FFT along longitude
           kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :   # Real-to-complex FFT
           kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) : # Complex-to-real IFFT
           error("unknown plan kind")
    _pfft_cache[key] = plan
    return plan
end


# ===== VERSION-AGNOSTIC COMPATIBILITY LAYER =====
# Handle API changes across different versions of PencilArrays/PencilFFTs
# These shims ensure compatibility with both older and newer package versions

# Check which API functions are available in the loaded PencilArrays version
const _has_pa_communicator = isdefined(PencilArrays, :communicator)
const _has_pa_allocate     = isdefined(PencilArrays, :allocate)
const _has_pa_globalidx    = isdefined(PencilArrays, :globalindices)

# Get MPI communicator from PencilArray (handles API name changes)
communicator(A) = _has_pa_communicator ? (PencilArrays.communicator)(A) : (
    isdefined(PencilArrays, :comm) ? (PencilArrays.comm)(A) : 
    hasfield(typeof(A), :pencil) && hasfield(typeof(A.pencil), :comm) ? A.pencil.comm :
    error("PencilArrays communicator not found"))

# Allocate PencilArray (handles API changes)  
function allocate(args...; kwargs...)
    if _has_pa_allocate 
        return (PencilArrays.allocate)(args...; kwargs...)
    elseif isdefined(PencilArrays, :PencilArray)
        # Fallback for older versions
        if length(args) >= 2 && isa(args[2], PencilArrays.Pencil)
            return PencilArrays.PencilArray(undef, args[1], args[2]; kwargs...)
        end
    end
    error("PencilArrays.allocate not found and no compatible fallback")
end

# Get global indices for a dimension (handles API name changes)
globalindices(A, dim) = _has_pa_globalidx ? (PencilArrays.globalindices)(A, dim) : (
    isdefined(PencilArrays, :global_indices) ? (PencilArrays.global_indices)(A, dim) :
    hasfield(typeof(A), :pencil) && hasfield(typeof(A.pencil), :axes) ? 
    A.pencil.axes[dim] : error("PencilArrays.globalindices not found"))

# ===== DISTRIBUTED FFT WRAPPERS =====
# Wrapper functions for PencilFFTs that work with distributed arrays
plan_fft(A; dims=:) = PencilFFTs.plan_fft(A; dims=dims)      # Plan forward FFT
fft(A, p) = PencilFFTs.fft(A, p)                            # Execute forward FFT
ifft(A, p) = PencilFFTs.ifft(A, p)                          # Execute inverse FFT
plan_rfft(A; dims=:) = PencilFFTs.plan_rfft(A; dims=dims)   # Plan real-to-complex FFT
plan_irfft(A; dims=:) = PencilFFTs.plan_irfft(A; dims=dims) # Plan complex-to-real IFFT
rfft(A, p) = PencilFFTs.rfft(A, p)                          # Execute real-to-complex FFT
irfft(A, p) = PencilFFTs.irfft(A, p)                        # Execute complex-to-real IFFT


# ===== PARALLEL EXTENSION MODULES =====
# Include specialized modules for different aspects of parallel spherical harmonic transforms
include("parallel_diagnostics.jl")      # Diagnostic and profiling tools for parallel operations
include("parallel_dispatch.jl")         # Function dispatch and interface definitions  
include("parallel_plans.jl")            # Distributed transform planning and setup
include("parallel_transforms.jl")       # Core parallel transform implementations
include("parallel_ops_pencil.jl")       # Parallel differential operators using PencilArrays
include("parallel_rotations_pencil.jl") # Parallel spherical rotation operations
include("parallel_local.jl")            # Local (per-process) operations and utilities

# Optimized communication patterns for large spectral arrays
function efficient_spectral_reduce!(local_data::AbstractMatrix, comm)
    # For large spectral arrays, use hierarchical reduction instead of flat Allreduce
    # This reduces communication overhead from O(P*N²) to O(log(P)*N²)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if nprocs <= 4 || length(local_data) < 10000
        # For small problems, use standard Allreduce
        MPI.Allreduce!(local_data, +, comm)
        return local_data
    end
    
    # Hierarchical reduction for large problems
    # Step 1: Local reduction within compute nodes (simulated here)
    temp_buf = similar(local_data)
    copyto!(temp_buf, local_data)
    
    # Step 2: Tree-based reduction across all processes
    step = 1
    while step < nprocs
        if rank % (2 * step) == 0 && rank + step < nprocs
            # Receive and accumulate from partner process
            MPI.Recv!(temp_buf, rank + step, 0, comm)
            local_data .+= temp_buf
        elseif (rank - step) % (2 * step) == 0 && rank >= step
            # Send to partner process
            MPI.Send(local_data, rank - step, 0, comm)
            break  # This process is done
        end
        step *= 2
    end
    
    # Step 3: Broadcast final result from root to all processes
    MPI.Bcast!(local_data, 0, comm)
    return local_data
end

function sparse_spectral_reduce!(local_data::AbstractVector, indices::Vector{Int}, comm)
    # For sparse spectral data, only communicate non-zero coefficients
    # This can significantly reduce communication volume
    nz_indices = findall(!iszero, local_data)
    
    if length(nz_indices) < length(local_data) * 0.1  # Less than 10% non-zero
        # Gather only non-zero entries
        local_nz_data = local_data[nz_indices]
        global_nz_data = MPI.Allreduce(local_nz_data, +, comm)
        
        # Reconstruct full array
        fill!(local_data, 0)
        local_data[nz_indices] = global_nz_data
        return local_data
    else
        # Use standard Allreduce for dense data
        MPI.Allreduce!(local_data, +, comm)
        return local_data
    end
end

# Note: Avoid forwarding Base.zeros(Pencil) to PencilArrays.zeros to prevent
# potential recursion when PencilArrays.zeros may call Base.zeros internally.
end # module SHTnsKitParallelExt
