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
_cache_key(kind::Symbol, A) = (kind, size(A,1), size(A,2), eltype(A), try Comm_size(communicator(A)) catch; 1 end)

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


# Local shims for commonly used PencilArrays/PencilFFTs functions (version-agnostic)
const _has_pa_communicator = isdefined(PencilArrays, :communicator)
const _has_pa_allocate     = isdefined(PencilArrays, :allocate)
const _has_pa_globalidx    = isdefined(PencilArrays, :globalindices)

communicator(A) = _has_pa_communicator ? (PencilArrays.communicator)(A) : (
    isdefined(PencilArrays, :comm) ? (PencilArrays.comm)(A) : error("PencilArrays communicator not found"))

allocate(args...; kwargs...) = _has_pa_allocate ? (PencilArrays.allocate)(args...; kwargs...) : error("PencilArrays.allocate not found")

globalindices(A, dim) = _has_pa_globalidx ? (PencilArrays.globalindices)(A, dim) : (
    isdefined(PencilArrays, :global_indices) ? (PencilArrays.global_indices)(A, dim) : error("PencilArrays.globalindices not found"))

# FFT helpers
plan_fft(A; dims=:) = PencilFFTs.plan_fft(A; dims=dims)
fft(A, p) = PencilFFTs.fft(A, p)
ifft(A, p) = PencilFFTs.ifft(A, p)
plan_rfft(A; dims=:) = PencilFFTs.plan_rfft(A; dims=dims)
plan_irfft(A; dims=:) = PencilFFTs.plan_irfft(A; dims=dims)
rfft(A, p) = PencilFFTs.rfft(A, p)
irfft(A, p) = PencilFFTs.irfft(A, p)


include("parallel_diagnostics.jl")
include("parallel_dispatch.jl")
include("parallel_plans.jl")
include("parallel_transforms.jl")
include("parallel_ops_pencil.jl")
include("parallel_rotations_pencil.jl")
include("parallel_local.jl")

# Convenience: forward Base.zeros for Pencil topologies to PencilArrays.zeros
# This helps users who call `zeros(P; eltype=...)` instead of `PencilArrays.zeros(P; ...)`.
import Base: zeros
zeros(P::Pencil; eltype=Float64) = PencilArrays.zeros(P; eltype=eltype)
end # module SHTnsKitParallelExt
