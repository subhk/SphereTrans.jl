module SHTnsKitParallelExt
__precompile__(false)

using MPI: Allreduce, Allreduce!, Allgather, Allgatherv, Comm_size, COMM_WORLD
using PencilArrays: Pencil, PencilArray
using PencilFFTs
using SHTnsKit

# Optional plan caching (opt-in via ENV SHTNSKIT_CACHE_PENCILFFTS=1)
const _CACHE_PENCILFFTS = Ref{Bool}(get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "0") == "1")

const _pfft_cache = IdDict{Any,Any}()

_cache_key(kind::Symbol, A) = (kind, size(A,1), size(A,2), eltype(A), try Comm_size(communicator(A)) catch; 1 end)

function _get_or_plan(kind::Symbol, A)
    
    if !_CACHE_PENCILFFTS[]
        return kind === :fft  ? plan_fft(A; dims=2) :
               kind === :ifft ? plan_fft(A; dims=2) :
               kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :
               kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) :
               error("unknown plan kind")
    end
    
    key = _cache_key(kind, A)
    
    if haskey(_pfft_cache, key)
        return _pfft_cache[key]
    end

    plan = kind === :fft  ? plan_fft(A; dims=2) :
           kind === :ifft ? plan_fft(A; dims=2) :
           kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :
           kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) :
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
end # module SHTnsKitParallelExt
