module SHTnsKitParallelExt

using MPI
using PencilArrays
using PencilFFTs
using ..SHTnsKit

# Optional plan caching (opt-in via ENV SHTNSKIT_CACHE_PENCILFFTS=1)
const _CACHE_PENCILFFTS = Ref{Bool}(get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "0") == "1")
const _pfft_cache = IdDict{Any,Any}()
_cache_key(kind::Symbol, A) = (kind, size(A,1), size(A,2), eltype(A), try MPI.Comm_size(PencilArrays.communicator(A)) catch; 1 end)
function _get_or_plan(kind::Symbol, A)
    if !_CACHE_PENCILFFTS[]
        return kind === :fft  ? PencilFFTs.plan_fft(A; dims=2) :
               kind === :ifft ? PencilFFTs.plan_fft(A; dims=2) :
               kind === :rfft ? (try PencilFFTs.plan_rfft(A; dims=2) catch; nothing end) :
               kind === :irfft? (try PencilFFTs.plan_irfft(A; dims=2) catch; nothing end) :
               error("unknown plan kind")
    end
    key = _cache_key(kind, A)
    if haskey(_pfft_cache, key)
        return _pfft_cache[key]
    end
    plan = kind === :fft  ? PencilFFTs.plan_fft(A; dims=2) :
           kind === :ifft ? PencilFFTs.plan_fft(A; dims=2) :
           kind === :rfft ? (try PencilFFTs.plan_rfft(A; dims=2) catch; nothing end) :
           kind === :irfft? (try PencilFFTs.plan_irfft(A; dims=2) catch; nothing end) :
           error("unknown plan kind")
    _pfft_cache[key] = plan
    return plan
end


include("parallel_diagnostics.jl")

include("parallel_dispatch.jl")
include("parallel_plans.jl")
include("parallel_transforms.jl")
