module SHTnsKitParallelExt

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


# Local aliases for commonly used PencilArrays/PencilFFTs functions
const communicator = PencilArrays.communicator
const allocate = PencilArrays.allocate
const globalindices = PencilArrays.globalindices

const plan_fft = PencilFFTs.plan_fft
const fft = PencilFFTs.fft
const ifft = PencilFFTs.ifft
const plan_rfft = PencilFFTs.plan_rfft
const plan_irfft = PencilFFTs.plan_irfft
const rfft = PencilFFTs.rfft
const irfft = PencilFFTs.irfft


include("parallel_diagnostics.jl")
include("parallel_dispatch.jl")
include("parallel_plans.jl")
include("parallel_transforms.jl")
include("parallel_ops_pencil.jl")
include("parallel_rotations_pencil.jl")
include("parallel_local.jl")
end # module SHTnsKitParallelExt
