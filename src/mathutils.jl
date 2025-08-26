"""
Lightweight math helpers local to SHTnsKit to avoid extra dependencies.

Provides cached `logfactorial` and `loggamma` for positive integers, which are
used in stable coefficient calculations (e.g., Wigner d and normalization).
"""

const _logfac_cache = Ref(Vector{Float64}([0.0]))  # cache[k+1] = log(k!) with log(0!) = 0

function _ensure_logfac!(n::Int)
    n >= 0 || throw(DomainError(n, "logfactorial expects n ≥ 0"))
    cache = _logfac_cache[]
    kmax = length(cache) - 1
    if n > kmax
        # extend cache incrementally: log((k)!) = log((k-1)!) + log(k)
        for k in (kmax + 1):n
            push!(cache, cache[end] + log(k))
        end
    end
    return nothing
end

"""
    logfactorial(n::Integer) -> Float64

Return log(n!) with a cached, exact summation (in Float64). Valid for n ≥ 0.
"""
function logfactorial(n::Integer)
    ni = Int(n)
    ni >= 0 || throw(DomainError(n, "logfactorial expects n ≥ 0"))
    _ensure_logfac!(ni)
    return _logfac_cache[][ni + 1]
end

"""
    loggamma(n::Integer) -> Float64

Return log(Γ(n)) for positive integers n using the identity Γ(n) = (n-1)!.
"""
function loggamma(n::Integer)
    ni = Int(n)
    ni >= 1 || throw(DomainError(n, "loggamma expects n ≥ 1 for Integer inputs"))
    return logfactorial(ni - 1)
end

"""
    loggamma(x::Real)

Fallback for non-integer reals. For now this throws to avoid silently using a
poor approximation. If needed, depend on SpecialFunctions.loggamma externally.
"""
function loggamma(x::Real)
    isinteger(x) && return loggamma(Int(round(x)))
    throw(ArgumentError("loggamma(::Real) is only defined for integers here; add SpecialFunctions for general inputs"))
end

