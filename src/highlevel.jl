"""
High-level convenience wrappers around the low-level SHTns C API.

These helpers provide allocation, shape-safe interfaces, and GPU-friendly
variants that operate on device arrays by staging through host memory.
They also add safe multi-threading by serializing concurrent transforms that
share the same SHTns configuration.
"""

import Libdl
using Base.Threads: Atomic, atomic_load, atomic_store!

# --- Optional native GPU entrypoint detection (runtime) ---
const _gpu_sh2spat_ptr = Atomic{Ptr{Cvoid}}(C_NULL)
const _gpu_spat2sh_ptr = Atomic{Ptr{Cvoid}}(C_NULL)

# --- Optional native vector transform entrypoints ---
const _vec_torpol2uv_ptr = Atomic{Ptr{Cvoid}}(C_NULL)
const _vec_uv2torpol_ptr = Atomic{Ptr{Cvoid}}(C_NULL)

"""
    enable_native_vec!(; torpol2uv=nothing, uv2torpol=nothing) -> Bool

Resolve and cache SHTns vector transform entrypoints by name. If not provided,
tries environment variables `SHTNSKIT_VEC_TORPOL2UV` and `SHTNSKIT_VEC_UV2TORPOL`.
Assumes signatures:
  torpol2uv: (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}) -> Cvoid
  uv2torpol: (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}) -> Cvoid
where inputs are (cfg, tor, pol, u, v) in double precision.
"""
function enable_native_vec!(; torpol2uv::Union{Nothing,String}=get(ENV, "SHTNSKIT_VEC_TORPOL2UV", nothing),
                               uv2torpol::Union{Nothing,String}=get(ENV, "SHTNSKIT_VEC_UV2TORPOL", nothing))
    found = false
    try
        handle = Libdl.dlopen(libshtns)
        if torpol2uv !== nothing
            if (sym = Libdl.dlsym_e(handle, torpol2uv)) !== C_NULL
                atomic_store!(_vec_torpol2uv_ptr, sym); found = true
            end
        end
        if uv2torpol !== nothing
            if (sym = Libdl.dlsym_e(handle, uv2torpol)) !== C_NULL
                atomic_store!(_vec_uv2torpol_ptr, sym); found = true
            end
        end
    catch
    end
    return found
end

# If still not enabled, try common default vector symbol names
try
    if !is_native_vec_enabled()
        enable_native_vec!(; torpol2uv="shtns_torpol2uv", uv2torpol="shtns_uv2torpol")
    end
catch
end

"""Return true if either vector transform entrypoint is enabled."""
is_native_vec_enabled() = (atomic_load(_vec_torpol2uv_ptr) != C_NULL) || (atomic_load(_vec_uv2torpol_ptr) != C_NULL)

"""Vector synthesis: (tor, pol) -> (u, v). Arrays must be preallocated."""
function synthesize_vec!(cfg::SHTnsConfig,
                         tor::AbstractVector{Float64}, pol::AbstractVector{Float64},
                         u::AbstractMatrix{Float64}, v::AbstractMatrix{Float64})
    @assert length(tor) == get_nlm(cfg) && length(pol) == get_nlm(cfg)
    @assert size(u) == (get_nlat(cfg), get_nphi(cfg)) && size(v) == size(u)
    ptr = atomic_load(_vec_torpol2uv_ptr)
    ptr == C_NULL && error("Vector synthesis entrypoint not enabled. Call enable_native_vec! or set env SHTNSKIT_VEC_TORPOL2UV.")
    ccall(ptr, Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, tor), Base.unsafe_convert(Ptr{Float64}, pol),
          Base.unsafe_convert(Ptr{Float64}, reshape(u, :)), Base.unsafe_convert(Ptr{Float64}, reshape(v, :)))
    return u, v
end

"""
    grid_latitudes(cfg; mode=:approx) -> Vector{Float64}

Return latitude coordinates (radians) for the configured grid. By default
computes an equiangular approximation. If you have native SHTns functions to
retrieve exact grid nodes, set environment variables:
- SHTNSKIT_GRID_GET_THETA: symbol taking (Ptr{Cvoid}, Ptr{Float64}) and filling
  `nlat` entries with latitude in radians.
"""
function grid_latitudes(cfg::SHTnsConfig; mode=:approx)
    nlat = get_nlat(cfg)
    lat = Vector{Float64}(undef, nlat)
    sym = try
        handle = Libdl.dlopen(libshtns)
        Libdl.dlsym_e(handle, get(ENV, "SHTNSKIT_GRID_GET_THETA", ""))
    catch
        C_NULL
    end
    if sym != C_NULL
        ccall(sym, Cvoid, (Ptr{Cvoid}, Ptr{Float64}), cfg.ptr, Base.unsafe_convert(Ptr{Float64}, lat))
        return lat
    end
    @inbounds for i in 1:nlat
        lat[i] = (i - 0.5) * π / nlat - π/2
    end
    return lat
end

"""Return longitude coordinates (radians) for the configured grid (equiangular)."""
function grid_longitudes(cfg::SHTnsConfig)
    nphi = get_nphi(cfg)
    lon = Vector{Float64}(undef, nphi)
    @inbounds for j in 1:nphi
        lon[j] = 2π * (j - 1) / nphi
    end
    return lon
end

"""Vector analysis: (u, v) -> (tor, pol). Arrays must be preallocated."""
function analyze_vec!(cfg::SHTnsConfig,
                      u::AbstractMatrix{Float64}, v::AbstractMatrix{Float64},
                      tor::AbstractVector{Float64}, pol::AbstractVector{Float64})
    @assert size(u) == (get_nlat(cfg), get_nphi(cfg)) && size(v) == size(u)
    @assert length(tor) == get_nlm(cfg) && length(pol) == get_nlm(cfg)
    ptr = atomic_load(_vec_uv2torpol_ptr)
    ptr == C_NULL && error("Vector analysis entrypoint not enabled. Call enable_native_vec! or set env SHTNSKIT_VEC_UV2TORPOL.")
    ccall(ptr, Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, reshape(u, :)), Base.unsafe_convert(Ptr{Float64}, reshape(v, :)),
          Base.unsafe_convert(Ptr{Float64}, tor), Base.unsafe_convert(Ptr{Float64}, pol))
    return tor, pol
end
# --- Per-config locks to ensure thread-safe calls on the same cfg ---
const _cfg_locks = Dict{Ptr{Cvoid}, Base.ReentrantLock}()
const _cfg_locks_guard = Base.ReentrantLock()

_get_lock(cfg::SHTnsConfig) = begin
    lk = nothing
    Base.lock(_cfg_locks_guard) do
        lk = get!(_cfg_locks, cfg.ptr, Base.ReentrantLock())
    end
    return lk
end

# Called from free_config to cleanup lock map
function _on_free_config(cfg::SHTnsConfig)
    Base.lock(_cfg_locks_guard) do
        if haskey(_cfg_locks, cfg.ptr)
            delete!(_cfg_locks, cfg.ptr)
        end
    end
    return nothing
end

"""
    enable_native_gpu!(; sh2spat=nothing, spat2sh=nothing) -> Bool

Attempt to enable native GPU-accelerated SHTns entrypoints by looking up
provided symbol names in the loaded `libshtns` shared library. If none are
provided, uses environment variables `SHTNSKIT_GPU_SH2SPAT` and
`SHTNSKIT_GPU_SPAT2SH` when set. Returns true if at least one symbol is found.

Notes:
- This expects the GPU entrypoints to be drop-in replacements with the same
  signature as the CPU variants (`(Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})`).
- If not available, the library falls back to CPU entrypoints.
"""
function enable_native_gpu!(; sh2spat::Union{Nothing,String}=get(ENV, "SHTNSKIT_GPU_SH2SPAT", nothing),
                               spat2sh::Union{Nothing,String}=get(ENV, "SHTNSKIT_GPU_SPAT2SH", nothing))
    found = false
    try
        handle = Libdl.dlopen(libshtns)
        if sh2spat !== nothing
            if (sym = Libdl.dlsym_e(handle, sh2spat)) !== C_NULL
                atomic_store!(_gpu_sh2spat_ptr, sym)
                found = true
            end
        end
        if spat2sh !== nothing
            if (sym = Libdl.dlsym_e(handle, spat2sh)) !== C_NULL
                atomic_store!(_gpu_spat2sh_ptr, sym)
                found = true
            end
        end
    catch
        # ignore and keep fallbacks
    end
    return found
end

"""
    is_native_gpu_enabled() -> Bool

True if a native GPU entrypoint for at least one transform direction is active.
"""
is_native_gpu_enabled() = (atomic_load(_gpu_sh2spat_ptr) != C_NULL) || (atomic_load(_gpu_spat2sh_ptr) != C_NULL)

# Try to enable from ENV at load time (no error if absent)
try
    enable_native_gpu!()
catch
end
try
    if !is_native_gpu_enabled()
        enable_native_gpu!(; sh2spat="shtns_sh_to_spat_gpu", spat2sh="shtns_spat_to_sh_gpu")
    end
catch
end

"""
    allocate_spectral(cfg; T=Float64) -> Vector{T}

Allocate a spectral coefficient vector of length `get_nlm(cfg)`.
"""
function allocate_spectral(cfg::SHTnsConfig; T::Type{<:Real}=Float64)
    return Vector{T}(undef, get_nlm(cfg))
end

"""
    allocate_spatial(cfg; T=Float64) -> Matrix{T}

Allocate a spatial grid matrix of size `(get_nlat(cfg), get_nphi(cfg))`.
"""
function allocate_spatial(cfg::SHTnsConfig; T::Type{<:Real}=Float64)
    nlat = get_nlat(cfg)
    nphi = get_nphi(cfg)
    return Matrix{T}(undef, nlat, nphi)
end

"""
    synthesize!(cfg, sh, spat)

In-place spectral-to-spatial transform. `sh` must have length `get_nlm(cfg)` and
`spat` shape `(get_nlat(cfg), get_nphi(cfg))`. Operates on Float64 data.
"""
function synthesize!(cfg::SHTnsConfig,
                     sh::AbstractVector{Float64},
                     spat::AbstractMatrix{Float64})
    @assert length(sh) == get_nlm(cfg) "length(sh) must be get_nlm(cfg)"
    @assert size(spat, 1) == get_nlat(cfg) && size(spat, 2) == get_nphi(cfg) "spat has wrong size"
    # Low-level expects linear memory; pass a vector view of spat
    lk = _get_lock(cfg)
    Base.lock(lk)
    try
        ptr = atomic_load(_gpu_sh2spat_ptr)
        if ptr != C_NULL
            ccall(ptr, Cvoid,
                  (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
                  Base.unsafe_convert(Ptr{Float64}, sh),
                  Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)))
        else
            sh_to_spat(cfg, sh, reshape(spat, :))
        end
    finally
        Base.unlock(lk)
    end
    return spat
end

"""
    analyze!(cfg, spat, sh)

In-place spatial-to-spectral transform. `spat` shape `(get_nlat(cfg), get_nphi(cfg))` and
`sh` must have length `get_nlm(cfg)`. Operates on Float64 data.
"""
function analyze!(cfg::SHTnsConfig,
                  spat::AbstractMatrix{Float64},
                  sh::AbstractVector{Float64})
    @assert size(spat, 1) == get_nlat(cfg) && size(spat, 2) == get_nphi(cfg) "spat has wrong size"
    @assert length(sh) == get_nlm(cfg) "length(sh) must be get_nlm(cfg)"
    # Low-level expects linear memory; pass a vector view of spat
    lk = _get_lock(cfg)
    Base.lock(lk)
    try
        ptr = atomic_load(_gpu_spat2sh_ptr)
        if ptr != C_NULL
            ccall(ptr, Cvoid,
                  (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
                  Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)),
                  Base.unsafe_convert(Ptr{Float64}, sh))
        else
            spat_to_sh(cfg, reshape(spat, :), sh)
        end
    finally
        Base.unlock(lk)
    end
    return sh
end

"""
    synthesize(cfg, sh) -> Matrix{Float64}

Allocate output and perform spectral-to-spatial transform.
"""
function synthesize(cfg::SHTnsConfig, sh::AbstractVector{<:Real})
    # Promote to Float64 as required by SHTns API
    sh64 = sh isa AbstractVector{Float64} ? sh : Float64.(sh)
    spat = allocate_spatial(cfg; T=Float64)
    return synthesize!(cfg, sh64, spat)
end

"""
    analyze(cfg, spat) -> Vector{Float64}

Allocate output and perform spatial-to-spectral transform.
"""
function analyze(cfg::SHTnsConfig, spat::AbstractMatrix{<:Real})
    # Promote to Float64 as required by SHTns API
    spat64 = spat isa AbstractMatrix{Float64} ? spat : Float64.(spat)
    sh = allocate_spectral(cfg; T=Float64)
    return analyze!(cfg, spat64, sh)
end

"""
    synthesize_gpu(cfg, sh_dev) -> spat_dev

GPU-friendly spectral-to-spatial: accepts a device vector `sh_dev` and returns
an output device matrix. Data transfer occurs via host staging; computation
executes on CPU SHTns routines.
"""
function synthesize_gpu(cfg::SHTnsConfig, sh_dev::AbstractVector{<:Real})
    # Stage input to host, ensure Float64
    sh_host_any = Array(sh_dev)
    sh_host = sh_host_any isa Vector{Float64} ? sh_host_any : Float64.(sh_host_any)
    # Compute on host
    spat_host = synthesize(cfg, sh_host)
    # Allocate device output with same array type as input
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    spat_dev = similar(sh_dev, Float64, nlat, nphi)
    copyto!(spat_dev, spat_host)
    return spat_dev
end

"""
    analyze_gpu(cfg, spat_dev) -> sh_dev

GPU-friendly spatial-to-spectral: accepts a device matrix `spat_dev` and returns
an output device vector. Data transfer occurs via host staging; computation
executes on CPU SHTns routines.
"""
function analyze_gpu(cfg::SHTnsConfig, spat_dev::AbstractMatrix{<:Real})
    # Stage input to host, ensure Float64
    spat_host_any = Array(spat_dev)
    spat_host = spat_host_any isa Matrix{Float64} ? spat_host_any : Float64.(spat_host_any)
    # Compute on host
    sh_host = analyze(cfg, spat_host)
    # Allocate device output with same array type as input
    nlm = get_nlm(cfg)
    sh_dev = similar(spat_dev, Float64, nlm)
    copyto!(sh_dev, sh_host)
    return sh_dev
end
