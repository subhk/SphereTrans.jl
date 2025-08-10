"""
High-level convenience wrappers around the low-level SHTns C API.

These helpers provide allocation, shape-safe interfaces, and GPU-friendly
variants that operate on device arrays by staging through host memory.
"""

# --- Optional native GPU entrypoint detection (runtime) ---
const _gpu_sh2spat_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _gpu_spat2sh_ptr = Ref{Ptr{Cvoid}}(C_NULL)

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
                _gpu_sh2spat_ptr[] = sym
                found = true
            end
        end
        if spat2sh !== nothing
            if (sym = Libdl.dlsym_e(handle, spat2sh)) !== C_NULL
                _gpu_spat2sh_ptr[] = sym
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
is_native_gpu_enabled() = (_gpu_sh2spat_ptr[] != C_NULL) || (_gpu_spat2sh_ptr[] != C_NULL)

# Try to enable from ENV at load time (no error if absent)
try
    enable_native_gpu!()
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
    if _gpu_sh2spat_ptr[] != C_NULL
        ccall(_gpu_sh2spat_ptr[], Cvoid,
              (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, sh),
              Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)))
    else
        sh_to_spat(cfg, sh, reshape(spat, :))
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
    if _gpu_spat2sh_ptr[] != C_NULL
        ccall(_gpu_spat2sh_ptr[], Cvoid,
              (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)),
              Base.unsafe_convert(Ptr{Float64}, sh))
    else
        spat_to_sh(cfg, reshape(spat, :), sh)
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
