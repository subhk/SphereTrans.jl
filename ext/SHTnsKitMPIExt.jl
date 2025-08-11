module SHTnsKitMPIExt

using SHTnsKit
import Libdl
import MPI
import Base
"""
MPI-backed distributed SHT wrappers for SHTns.

These wrappers dynamically resolve SHTns MPI entrypoints at runtime via
environment variables or an explicit `enable_native_mpi!` call. This avoids
hard-coding symbol names across different SHTns builds.

Required symbols (set via ENV or function args):
- SHTNSKIT_MPI_CREATE   :: name of the MPI create function
- SHTNSKIT_MPI_SET_GRID :: name of the MPI set_grid function
- SHTNSKIT_MPI_SH2SPAT  :: name of the MPI spectral→spatial function
- SHTNSKIT_MPI_SPAT2SH  :: name of the MPI spatial→spectral function
- SHTNSKIT_MPI_FREE     :: name of the MPI free function

All functions are assumed to be drop-in equivalents of the CPU ones, possibly
capturing MPI communicator inside the returned configuration. If your SHTns
build expects different signatures (e.g., explicit MPI_Comm arguments), please
share the exact C prototypes so we can wire precise calls.
"""

# Thread-safe pointer storage for MPI functions
const _mpi_ptr_lock = ReentrantLock()

const _mpi_create_ptr   = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_setgrid_ptr  = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_sh2spat_ptr  = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_spat2sh_ptr  = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_free_ptr     = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_vec_t2u_ptr  = Ref{Ptr{Cvoid}}(C_NULL)
const _mpi_vec_u2t_ptr  = Ref{Ptr{Cvoid}}(C_NULL)

@inline function _mpi_load_ptr(ref::Ref{Ptr{Cvoid}})
    lock(_mpi_ptr_lock) do
        ref[]
    end
end

@inline function _mpi_store_ptr!(ref::Ref{Ptr{Cvoid}}, value::Ptr{Cvoid})
    lock(_mpi_ptr_lock) do
        ref[] = value
    end
end

"""
    enable_native_mpi!(; create, set_grid, sh2spat, spat2sh, free) -> Bool

Resolve and cache SHTns MPI entrypoints from the loaded `libshtns` shared
library. If keyword arguments are omitted, environment variables with matching
names are used. Returns true if at least one symbol is found.
"""
function enable_native_mpi!(; create   ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_CREATE", nothing),
                               set_grid ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_SET_GRID", nothing),
                               sh2spat  ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_SH2SPAT", nothing),
                               spat2sh  ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_SPAT2SH", nothing),
                               free     ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_FREE", nothing),
                               vec_t2u  ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_VEC_TORPOL2UV", nothing),
                               vec_u2t  ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_VEC_UV2TORPOL", nothing))
    found = false
    try
        handle = Libdl.dlopen(SHTnsKit.libshtns)
        if create !== nothing
            if (sym = Libdl.dlsym_e(handle, create)) !== C_NULL
                _mpi_store_ptr!(_mpi_create_ptr, sym); found = true
            end
        end
        if set_grid !== nothing
            if (sym = Libdl.dlsym_e(handle, set_grid)) !== C_NULL
                _mpi_store_ptr!(_mpi_setgrid_ptr, sym); found = true
            end
        end
        if sh2spat !== nothing
            if (sym = Libdl.dlsym_e(handle, sh2spat)) !== C_NULL
                _mpi_store_ptr!(_mpi_sh2spat_ptr, sym); found = true
            end
        end
        if spat2sh !== nothing
            if (sym = Libdl.dlsym_e(handle, spat2sh)) !== C_NULL
                _mpi_store_ptr!(_mpi_spat2sh_ptr, sym); found = true
            end
        end
        if free !== nothing
            if (sym = Libdl.dlsym_e(handle, free)) !== C_NULL
                _mpi_store_ptr!(_mpi_free_ptr, sym); found = true
            end
        end
        if vec_t2u !== nothing
            if (sym = Libdl.dlsym_e(handle, vec_t2u)) !== C_NULL
                _mpi_store_ptr!(_mpi_vec_t2u_ptr, sym); found = true
            end
        end
        if vec_u2t !== nothing
            if (sym = Libdl.dlsym_e(handle, vec_u2t)) !== C_NULL
                _mpi_store_ptr!(_mpi_vec_u2t_ptr, sym); found = true
            end
        end
    catch
        # ignore
    end
    return found
end

"""Return true if at least one MPI entrypoint is enabled."""
is_native_mpi_enabled() = (_mpi_load_ptr(_mpi_create_ptr)  != C_NULL) ||
                          (_mpi_load_ptr(_mpi_setgrid_ptr) != C_NULL) ||
                          (_mpi_load_ptr(_mpi_sh2spat_ptr) != C_NULL) ||
                          (_mpi_load_ptr(_mpi_spat2sh_ptr) != C_NULL) ||
                          (_mpi_load_ptr(_mpi_free_ptr)    != C_NULL) ||
                          (_mpi_load_ptr(_mpi_vec_t2u_ptr) != C_NULL) ||
                          (_mpi_load_ptr(_mpi_vec_u2t_ptr) != C_NULL)

# Try common default MPI symbol names if none provided via ENV
try
    if !is_native_mpi_enabled()
        enable_native_mpi!(; create="shtns_mpi_create_with_opts",
                              set_grid="shtns_mpi_set_grid",
                              sh2spat="shtns_mpi_sh_to_spat",
                              spat2sh="shtns_mpi_spat_to_sh",
                              free="shtns_mpi_free",
                              vec_t2u="shtns_mpi_torpol2uv",
                              vec_u2t="shtns_mpi_uv2torpol")
    end
catch
end

"""
    SHTnsMPIConfig

Represents an MPI-enabled SHTns configuration. Internally wraps a standard
`SHTnsConfig` pointer. Creation uses the MPI-enabled constructor if available;
otherwise falls back to CPU constructor on each rank.
"""
struct SHTnsMPIConfig
    cfg::SHTnsConfig
end

"""
    create_mpi_config(comm, lmax, mmax, mres; flags=UInt32(0)) -> SHTnsMPIConfig

Create a new SHTns configuration, using the MPI-enabled entrypoint if present.
If not enabled, falls back to per-rank CPU config creation.
"""
function create_mpi_config(comm::MPI.Comm, lmax::Integer, mmax::Integer, mres::Integer; flags::UInt32=UInt32(0))
    ptr = _mpi_load_ptr(_mpi_create_ptr)
    if ptr == C_NULL
        # Fallback: per-rank CPU config
        return SHTnsMPIConfig(create_config(lmax, mmax, mres, flags))
    end
    # If the MPI constructor has the same signature as CPU, call directly.
    # Otherwise, please provide the exact prototype and we will adjust.
    cfg_ptr = ccall(ptr, Ptr{Cvoid}, (Cint, Cint, Cint, UInt32), lmax, mmax, mres, flags)
    cfg_ptr == C_NULL && error("SHTns MPI create returned NULL; ensure symbol/signature match")
    return SHTnsMPIConfig(SHTnsConfig(cfg_ptr))
end

"""
    set_grid(cfg, nlat, nphi, grid_type)

Set the spatial grid. Uses MPI entrypoint if available; otherwise CPU variant.
"""
function set_grid(cfg::SHTnsMPIConfig, nlat::Integer, nphi::Integer, grid_type::Integer)
    ptr = _mpi_load_ptr(_mpi_setgrid_ptr)
    if ptr == C_NULL
        return set_grid(cfg.cfg, nlat, nphi, grid_type)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint), cfg.cfg.ptr, nlat, nphi, grid_type)
    return cfg
end

"""MPI-enabled synthesis (spectral→spatial) in-place."""
function synthesize!(cfg::SHTnsMPIConfig,
                              sh::AbstractVector{Float64},
                              spat::AbstractMatrix{Float64})
    @assert length(sh) == get_nlm(cfg.cfg)
    @assert size(spat) == (get_nlat(cfg.cfg), get_nphi(cfg.cfg))
    ptr = _mpi_load_ptr(_mpi_sh2spat_ptr)
    if ptr == C_NULL
        return synthesize!(cfg.cfg, sh, spat)
    end
    # Reuse the per-config lock from high-level helpers if available
    lk = try
        SHTnsKit._get_lock(cfg.cfg)
    catch
        nothing
    end
    if lk !== nothing
        Base.lock(lk)
    end
    try
        ccall(ptr, Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, sh),
              Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)))
    finally
        if lk !== nothing
            Base.unlock(lk)
        end
    end
    return spat
end

"""MPI-enabled analysis (spatial→spectral) in-place."""
function analyze!(cfg::SHTnsMPIConfig,
                           spat::AbstractMatrix{Float64},
                           sh::AbstractVector{Float64})
    @assert size(spat) == (get_nlat(cfg.cfg), get_nphi(cfg.cfg))
    @assert length(sh) == get_nlm(cfg.cfg)
    ptr = _mpi_load_ptr(_mpi_spat2sh_ptr)
    if ptr == C_NULL
        return analyze!(cfg.cfg, spat, sh)
    end
    lk = try
        SHTnsKit._get_lock(cfg.cfg)
    catch
        nothing
    end
    if lk !== nothing
        Base.lock(lk)
    end
    try
        ccall(ptr, Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)),
              Base.unsafe_convert(Ptr{Float64}, sh))
    finally
        if lk !== nothing
            Base.unlock(lk)
        end
    end
    return sh
end

"""Free the MPI-enabled configuration (uses MPI entrypoint if available)."""
function free_config(cfg::SHTnsMPIConfig)
    ptr = _mpi_load_ptr(_mpi_free_ptr)
    if ptr == C_NULL
        return free_config(cfg.cfg)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid},), cfg.cfg.ptr)
    try
        SHTnsKit._on_free_config(cfg.cfg)
    catch
    end
    return nothing
end

"""MPI-enabled vector synthesis (tor/pol -> u,v) in-place, if symbols are set."""
function synthesize_vec!(cfg::SHTnsMPIConfig,
                                  tor::AbstractVector{Float64}, pol::AbstractVector{Float64},
                                  u::AbstractMatrix{Float64}, v::AbstractMatrix{Float64})
    ptr = _mpi_load_ptr(_mpi_vec_t2u_ptr)
    if ptr == C_NULL
        return synthesize_vec!(cfg.cfg, tor, pol, u, v)
    end
    @assert length(tor) == get_nlm(cfg.cfg) && length(pol) == get_nlm(cfg.cfg)
    @assert size(u) == (get_nlat(cfg.cfg), get_nphi(cfg.cfg)) && size(v) == size(u)
    lk = try
        SHTnsKit._get_lock(cfg.cfg)
    catch
        nothing
    end
    if lk !== nothing; Base.lock(lk); end
    try
        ccall(ptr, Cvoid,
              (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, tor), Base.unsafe_convert(Ptr{Float64}, pol),
              Base.unsafe_convert(Ptr{Float64}, reshape(u, :)), Base.unsafe_convert(Ptr{Float64}, reshape(v, :)))
    finally
        if lk !== nothing; Base.unlock(lk); end
    end
    return u, v
end

"""MPI-enabled vector analysis (u,v -> tor/pol) in-place, if symbols are set."""
function analyze_vec!(cfg::SHTnsMPIConfig,
                               u::AbstractMatrix{Float64}, v::AbstractMatrix{Float64},
                               tor::AbstractVector{Float64}, pol::AbstractVector{Float64})
    ptr = _mpi_load_ptr(_mpi_vec_u2t_ptr)
    if ptr == C_NULL
        return analyze_vec!(cfg.cfg, u, v, tor, pol)
    end
    @assert size(u) == (get_nlat(cfg.cfg), get_nphi(cfg.cfg)) && size(v) == size(u)
    @assert length(tor) == get_nlm(cfg.cfg) && length(pol) == get_nlm(cfg.cfg)
    lk = try
        SHTnsKit._get_lock(cfg.cfg)
    catch
        nothing
    end
    if lk !== nothing; Base.lock(lk); end
    try
        ccall(ptr, Cvoid,
              (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
              Base.unsafe_convert(Ptr{Float64}, reshape(u, :)), Base.unsafe_convert(Ptr{Float64}, reshape(v, :)),
              Base.unsafe_convert(Ptr{Float64}, tor), Base.unsafe_convert(Ptr{Float64}, pol))
    finally
        if lk !== nothing; Base.unlock(lk); end
    end
    return tor, pol
end


end # module
