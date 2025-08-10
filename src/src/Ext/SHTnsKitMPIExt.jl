module SHTnsKitMPIExt

using SHTnsKit
import Libdl
import MPI
using Base.Threads: Atomic, atomic_load, atomic_store!

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

const _mpi_create_ptr   = Atomic{Ptr{Cvoid}}(C_NULL)
const _mpi_setgrid_ptr  = Atomic{Ptr{Cvoid}}(C_NULL)
const _mpi_sh2spat_ptr  = Atomic{Ptr{Cvoid}}(C_NULL)
const _mpi_spat2sh_ptr  = Atomic{Ptr{Cvoid}}(C_NULL)
const _mpi_free_ptr     = Atomic{Ptr{Cvoid}}(C_NULL)

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
                               free     ::Union{Nothing,String}=get(ENV, "SHTNSKIT_MPI_FREE", nothing))
    found = false
    try
        handle = Libdl.dlopen(SHTnsKit.libshtns)
        if create !== nothing
            if (sym = Libdl.dlsym_e(handle, create)) !== C_NULL
                atomic_store!(_mpi_create_ptr, sym); found = true
            end
        end
        if set_grid !== nothing
            if (sym = Libdl.dlsym_e(handle, set_grid)) !== C_NULL
                atomic_store!(_mpi_setgrid_ptr, sym); found = true
            end
        end
        if sh2spat !== nothing
            if (sym = Libdl.dlsym_e(handle, sh2spat)) !== C_NULL
                atomic_store!(_mpi_sh2spat_ptr, sym); found = true
            end
        end
        if spat2sh !== nothing
            if (sym = Libdl.dlsym_e(handle, spat2sh)) !== C_NULL
                atomic_store!(_mpi_spat2sh_ptr, sym); found = true
            end
        end
        if free !== nothing
            if (sym = Libdl.dlsym_e(handle, free)) !== C_NULL
                atomic_store!(_mpi_free_ptr, sym); found = true
            end
        end
    catch
        # ignore
    end
    return found
end

"""Return true if at least one MPI entrypoint is enabled."""
is_native_mpi_enabled() = (atomic_load(_mpi_create_ptr)  != C_NULL) ||
                          (atomic_load(_mpi_setgrid_ptr) != C_NULL) ||
                          (atomic_load(_mpi_sh2spat_ptr) != C_NULL) ||
                          (atomic_load(_mpi_spat2sh_ptr) != C_NULL) ||
                          (atomic_load(_mpi_free_ptr)    != C_NULL)

# Try default SHTns MPI symbol names if none provided via ENV
try
    if !is_native_mpi_enabled()
        enable_native_mpi!(; create="create", set_grid="set_grid",
                              sh2spat="sh2spat", spat2sh="spat2sh", free="free")
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
    cfg::SHTnsKit.SHTnsConfig
end

"""
    create_mpi_config(comm, lmax, mmax, mres; flags=UInt32(0)) -> SHTnsMPIConfig

Create a new SHTns configuration, using the MPI-enabled entrypoint if present.
If not enabled, falls back to per-rank CPU config creation.
"""
function create_mpi_config(comm::MPI.Comm, lmax::Integer, mmax::Integer, mres::Integer; flags::UInt32=UInt32(0))
    ptr = atomic_load(_mpi_create_ptr)
    if ptr == C_NULL
        # Fallback: per-rank CPU config
        return SHTnsMPIConfig(SHTnsKit.create_config(lmax, mmax, mres, flags))
    end
    # If the MPI constructor has the same signature as CPU, call directly.
    # Otherwise, please provide the exact prototype and we will adjust.
    cfg_ptr = ccall(ptr, Ptr{Cvoid}, (Cint, Cint, Cint, UInt32), lmax, mmax, mres, flags)
    cfg_ptr == C_NULL && error("SHTns MPI create returned NULL; ensure symbol/signature match")
    return SHTnsMPIConfig(SHTnsKit.SHTnsConfig(cfg_ptr))
end

"""
    set_grid(cfg, nlat, nphi, grid_type)

Set the spatial grid. Uses MPI entrypoint if available; otherwise CPU variant.
"""
function SHTnsKit.set_grid(cfg::SHTnsMPIConfig, nlat::Integer, nphi::Integer, grid_type::Integer)
    ptr = atomic_load(_mpi_setgrid_ptr)
    if ptr == C_NULL
        return SHTnsKit.set_grid(cfg.cfg, nlat, nphi, grid_type)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint), cfg.cfg.ptr, nlat, nphi, grid_type)
    return cfg
end

"""MPI-enabled synthesis (spectral→spatial) in-place."""
function SHTnsKit.synthesize!(cfg::SHTnsMPIConfig,
                              sh::AbstractVector{Float64},
                              spat::AbstractMatrix{Float64})
    @assert length(sh) == SHTnsKit.get_nlm(cfg.cfg)
    @assert size(spat) == (SHTnsKit.get_nlat(cfg.cfg), SHTnsKit.get_nphi(cfg.cfg))
    ptr = atomic_load(_mpi_sh2spat_ptr)
    if ptr == C_NULL
        return SHTnsKit.synthesize!(cfg.cfg, sh, spat)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, sh),
          Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)))
    return spat
end

"""MPI-enabled analysis (spatial→spectral) in-place."""
function SHTnsKit.analyze!(cfg::SHTnsMPIConfig,
                           spat::AbstractMatrix{Float64},
                           sh::AbstractVector{Float64})
    @assert size(spat) == (SHTnsKit.get_nlat(cfg.cfg), SHTnsKit.get_nphi(cfg.cfg))
    @assert length(sh) == SHTnsKit.get_nlm(cfg.cfg)
    ptr = atomic_load(_mpi_spat2sh_ptr)
    if ptr == C_NULL
        return SHTnsKit.analyze!(cfg.cfg, spat, sh)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, reshape(spat, :)),
          Base.unsafe_convert(Ptr{Float64}, sh))
    return sh
end

"""Free the MPI-enabled configuration (uses MPI entrypoint if available)."""
function SHTnsKit.free_config(cfg::SHTnsMPIConfig)
    ptr = atomic_load(_mpi_free_ptr)
    if ptr == C_NULL
        return SHTnsKit.free_config(cfg.cfg)
    end
    ccall(ptr, Cvoid, (Ptr{Cvoid},), cfg.cfg.ptr)
    try
        SHTnsKit._on_free_config(cfg.cfg)
    catch
    end
    return nothing
end

end # module
