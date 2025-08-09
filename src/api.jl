# Low-level wrappers for the SHTns C library

"""SHTns configuration handle."""
struct SHTnsConfig
    ptr::Ptr{Cvoid}
end

# Name of the shared library. Adjust if needed on other platforms.
const libshtns = "libshtns"

"""
    create_config(lmax, mmax, mres, flags=UInt32(0)) -> SHTnsConfig

Create a new SHTns configuration using `shtns_create_with_opts`.
"""
function create_config(lmax::Integer, mmax::Integer, mres::Integer, flags::UInt32=UInt32(0))
    cfg = ccall((:shtns_create_with_opts, libshtns), Ptr{Cvoid},
                (Cint, Cint, Cint, UInt32), lmax, mmax, mres, flags)
    cfg == C_NULL && error("shtns_create_with_opts returned NULL")
    return SHTnsConfig(cfg)
end

"""
    set_grid(cfg, nlat, nphi, grid_type)

Configure the spatial grid for the transform using `shtns_set_grid`.
"""
function set_grid(cfg::SHTnsConfig, nlat::Integer, nphi::Integer, grid_type::Integer)
    ccall((:shtns_set_grid, libshtns), Cvoid,
          (Ptr{Cvoid}, Cint, Cint, Cint), cfg.ptr, nlat, nphi, grid_type)
    return cfg
end

"""
    sh_to_spat(cfg, sh, spat)

Perform a synthesis (spectral to spatial) transform using `shtns_sh_to_spat`.
The arrays `sh` and `spat` must be pre-allocated.
"""
function sh_to_spat(cfg::SHTnsConfig, sh::AbstractVector{Float64}, spat::AbstractVector{Float64})
    ccall((:shtns_sh_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, sh),
          Base.unsafe_convert(Ptr{Float64}, spat))
    return spat
end

"""
    spat_to_sh(cfg, spat, sh)

Perform an analysis (spatial to spectral) transform using `shtns_spat_to_sh`.
The arrays `spat` and `sh` must be pre-allocated.
"""
function spat_to_sh(cfg::SHTnsConfig, spat::AbstractVector{Float64}, sh::AbstractVector{Float64})
    ccall((:shtns_spat_to_sh, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, spat),
          Base.unsafe_convert(Ptr{Float64}, sh))
    return sh
end

"""
    free_config(cfg)

Free resources associated with a configuration using `shtns_free`.
"""
function free_config(cfg::SHTnsConfig)
    ccall((:shtns_free, libshtns), Cvoid, (Ptr{Cvoid},), cfg.ptr)
    return nothing
end

