# Low-level wrappers for the SHTns C library

"""SHTns configuration handle."""
struct SHTnsConfig
    ptr::Ptr{Cvoid}
end

# Name/handle of the shared library. Prefer SHTns_jll if available.
const libshtns = let
    lib = "libshtns"
    try
        Base.require(:SHTns_jll)
        if get(Base.loaded_modules, :SHTns_jll, nothing) !== nothing
            lib = Base.loaded_modules[:SHTns_jll].libshtns
        end
    catch
        # fall back to system library name
    end
    lib
end

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
    get_lmax(cfg) -> Int

Return the maximum spherical harmonic degree associated with `cfg` using
`shtns_get_lmax`.
"""
function get_lmax(cfg::SHTnsConfig)
    return ccall((:shtns_get_lmax, libshtns), Cint, (Ptr{Cvoid},), cfg.ptr)
end

"""
    get_mmax(cfg) -> Int

Return the maximum order associated with `cfg` using `shtns_get_mmax`.
"""
function get_mmax(cfg::SHTnsConfig)
    return ccall((:shtns_get_mmax, libshtns), Cint, (Ptr{Cvoid},), cfg.ptr)
end

"""
    get_nlat(cfg) -> Int

Retrieve the number of latitudinal grid points set for `cfg` using
`shtns_get_nlat`.
"""
function get_nlat(cfg::SHTnsConfig)
    return ccall((:shtns_get_nlat, libshtns), Cint, (Ptr{Cvoid},), cfg.ptr)
end

"""
    get_nphi(cfg) -> Int

Retrieve the number of longitudinal grid points set for `cfg` using
`shtns_get_nphi`.
"""
function get_nphi(cfg::SHTnsConfig)
    return ccall((:shtns_get_nphi, libshtns), Cint, (Ptr{Cvoid},), cfg.ptr)
end

"""
    get_nlm(cfg) -> Int

Return the number of spherical harmonic coefficients using `shtns_get_nlm`.
"""
function get_nlm(cfg::SHTnsConfig)
    return ccall((:shtns_get_nlm, libshtns), Cint, (Ptr{Cvoid},), cfg.ptr)
end

"""
    lmidx(cfg, l, m) -> Int

Return the packed index corresponding to the spherical harmonic degree `l` and
order `m` using `shtns_lmidx`.
"""
function lmidx(cfg::SHTnsConfig, l::Integer, m::Integer)
    return ccall((:shtns_lmidx, libshtns), Cint,
                 (Ptr{Cvoid}, Cint, Cint), cfg.ptr, l, m)
end

"""
    free_config(cfg)

Free resources associated with a configuration using `shtns_free`.
"""
function free_config(cfg::SHTnsConfig)
    ccall((:shtns_free, libshtns), Cvoid, (Ptr{Cvoid},), cfg.ptr)
    # Allow high-level helpers to cleanup per-config state (locks)
    try
        _on_free_config(cfg)
    catch
        # ignore if high-level not loaded yet
    end
    return nothing
end
