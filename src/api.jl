# Low-level wrappers for the SHTns C library

using LinearAlgebra
using FFTW
using Libdl

# SHTns flags and grid types
module SHTnsFlags
    # Normalization and configuration flags
    const SHT_NO_CS_PHASE = 0
    const SHT_REAL_NORM = 1
    const SHT_ORTHONORMAL = 2
    const SHT_FOURPI = 4
    const SHT_SCHMIDT = 8
    const SHT_SOUTH_POLE_FIRST = 16
    const SHT_LOAD_SAVE_CFG = 32
    const SHT_ALLOW_GPU = 64
    const SHT_ALLOW_PADDING = 128
    const SHT_THETA_CONTIGUOUS = 256
    const SHT_PHI_CONTIGUOUS = 512
    
    # Grid types for set_grid
    const SHT_GAUSS = 0
    const SHT_REGULAR = 1
    const SHT_DCT = 2
    const SHT_QUICK_INIT = 4
end

"""SHTns configuration handle."""
struct SHTnsConfig
    ptr::Ptr{Cvoid}
end

# Name/handle of the shared library. Prefer custom path, then SHTns_jll if available.
const libshtns = let
    # Check for user-specified custom library path
    custom_lib = get(ENV, "SHTNS_LIBRARY_PATH", nothing)
    if custom_lib !== nothing
        # Validate custom library path
        if !isfile(custom_lib)
            @error "Custom SHTns library not found: $custom_lib" 
            @info "Falling back to system library. Check SHTNS_LIBRARY_PATH environment variable."
        else
            # Try to validate it's actually a SHTns library by checking for key symbols
            try
                handle = Libdl.dlopen(custom_lib, Libdl.RTLD_LAZY)
                has_shtns = Libdl.dlsym_e(handle, :shtns_create_with_opts) != C_NULL
                Libdl.dlclose(handle)
                if has_shtns
                    return custom_lib
                else
                    @error "Library $custom_lib does not appear to be a valid SHTns library (missing shtns_create_with_opts symbol)"
                    @info "Falling back to system library."
                end
            catch e
                @error "Failed to validate custom SHTns library $custom_lib: $e"
                @info "Falling back to system library."
            end
        end
    end
    
    # Try SHTns_jll first
    lib = "libshtns"
    try
        Base.require(:SHTns_jll)
        if get(Base.loaded_modules, :SHTns_jll, nothing) !== nothing
            lib = Base.loaded_modules[:SHTns_jll].libshtns
        end
    catch e
        # fall back to system library name
        @debug "SHTns_jll not available, using system library: $e"
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
# === GRID AND COORDINATES ===

"""Get theta coordinate of grid point i using `shtns_gauss_wt`."""
function get_theta(cfg::SHTnsConfig, i::Integer)
    return ccall((:shtns_gauss_wt, libshtns), Float64,
                 (Ptr{Cvoid}, Cint), cfg.ptr, i)
end

"""Get phi coordinate of grid point j."""
function get_phi(cfg::SHTnsConfig, j::Integer)
    nphi = get_nphi(cfg)
    return 2π * (j - 1) / nphi
end

"""Get Gauss-Legendre weights for quadrature using `shtns_gauss_wt`."""
function get_gauss_weights(cfg::SHTnsConfig)
    nlat = get_nlat(cfg)
    weights = Vector{Float64}(undef, nlat)
    for i in 1:nlat
        weights[i] = ccall((:shtns_gauss_wt, libshtns), Float64,
                          (Ptr{Cvoid}, Cint), cfg.ptr, i-1)
    end
    return weights
end

# === COMPLEX FIELD TRANSFORMS ===

"""Complex spectral to spatial transform using `shtns_cplx_sh_to_spat`."""
function cplx_sh_to_spat(cfg::SHTnsConfig, sh::AbstractVector{ComplexF64}, spat::AbstractVector{ComplexF64})
    ccall((:shtns_cplx_sh_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Ptr{ComplexF64}), cfg.ptr,
          Base.unsafe_convert(Ptr{ComplexF64}, sh),
          Base.unsafe_convert(Ptr{ComplexF64}, spat))
    return spat
end

"""Complex spatial to spectral transform using `shtns_cplx_spat_to_sh`."""
function cplx_spat_to_sh(cfg::SHTnsConfig, spat::AbstractVector{ComplexF64}, sh::AbstractVector{ComplexF64})
    ccall((:shtns_cplx_spat_to_sh, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Ptr{ComplexF64}), cfg.ptr,
          Base.unsafe_convert(Ptr{ComplexF64}, spat),
          Base.unsafe_convert(Ptr{ComplexF64}, sh))
    return sh
end

# === VECTOR TRANSFORMS ===

"""Transform spheroidal and toroidal coefficients to vector components using `shtns_SHsphtor_to_spat`."""
function SHsphtor_to_spat(cfg::SHTnsConfig, 
                         Slm::AbstractVector{Float64}, Tlm::AbstractVector{Float64},
                         Vt::AbstractVector{Float64}, Vp::AbstractVector{Float64})
    ccall((:shtns_SHsphtor_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, Slm), Base.unsafe_convert(Ptr{Float64}, Tlm),
          Base.unsafe_convert(Ptr{Float64}, Vt), Base.unsafe_convert(Ptr{Float64}, Vp))
    return Vt, Vp
end

"""Transform vector components to spheroidal and toroidal coefficients using `shtns_spat_to_SHsphtor`."""
function spat_to_SHsphtor(cfg::SHTnsConfig,
                         Vt::AbstractVector{Float64}, Vp::AbstractVector{Float64},
                         Slm::AbstractVector{Float64}, Tlm::AbstractVector{Float64})
    ccall((:shtns_spat_to_SHsphtor, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, Vt), Base.unsafe_convert(Ptr{Float64}, Vp),
          Base.unsafe_convert(Ptr{Float64}, Slm), Base.unsafe_convert(Ptr{Float64}, Tlm))
    return Slm, Tlm
end

"""Transform spheroidal coefficients to gradient components using `shtns_SHsph_to_spat`."""
function SHsph_to_spat(cfg::SHTnsConfig,
                      Slm::AbstractVector{Float64},
                      Vt::AbstractVector{Float64}, Vp::AbstractVector{Float64})
    ccall((:shtns_SHsph_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, Slm),
          Base.unsafe_convert(Ptr{Float64}, Vt), Base.unsafe_convert(Ptr{Float64}, Vp))
    return Vt, Vp
end

"""Transform toroidal coefficients to rotational components using `shtns_SHtor_to_spat`."""
function SHtor_to_spat(cfg::SHTnsConfig,
                      Tlm::AbstractVector{Float64},
                      Vt::AbstractVector{Float64}, Vp::AbstractVector{Float64})
    ccall((:shtns_SHtor_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, Tlm),
          Base.unsafe_convert(Ptr{Float64}, Vt), Base.unsafe_convert(Ptr{Float64}, Vp))
    return Vt, Vp
end

# === ROTATIONS ===

"""Rotate spherical harmonic coefficients using `shtns_rotation_wigner`."""
function rotation_wigner(cfg::SHTnsConfig, 
                        alpha::Float64, beta::Float64, gamma::Float64,
                        sh_in::AbstractVector{Float64}, sh_out::AbstractVector{Float64})
    ccall((:shtns_rotation_wigner, libshtns), Cvoid,
          (Ptr{Cvoid}, Float64, Float64, Float64, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          alpha, beta, gamma,
          Base.unsafe_convert(Ptr{Float64}, sh_in),
          Base.unsafe_convert(Ptr{Float64}, sh_out))
    return sh_out
end

"""Apply rotation matrix to spherical harmonics using `shtns_rotate_to_grid`."""
function rotate_to_grid(cfg::SHTnsConfig,
                       alpha::Float64, beta::Float64, gamma::Float64,
                       spat_in::AbstractVector{Float64}, spat_out::AbstractVector{Float64})
    ccall((:shtns_rotate_to_grid, libshtns), Cvoid,
          (Ptr{Cvoid}, Float64, Float64, Float64, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          alpha, beta, gamma,
          Base.unsafe_convert(Ptr{Float64}, spat_in),
          Base.unsafe_convert(Ptr{Float64}, spat_out))
    return spat_out
end

# === MULTIPOLE ANALYSIS ===

"""Compute multipole expansion coefficients using `shtns_multipole`."""
function multipole(cfg::SHTnsConfig, 
                  sh::AbstractVector{Float64}, 
                  Q::AbstractVector{Float64})
    lmax = get_lmax(cfg)
    ccall((:shtns_multipole, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr,
          Base.unsafe_convert(Ptr{Float64}, sh),
          Base.unsafe_convert(Ptr{Float64}, Q))
    return Q
end

# === ENERGY AND POWER SPECTRA ===

"""Compute power spectrum from spherical harmonic coefficients."""
function power_spectrum(cfg::SHTnsConfig, sh::AbstractVector{Float64})
    lmax = get_lmax(cfg)
    power = zeros(Float64, lmax + 1)
    for l in 0:lmax
        for m in 0:min(l, get_mmax(cfg))
            idx = lmidx(cfg, l, m)
            if m == 0
                power[l+1] += sh[idx+1]^2
            else
                power[l+1] += 2 * sh[idx+1]^2
            end
        end
    end
    return power
end

# === ON-THE-FLY TRANSFORMS ===

"""Enable on-the-fly mode for memory-efficient transforms using `shtns_set_size`."""
function set_size(cfg::SHTnsConfig, lmax::Integer, mmax::Integer, mres::Integer)
    ccall((:shtns_set_size, libshtns), Cvoid,
          (Ptr{Cvoid}, Cint, Cint, Cint), cfg.ptr, lmax, mmax, mres)
    return cfg
end

# === OPENMP THREADING ===

"""Set number of OpenMP threads using `shtns_set_num_threads`."""
function set_num_threads(nthreads::Integer)
    ccall((:shtns_set_num_threads, libshtns), Cvoid, (Cint,), nthreads)
    return nothing
end

"""Get current number of OpenMP threads using `shtns_get_num_threads`."""
function get_num_threads()
    return ccall((:shtns_get_num_threads, libshtns), Cint, ())
end

# === GPU ACCELERATION ===

"""Initialize GPU context using `shtns_gpu_init`."""
function gpu_init(device_id::Integer = 0)
    return ccall((:shtns_gpu_init, libshtns), Cint, (Cint,), device_id)
end

"""Finalize GPU context using `shtns_gpu_finalize`."""
function gpu_finalize()
    ccall((:shtns_gpu_finalize, libshtns), Cvoid, ())
    return nothing
end

"""GPU spectral to spatial transform using `shtns_gpu_sh_to_spat`."""
function gpu_sh_to_spat(cfg::SHTnsConfig, sh_d::Ptr{Float64}, spat_d::Ptr{Float64})
    ccall((:shtns_gpu_sh_to_spat, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr, sh_d, spat_d)
    return nothing
end

"""GPU spatial to spectral transform using `shtns_gpu_spat_to_sh`."""
function gpu_spat_to_sh(cfg::SHTnsConfig, spat_d::Ptr{Float64}, sh_d::Ptr{Float64})
    ccall((:shtns_gpu_spat_to_sh, libshtns), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), cfg.ptr, spat_d, sh_d)
    return nothing
end

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

# === HELPER FUNCTIONS FOR AD ===

"""
    get_lm_from_index(cfg::SHTnsConfig, idx::Int) -> (l::Int, m::Int)

Get the spherical harmonic degree l and order m for a given linear index.
This is needed for automatic differentiation rules that need to know
which (l,m) mode corresponds to each spectral coefficient.

This function uses SHTns internal indexing by searching through all valid (l,m) pairs.
"""
function get_lm_from_index(cfg::SHTnsConfig, idx::Int)
    lmax = get_lmax(cfg)
    nlm = get_nlm(cfg)
    @assert 1 <= idx <= nlm "Index must be between 1 and nlm"
    
    # Search through all (l,m) pairs to find the one with the matching index
    for l in 0:lmax
        for m in -l:l
            if lmidx(cfg, l, m) == idx - 1  # lmidx returns 0-based index
                return l, m
            end
        end
    end
    
    error("Could not find (l,m) for index $idx")
end

"""
    get_index_from_lm(cfg::SHTnsConfig, l::Int, m::Int) -> Int

Get the linear index for spherical harmonic degree l and order m.
This is the inverse of get_lm_from_index.

This function uses the SHTns library's built-in indexing via lmidx.
"""
function get_index_from_lm(cfg::SHTnsConfig, l::Int, m::Int)
    lmax = get_lmax(cfg)
    @assert 0 <= l <= lmax "l must be between 0 and lmax"
    @assert -l <= m <= l "m must be between -l and l"
    
    # Use SHTns built-in indexing (converts to 1-based)
    return lmidx(cfg, l, m) + 1
end
