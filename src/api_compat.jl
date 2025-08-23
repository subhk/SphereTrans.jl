"""
API-compatibility layer that mirrors the C `shtns.h` surface in pure Julia.
Only a subset is fully implemented today; the rest throw `NotImplementedError`.
"""

const _SHTNS_VERBOSE = Ref(0)

"""shtns_verbose(level::Integer)"""
function shtns_verbose(level::Integer)
    _SHTNS_VERBOSE[] = Int(level)
    return nothing
end

"""shtns_print_version()"""
function shtns_print_version()
    println("SHTnsKit julia-compatible API; pure Julia core")
end

"""shtns_get_build_info() -> String"""
function shtns_get_build_info()
    return "SHTnsKit.jl: pure Julia SHT; orthonormal normalization; Gauss grid"
end

"""shtns_init(flags, lmax, mmax, mres, nlat, nphi) -> SHTConfig"""
function shtns_init(flags::Integer, lmax::Integer, mmax::Integer, mres::Integer, nlat::Integer, nphi::Integer)
    # flags ignored for now; Gauss grid with on-the-fly Legendre
    return create_gauss_config(Int(lmax), Int(nlat); mmax=Int(mmax), mres=Int(mres), nlon=Int(nphi))
end

"""shtns_create(lmax, mmax, mres, norm) -> SHTConfig"""
function shtns_create(lmax::Integer, mmax::Integer, mres::Integer, norm::Integer)
    if norm != 0 # sht_orthonormal
        throw(ArgumentError("only orthonormal normalization is supported"))
    end
    # defer grid selection to set_grid; but provide minimal Gauss grid consistent with sizes
    nlat = Int(lmax) + 1
    nphi = max(2*Int(mmax)+1, 4)
    return create_gauss_config(Int(lmax), nlat; mmax=Int(mmax), mres=Int(mres), nlon=nphi)
end

"""shtns_set_grid(cfg, flags, eps, nlat, nphi) -> Int"""
function shtns_set_grid(cfg::SHTConfig, flags::Integer, eps::Real, nlat::Integer, nphi::Integer)
    # Recreate cfg with new grid sizes; return 0 on success
    new = create_gauss_config(cfg.lmax, Int(nlat); mmax=cfg.mmax, mres=cfg.mres, nlon=Int(nphi))
    for f in fieldnames(SHTConfig)
        setfield!(cfg, f, getfield(new, f))
    end
    return 0
end

"""shtns_set_grid_auto(cfg, flags, eps, nl_order, nlat_ref, nphi_ref) -> Int"""
function shtns_set_grid_auto(cfg::SHTConfig, flags::Integer, eps::Real, nl_order::Integer, nlat_ref::Ref{Int}, nphi_ref::Ref{Int})
    nlat_ref[] = cfg.lmax + 1
    nphi_ref[] = max(2*cfg.mmax+1, 4)
    return 0
end

"""shtns_create_with_grid(cfg, mmax_new, nofft) -> SHTConfig"""
function shtns_create_with_grid(cfg::SHTConfig, mmax_new::Integer, nofft::Integer)
    mmax2 = Int(mmax_new)
    mmax2 ≤ cfg.mmax || throw(ArgumentError("mmax_new must be ≤ cfg.mmax"))
    return create_gauss_config(cfg.lmax, cfg.nlat; mmax=mmax2, mres=cfg.mres, nlon=cfg.nlon)
end

"""shtns_use_threads(num_threads) -> Int"""
function shtns_use_threads(num_threads::Integer)
    # FFTW threading can be set externally; just return clamped value
    return max(1, Int(num_threads))
end

"""shtns_reset()"""
shtns_reset() = nothing

"""shtns_destroy(cfg)"""
shtns_destroy(::SHTConfig) = nothing

"""shtns_unset_grid(cfg)"""
shtns_unset_grid(::SHTConfig) = nothing

"""shtns_robert_form(cfg, robert)"""
function shtns_robert_form(::SHTConfig, robert::Integer)
    if robert != 0
        throw(ArgumentError("Robert form not implemented in pure Julia core yet"))
    end
    return nothing
end

"""sh00_1(cfg) -> Float64"""
sh00_1(::SHTConfig) = sqrt(4π)

"""sh10_ct(cfg) -> Float64"""
sh10_ct(::SHTConfig) = sqrt(4π/3)

"""sh11_st(cfg) -> Float64"""
sh11_st(::SHTConfig) = -sqrt(2π/3)  # coefficient for sinθ cosφ in complex basis (m=+1)

"""shlm_e1(cfg, l::Integer, m::Integer) -> Float64"""
function shlm_e1(cfg::SHTConfig, l::Integer, m::Integer)
    (0 ≤ m ≤ cfg.mmax && m ≤ l ≤ cfg.lmax) || return 0.0
    return 1.0
end

"""shtns_gauss_wts(cfg, wts::AbstractVector{<:Real}) -> Int"""
function shtns_gauss_wts(cfg::SHTConfig, wts)
    n = min(length(wts), cfg.nlat)
    @inbounds for i in 1:n
        wts[i] = cfg.w[i]
    end
    return n
end

# Rotation and special-operator APIs: not yet implemented
for fname in (
    :SH_Zrotate, :SH_Yrotate, :SH_Yrotate90, :SH_Xrotate90,
    :shtns_rotation_create, :shtns_rotation_destroy,
    :shtns_rotation_set_angles_ZYZ, :shtns_rotation_set_angles_ZXZ, :shtns_rotation_set_angle_axis,
    :shtns_rotation_wigner_d_matrix, :shtns_rotation_apply_cplx, :shtns_rotation_apply_real,
    :spat_cplx_to_SHsphtor, :SHsphtor_to_spat_cplx,
    
    :SHsph_to_spat_l, :SHtor_to_spat_l,
    :SH_to_grad_point, :SHqst_to_point, :SH_to_lat, :SHqst_to_lat,
    :shtns_profiling, :shtns_profiling_read_time, :SH_to_spat_time, :spat_to_SH_time)
    @eval function ($fname)(args...)
        throw(ErrorException(string($(QuoteNode(Symbol(fname))), " not implemented in pure Julia core yet")))
    end
end
