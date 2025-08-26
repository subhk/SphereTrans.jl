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
    # Parse norm and flags per SHTns
    nval = Int(norm)
    SHT_NO_CS_PHASE = 256*4
    SHT_REAL_NORM = 256*8
    base_norm = nval % 256
    cs_phase = (nval & SHT_NO_CS_PHASE) == 0
    real_norm = (nval & SHT_REAL_NORM) != 0
    nsym = base_norm == 0 ? :orthonormal : base_norm == 1 ? :fourpi : base_norm == 2 ? :schmidt : :orthonormal
    # defer grid selection to set_grid; but provide minimal Gauss grid consistent with sizes
    nlat = Int(lmax) + 1
    nphi = max(2*Int(mmax)+1, 4)
    return create_gauss_config(Int(lmax), nlat; mmax=Int(mmax), mres=Int(mres), nlon=nphi, norm=nsym, cs_phase=cs_phase, real_norm=real_norm)
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
function shtns_robert_form(cfg::SHTConfig, robert::Integer)
    cfg.robert_form = robert != 0
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

# Print configuration
"""shtns_print_cfg(cfg::SHTConfig)"""
function shtns_print_cfg(cfg::SHTConfig)
    println("SHTConfig:")
    println("  lmax=$(cfg.lmax), mmax=$(cfg.mmax), mres=$(cfg.mres)")
    println("  nlat=$(cfg.nlat), nlon=$(cfg.nlon)")
    println("  nlm=$(cfg.nlm), nspat=$(cfg.nspat)")
    return nothing
end

# Legendre arrays (spherical-harmonic normalized)
"""
    legendre_sphPlm_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, yl::AbstractVector{<:Real}) -> Int

Fill `yl[k]` with normalized P_l^m(x) for l = m..lmax where m = im*mres.
Returns the number of values written.
"""
function legendre_sphPlm_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, yl)
    lmax = Int(lmax); im = Int(im); x = float(x)
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || return 0
    (m ≤ lmax ≤ cfg.lmax) || return 0
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Plm_row!(P, x, cfg.lmax, m)
    n = min(length(yl), lmax - m + 1)
    @inbounds for (k, l) in enumerate(m:(m + n - 1))
        yl[k] = cfg.Nlm[l+1, m+1] * P[l+1]
    end
    return n
end

"""
    legendre_sphPlm_deriv_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, sint::Real,
                                yl::AbstractVector{<:Real}, dyl::AbstractVector{<:Real}) -> Int

Fill `yl` with normalized P_l^m(x) and `dyl` with ∂θ of the normalized functions: `dyl = -sinθ * d/dx (normalized P)`.
`x = cosθ`, `sint = sinθ`.
Returns the number of values written.
"""
function legendre_sphPlm_deriv_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, sint::Real, yl, dyl)
    lmax = Int(lmax); im = Int(im); x = float(x); sint = float(sint)
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || return 0
    (m ≤ lmax ≤ cfg.lmax) || return 0
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
    n = min(length(yl), length(dyl), lmax - m + 1)
    @inbounds for (k, l) in enumerate(m:(m + n - 1))
        N = cfg.Nlm[l+1, m+1]
        yl[k]  = N * P[l+1]
        dyl[k] = -sint * N * dPdx[l+1]
    end
    return n
end

# Memory helpers (no-ops in Julia)
"""shtns_malloc(bytes::Integer) -> Vector{UInt8}"""
function shtns_malloc(bytes::Integer)
    return Vector{UInt8}(undef, Int(bytes))
end
"""shtns_free(::Any)""" shtns_free(::Any) = nothing

"""shtns_set_many(cfg::SHTConfig, howmany::Integer, spec_dist::Integer) -> Int"""
function shtns_set_many(::SHTConfig, howmany::Integer, ::Integer)
    return max(1, Int(howmany))
end

# Remaining unimplemented APIs
for fname in (
)
    @eval function ($fname)(args...)
        throw(ErrorException(string($(QuoteNode(Symbol(fname))), " not implemented in pure Julia core yet")))
    end
end

"""
    SH_to_spat_time(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Vr::AbstractVector{<:Real}) -> Float64

Timed scalar synthesis; returns elapsed seconds and writes result into `Vr`.
"""
function SH_to_spat_time(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Vr::AbstractVector{<:Real})
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    t = @elapsed f = synthesis(cfg, alm_mat; real_output=true)
    Vr .= vec(f)
    return t
end

"""
    spat_to_SH_time(cfg::SHTConfig, Vr::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}) -> Float64

Timed scalar analysis; returns elapsed seconds and writes result into `Qlm`.
"""
function spat_to_SH_time(cfg::SHTConfig, Vr::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex})
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    t = @elapsed alm = analysis(cfg, f)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm[l+1, m+1]
        end
    end
    return t
end

"""shtns_profiling(cfg, on)"""
shtns_profiling(::SHTConfig, on::Integer) = nothing

"""shtns_profiling_read_time(cfg, t1::Ref, t2::Ref) -> Float64"""
function shtns_profiling_read_time(::SHTConfig, t1::Ref{Float64}, t2::Ref{Float64})
    t1[] = 0.0; t2[] = 0.0; return 0.0
end
