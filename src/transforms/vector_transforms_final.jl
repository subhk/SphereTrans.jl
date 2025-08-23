"""
Vector spherical harmonic transforms (real API).
Thin wrappers over complex vector transforms using a clean and reversible
packed-real (m≥0) ↔ canonical complex (±m) mapping consistent with SHTns.
"""

# Internal helpers: packed real (one coeff per m≥0) ↔ real-pair (cos,sin) ↔ complex ±m.
function _packed_to_realpair(cfg::SHTnsConfig{T}, packed::AbstractVector{T}) where T
    length(packed) == cfg.nlm || error("packed real coeff length must equal nlm")
    rp = Vector{T}(undef, SHTnsKit.real_nlm(cfg))
    pos = 1
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m=0
        # find index in packed for (l,0)
        # cfg.lm_indices enumerates packed order already
        # We can walk packed via a cursor as well
        # Use lm_indices: find tuple (l,0)
        for (k, (ll, mm)) in enumerate(cfg.lm_indices)
            if ll == l && mm == 0
                rp[pos] = packed[k]
                pos += 1
                break
            end
        end
        # m>0: set cos=a, sin=0
        for m in cfg.mres:cfg.mres:maxm
            # find (l,m) in packed
            for (k, (ll, mm)) in enumerate(cfg.lm_indices)
                if ll == l && mm == m
                    rp[pos] = packed[k]   # a_c
                    rp[pos+1] = zero(T)   # a_s
                    pos += 2
                    break
                end
            end
        end
    end
    return rp
end

function _realpair_to_packed(cfg::SHTnsConfig{T}, rp::AbstractVector{T}) where T
    length(rp) == SHTnsKit.real_nlm(cfg) || error("real-pair coeff length mismatch")
    out = Vector{T}(undef, cfg.nlm)
    # Build map from (l,m) to packed index
    pmap = Dict{Tuple{Int,Int}, Int}()
    for (k, lm) in enumerate(cfg.lm_indices)
        pmap[lm] = k
    end
    pos = 1
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m=0
        out[pmap[(l,0)]] = rp[pos]; pos += 1
        # m>0: take a_c, ignore a_s
        for m in cfg.mres:cfg.mres:maxm
            out[pmap[(l,m)]] = rp[pos]; pos += 2
        end
    end
    return out
end

function _packed_real_to_complex(cfg::SHTnsConfig{T}, packed::AbstractVector{T}) where T
    rp = _packed_to_realpair(cfg, packed)
    return SHTnsKit.real_to_complex_coeffs(cfg, rp)
end

function _complex_to_packed_real(cfg::SHTnsConfig{T}, cplx::AbstractVector{Complex{T}}) where T
    rp = SHTnsKit.complex_to_real_coeffs(cfg, cplx)
    return _realpair_to_packed(cfg, rp)
end

"""
    sphtor_to_spat!(cfg::SHTnsConfig{T}, 
                   sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                   u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Synthesize real tangential vector field (u_θ, u_φ) from packed-real spheroidal and toroidal coefficients.
"""
function sphtor_to_spat!(cfg::SHTnsConfig{T},
                        sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                        u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    validate_config(cfg)
    length(sph_coeffs) == cfg.nlm || error("sph_coeffs length must equal nlm")
    length(tor_coeffs) == cfg.nlm || error("tor_coeffs length must equal nlm")
    size(u_theta) == (cfg.nlat, cfg.nphi) || error("u_theta size mismatch")
    size(u_phi) == (cfg.nlat, cfg.nphi) || error("u_phi size mismatch")

    # Convert to canonical complex ±m coefficients and synthesize using complex core
    S_c = _packed_real_to_complex(cfg, sph_coeffs)
    T_c = _packed_real_to_complex(cfg, tor_coeffs)
    uθ_c, uφ_c = SHTnsKit.cplx_synthesize_vector(cfg, S_c, T_c)

    if SHTnsKit.is_robert_form(cfg)
        sines = sin.(cfg.theta_grid)
        @inbounds for i in 1:cfg.nlat
            s = sines[i]
            u_theta[i, :] .= real.(uθ_c[i, :]) .* s
            u_phi[i,   :] .= real.(uφ_c[i, :]) .* s
        end
    else
        u_theta .= real.(uθ_c)
        u_phi  .= real.(uφ_c)
    end
    return u_theta, u_phi
end

"""
    spat_to_sphtor!(cfg::SHTnsConfig{T},
                   u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                   sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T

Analyze real tangential vector field (u_θ, u_φ) into packed-real spheroidal and toroidal coefficients.
"""
function spat_to_sphtor!(cfg::SHTnsConfig{T},
                        u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                        sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T
    validate_config(cfg)
    size(u_theta) == (cfg.nlat, cfg.nphi) || error("u_theta size mismatch")
    size(u_phi) == (cfg.nlat, cfg.nphi) || error("u_phi size mismatch")
    length(sph_coeffs) == cfg.nlm || error("sph_coeffs length must equal nlm")
    length(tor_coeffs) == cfg.nlm || error("tor_coeffs length must equal nlm")

    # Prepare complex inputs, handling Robert form if enabled
    if SHTnsKit.is_robert_form(cfg)
        sines = sin.(cfg.theta_grid)
        uθc = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
        uφc = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nphi)
        @inbounds for i in 1:cfg.nlat
            s = sines[i]
            invs = s > 1e-12 ? (one(T)/s) : zero(T)
            uθc[i, :] = Complex{T}.(u_theta[i, :] .* invs)
            uφc[i, :] = Complex{T}.(u_phi[i,   :] .* invs)
        end
        S_c, T_c = SHTnsKit.cplx_analyze_vector(cfg, uθc, uφc)
        sph_coeffs .= _complex_to_packed_real(cfg, S_c)
        tor_coeffs .= _complex_to_packed_real(cfg, T_c)
    else
        S_c, T_c = SHTnsKit.cplx_analyze_vector(cfg, Complex{T}.(u_theta), Complex{T}.(u_phi))
        sph_coeffs .= _complex_to_packed_real(cfg, S_c)
        tor_coeffs .= _complex_to_packed_real(cfg, T_c)
    end
    return sph_coeffs, tor_coeffs
end

"""
    synthesize_vector(cfg, sph_coeffs, tor_coeffs)

Allocating version of sphtor_to_spat!.
"""
function synthesize_vector(cfg::SHTnsConfig{T}, sph_coeffs::AbstractVector{T}, 
                          tor_coeffs::AbstractVector{T}) where T
    u_theta = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    u_phi = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    return sphtor_to_spat!(cfg, sph_coeffs, tor_coeffs, u_theta, u_phi)
end

"""
    analyze_vector(cfg, u_theta, u_phi)

Allocating version of spat_to_sphtor!.
"""
function analyze_vector(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, 
                       u_phi::AbstractMatrix{T}) where T
    sph_coeffs = Vector{T}(undef, cfg.nlm)
    tor_coeffs = Vector{T}(undef, cfg.nlm)
    return spat_to_sphtor!(cfg, u_theta, u_phi, sph_coeffs, tor_coeffs)
end