"""
Full 3D Vector Spherical Harmonic Transforms (Real API)

Implements the recommended SHTns 3D approach using combined radial-spheroidal-toroidal 
transforms for optimal accuracy and performance.

Based on SHTns documentation: "They should be prefered over separate calls to scalar 
and 2D vector transforms as they can be significantly faster."
"""

"""
    qst_to_spat!(cfg::SHTnsConfig{T}, 
                 q_coeffs::AbstractVector{T}, s_coeffs::AbstractVector{T}, t_coeffs::AbstractVector{T},
                 v_r::AbstractMatrix{T}, v_theta::AbstractMatrix{T}, v_phi::AbstractMatrix{T}) where T

Transform radial-spheroidal-toroidal coefficients to 3D spatial vector field (Vr, Vθ, Vφ).
This implements the 3D approach recommended by SHTns for optimal accuracy.
"""
function qst_to_spat!(cfg::SHTnsConfig{T},
                     q_coeffs::AbstractVector{T}, s_coeffs::AbstractVector{T}, t_coeffs::AbstractVector{T},
                     v_r::AbstractMatrix{T}, v_theta::AbstractMatrix{T}, v_phi::AbstractMatrix{T}) where T
    validate_config(cfg)
    length(q_coeffs) == cfg.nlm || error("q_coeffs length must equal nlm")
    length(s_coeffs) == cfg.nlm || error("s_coeffs length must equal nlm") 
    length(t_coeffs) == cfg.nlm || error("t_coeffs length must equal nlm")
    size(v_r) == (cfg.nlat, cfg.nphi) || error("v_r size mismatch")
    size(v_theta) == (cfg.nlat, cfg.nphi) || error("v_theta size mismatch")
    size(v_phi) == (cfg.nlat, cfg.nphi) || error("v_phi size mismatch")

    nlat, nphi = cfg.nlat, cfg.nphi
    
    # Convert to complex coefficients using the same mapping as vector_transforms_final.jl
    Q_c = _packed_real_to_complex(cfg, q_coeffs)
    S_c = _packed_real_to_complex(cfg, s_coeffs) 
    T_c = _packed_real_to_complex(cfg, t_coeffs)
    
    # Use complex 3D synthesis
    Vr_c, Vθ_c, Vφ_c = SHTnsKit.cplx_synthesize_3d_vector(cfg, Q_c, S_c, T_c)
    
    # Extract real parts
    v_r .= real.(Vr_c)
    v_theta .= real.(Vθ_c)
    v_phi .= real.(Vφ_c)
    
    return v_r, v_theta, v_phi
end

"""
    spat_to_qst!(cfg::SHTnsConfig{T},
                 v_r::AbstractMatrix{T}, v_theta::AbstractMatrix{T}, v_phi::AbstractMatrix{T},
                 q_coeffs::AbstractVector{T}, s_coeffs::AbstractVector{T}, t_coeffs::AbstractVector{T}) where T

Transform 3D spatial vector field (Vr, Vθ, Vφ) to radial-spheroidal-toroidal coefficients.
This implements the 3D approach recommended by SHTns for optimal accuracy.
"""
function spat_to_qst!(cfg::SHTnsConfig{T},
                     v_r::AbstractMatrix{T}, v_theta::AbstractMatrix{T}, v_phi::AbstractMatrix{T},
                     q_coeffs::AbstractVector{T}, s_coeffs::AbstractVector{T}, t_coeffs::AbstractVector{T}) where T
    validate_config(cfg)
    size(v_r) == (cfg.nlat, cfg.nphi) || error("v_r size mismatch")
    size(v_theta) == (cfg.nlat, cfg.nphi) || error("v_theta size mismatch") 
    size(v_phi) == (cfg.nlat, cfg.nphi) || error("v_phi size mismatch")
    length(q_coeffs) == cfg.nlm || error("q_coeffs length must equal nlm")
    length(s_coeffs) == cfg.nlm || error("s_coeffs length must equal nlm")
    length(t_coeffs) == cfg.nlm || error("t_coeffs length must equal nlm")

    # Use complex 3D analysis  
    Q_c, S_c, T_c = SHTnsKit.cplx_analyze_3d_vector(cfg, Complex{T}.(v_r), Complex{T}.(v_theta), Complex{T}.(v_phi))
    
    # Convert back to packed real format
    q_coeffs .= _complex_to_packed_real(cfg, Q_c)
    s_coeffs .= _complex_to_packed_real(cfg, S_c)
    t_coeffs .= _complex_to_packed_real(cfg, T_c)
    
    return q_coeffs, s_coeffs, t_coeffs
end

"""
    synthesize_3d_vector(cfg, q_coeffs, s_coeffs, t_coeffs)

Allocating version of qst_to_spat!.
"""
function synthesize_3d_vector(cfg::SHTnsConfig{T}, q_coeffs::AbstractVector{T}, 
                             s_coeffs::AbstractVector{T}, t_coeffs::AbstractVector{T}) where T
    v_r = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    v_theta = Matrix{T}(undef, cfg.nlat, cfg.nphi) 
    v_phi = Matrix{T}(undef, cfg.nlat, cfg.nphi)
    return qst_to_spat!(cfg, q_coeffs, s_coeffs, t_coeffs, v_r, v_theta, v_phi)
end

"""
    analyze_3d_vector(cfg, v_r, v_theta, v_phi)

Allocating version of spat_to_qst!.
"""
function analyze_3d_vector(cfg::SHTnsConfig{T}, v_r::AbstractMatrix{T}, 
                          v_theta::AbstractMatrix{T}, v_phi::AbstractMatrix{T}) where T
    q_coeffs = Vector{T}(undef, cfg.nlm)
    s_coeffs = Vector{T}(undef, cfg.nlm)
    t_coeffs = Vector{T}(undef, cfg.nlm)
    return spat_to_qst!(cfg, v_r, v_theta, v_phi, q_coeffs, s_coeffs, t_coeffs)
end

# Internal helper functions (copied from vector_transforms_final.jl)
function _packed_to_realpair(cfg::SHTnsConfig{T}, packed::AbstractVector{T}) where T
    length(packed) == cfg.nlm || error("packed real coeff length must equal nlm")
    rp = Vector{T}(undef, SHTnsKit.real_nlm(cfg))
    pos = 1
    for l in 0:cfg.lmax
        maxm = min(l, cfg.mmax)
        # m=0
        for (k, (ll, mm)) in enumerate(cfg.lm_indices)
            if ll == l && mm == 0
                rp[pos] = packed[k]
                pos += 1
                break
            end
        end
        # m>0: set cos=a, sin=0
        for m in cfg.mres:cfg.mres:maxm
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