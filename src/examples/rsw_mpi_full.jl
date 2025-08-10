#!/usr/bin/env julia

using MPI
using SHTnsKit

ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OPENBLAS_NUM_THREADS"] = get(ENV, "OPENBLAS_NUM_THREADS", "1")

"""
Nonlinear Rotating Shallow Water on the sphere (flux + vector-invariant form).

Requirements
- Vector transforms must be available: set env variables to point to your SHTns
  vector entrypoints, or call `enable_native_vec!` with symbol names.
  - `SHTNSKIT_VEC_TORPOL2UV` for (ψ, χ) → (u, v)
  - `SHTNSKIT_VEC_UV2TORPOL` for (u, v) → (ψ, χ)
"""

const a   = 6.371e6
const g   = 9.80616
const H0  = 1.0e4
const Ω   = 7.292115e-5
const dt  = 50.0
const nsteps = 200
const out_stride = 50

const lmax = 63
const mmax = 63
const mres = 1
const nlat = 128
const nphi = 256

"""Compute L(l,m)=l(l+1)/a^2 in a packed array order."""
function spectral_L(cfg)
    nlm = get_nlm(cfg)
    L = zeros(Float64, nlm)
    for m in 0:get_mmax(cfg)
        for l in m:get_lmax(cfg)
            idx = lmidx(cfg, l, m) + 1
            L[idx] = l * (l + 1) / (a^2)
        end
    end
    return L
end

"""Compute f(φ)=2Ω sinφ on grid using an equiangular approximation for φ."""
function coriolis_grid(cfg)
    nlat = get_nlat(cfg); nphi = get_nphi(cfg)
    f = Matrix{Float64}(undef, nlat, nphi)
    for i in 1:nlat
        φ = (i - 0.5) * π / nlat - π/2
        fi = 2*Ω * sin(φ)
        @inbounds for j in 1:nphi
            f[i,j] = fi
        end
    end
    return f
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Ensure vector entrypoints are available
    if !is_native_vec_enabled()
        ok = enable_native_vec!()
        !ok && error("Vector transform entrypoints not enabled. Set SHTNSKIT_VEC_TORPOL2UV and SHTNSKIT_VEC_UV2TORPOL.")
    end

    cfg = create_config(lmax, mmax, mres)
    set_grid(cfg, nlat, nphi, 0)

    nlm = get_nlm(cfg)
    L = spectral_L(cfg)
    f = coriolis_grid(cfg)

    # Prognostic fields: total depth h, toroidal ψ_hat, poloidal χ_hat
    h = allocate_spatial(cfg)
    fill!(h, H0)
    ψ = zeros(Float64, nlm)
    χ = zeros(Float64, nlm)

    # Initial perturbation in h: a zonal wave m=2
    @inbounds for j in 1:nphi
        λ = 2π*(j-1)/nphi
        for i in 1:nlat
            h[i,j] += 1.0 * cos(2λ)
        end
    end

    # Allocate work arrays
    u = allocate_spatial(cfg); v = allocate_spatial(cfg)
    Fu = allocate_spatial(cfg); Fv = allocate_spatial(cfg)
    ψF = zeros(Float64, nlm); χF = zeros(Float64, nlm)
    ψrhs = zeros(Float64, nlm); χrhs = zeros(Float64, nlm)
    hrhs = zeros(Float64, nlm)
    gradP = allocate_spatial(cfg)

    for step in 1:nsteps
        # Compute velocities from ψ, χ
        synthesize_vec!(cfg, ψ, χ, u, v)

        # Mass continuity: h_t = -div(h u)
        @. Fu = h * u
        @. Fv = h * v
        analyze_vec!(cfg, Fu, Fv, ψF, χF)
        # divergence = -L * χF; flux divergence tendency in spectral space
        @inbounds for i in 1:nlm
            hrhs[i] = -(-L[i] * χF[i])
        end

        # Momentum (vector-invariant): u_t = - q k×u - ∇(g h + K)
        # Compute PV q = (ζ+f)/h, with ζ = -L*ψ in spectral → synthesize to grid
        ζ = synthesize(cfg, @. -L * ψ)
        K = @. 0.5 * (u^2 + v^2)
        q = similar(h)
        @. q = (ζ + f) / h

        # Compute RHS vector field G = - q k×u - ∇(g h + K)
        Gu = similar(u); Gv = similar(v)
        # k×u = (-v, u)
        @. Gu = - q * (-v)
        @. Gv = - q * (u)
        # Pressure gradient term: ∇(g h + K)
        # Approximate gradient via spectral trick: analyze grad via vector analysis of gradient field
        @. gradP = g*h + K
        # Gradient vector from scalar can be approximated via (tor, pol) = (0, χg), with χg s.t. -L*χg = divergence(∇φ) = ∇²φ
        # So get χg from scalar Laplacian in spectral space
        φhat = analyze(cfg, gradP)
        χg = zeros(Float64, nlm)
        @inbounds for i in 1:nlm
            χg[i] = -φhat[i]   # since divergence = -L*χg = ∇²φ = -L*φhat ⇒ χg = φhat
        end
        # Synthesize gradient vector via vector synthesis with (ψ=0, χ=χg)
        grad_u = allocate_spatial(cfg); grad_v = allocate_spatial(cfg)
        fill!(grad_u, 0.0); fill!(grad_v, 0.0)
        synthesize_vec!(cfg, zeros(Float64, nlm), χg, grad_u, grad_v)
        @. Gu -= grad_u
        @. Gv -= grad_v

        # Project G onto tor/pol potentials to get tendencies for ψ, χ
        analyze_vec!(cfg, Gu, Gv, ψrhs, χrhs)

        # Time update (RK2 for simplicity)
        @. ψ += dt * ψrhs
        @. χ += dt * χrhs
        η = analyze(cfg, h)
        @. η += dt * hrhs
        h .= synthesize(cfg, η)

        if step % out_stride == 0 && rank == 0
            totE = sum(@. 0.5*h*(u^2 + v^2) + 0.5*g*(h - H0)^2)
            @info "step=$(step) energy=$(totE)"
        end
    end

    if rank == 0
        η = analyze(cfg, h)
        spatη = synthesize(cfg, η)
        println("Final sum(|η|) = ", sum(abs, spatη))
    end

    free_config(cfg)
    MPI.Finalize()
end

abspath(PROGRAM_FILE) == @__FILE__ && main()

