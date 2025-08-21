using Test
using SHTnsKit

@testset "Spectral ops (complex basis)" begin
    cfg = create_gauss_config(8, 8)
    # Random complex coefficients helper
    function randc(len; seed=123)
        rng = MersenneTwister(seed)
        z = Vector{ComplexF64}(undef, len)
        for i in 1:len
            z[i] = randn(rng) + randn(rng)*im
        end
        z
    end
    # One-hot coefficient at (l,m)
    for (l, m) in ((2,1), (4,-2), (5,0))
        sh = allocate_complex_spectral(cfg)
        # zero init
        fill!(sh, 0.0 + 0.0im)
        # find index
        idx = nothing
        for (k, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
            if ll == l && mm == m
                idx = k
                break
            end
        end
        @test idx !== nothing
        sh[idx] = 1.0 + 0.0im

        # d/dphi => i*m
        dphi = cplx_spectral_derivative_phi(cfg, sh)
        @test dphi[idx] ≈ (0.0 + 1.0im*m)
        # Laplacian => -l(l+1)
        lap = cplx_spectral_laplacian(cfg, sh)
        @test lap[idx] ≈ (-l*(l+1))
    end
end

@testset "Vector potentials identities" begin
    cfg = create_gauss_config(6, 6)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    # Spheroidal: div(u) = Δ_S S, vort(u) = 0
    S = randc(n, seed=7)
    T = zeros(ComplexF64, n)
    div_spat = cplx_divergence_spatial_from_potentials(cfg, S, T)
    lapS_spat = cplx_sh_to_spat(cfg, cplx_spectral_laplacian(cfg, S))
    @test maximum(abs.(div_spat .- lapS_spat)) / (maximum(abs.(lapS_spat)) + eps()) < 1e-6
    vort_spat = cplx_vorticity_spatial_from_potentials(cfg, S, T)
    @test maximum(abs.(vort_spat)) < 1e-10

    # Toroidal: vort(u) = Δ_S T, div(u) = 0
    S .= 0
    T = randc(n, seed=9)
    vort_spat = cplx_vorticity_spatial_from_potentials(cfg, S, T)
    lapT_spat = cplx_sh_to_spat(cfg, cplx_spectral_laplacian(cfg, T))
    @test maximum(abs.(vort_spat .- lapT_spat)) / (maximum(abs.(lapT_spat)) + eps()) < 1e-6
    div_spat = cplx_divergence_spatial_from_potentials(cfg, S, T)
    @test maximum(abs.(div_spat)) < 1e-10
end
