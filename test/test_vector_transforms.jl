using Test
using SHTnsKit
using Random

@testset "Vector transforms: roundtrip" begin
    for (lmax, mmax) in ((4,4), (6,4))
        cfg = create_gauss_config(lmax, mmax)
        nlm = get_nlm(cfg)
        # Random spheroidal/toroidal coefficients (l>=1)
        rng = MersenneTwister(7)
        sph = zeros(Float64, nlm)
        tor = zeros(Float64, nlm)
        for (idx, (l, m)) in enumerate(SHTnsKit.lm_from_index.(Ref(cfg), 1:nlm))
            if l >= 1
                sph[idx] = randn(rng)
                tor[idx] = randn(rng)
            end
        end
        uθ, uϕ = synthesize_vector(cfg, sph, tor)
        sph2, tor2 = analyze_vector(cfg, uθ, uϕ)

        # Ignore l=0 modes (should be zero anyway)
        mask = [l >= 1 for (l, m) in (SHTnsKit.lm_from_index(cfg, i) for i in 1:nlm)]

        num_s = maximum(abs.(sph2[mask] .- sph[mask]))
        den_s = maximum(abs.(sph[mask])) + eps()
        num_t = maximum(abs.(tor2[mask] .- tor[mask]))
        den_t = maximum(abs.(tor[mask])) + eps()
        @test num_s / den_s < 1e-3
        @test num_t / den_t < 1e-3
        destroy_config(cfg)
    end
end
