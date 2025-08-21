using Test
using SHTnsKit
using Random

@testset "Complex vector roundtrip" begin
    for (lmax, mmax) in ((4,4), (6,4))
        cfg = create_gauss_config(lmax, mmax)
        n = length(SHTnsKit._cplx_lm_indices(cfg))
        rng = MersenneTwister(11)
        S = [randn(rng) + randn(rng)*im for _ in 1:n]
        T = [randn(rng) + randn(rng)*im for _ in 1:n]
        # zero-out l=0
        for (idx, (l, m)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
            if l == 0
                S[idx] = 0
                T[idx] = 0
            end
        end
        uθ, uφ = cplx_synthesize_vector(cfg, S, T)
        S2, T2 = cplx_analyze_vector(cfg, uθ, uφ)
        # mask l>=1
        mask = [l >= 1 for (l, m) in SHTnsKit._cplx_lm_indices(cfg)]
        S_err = maximum(abs.((S2 .- S)[mask])) / (maximum(abs.(S[mask])) + eps())
        T_err = maximum(abs.((T2 .- T)[mask])) / (maximum(abs.(T[mask])) + eps())
        @test S_err < 2e-6
        @test T_err < 2e-6
        destroy_config(cfg)
    end
end

