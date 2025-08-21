using Test
using SHTnsKit
using Random

@testset "Complex transforms: roundtrip" begin
    for (lmax, mmax) in ((4,4), (6,4))
        cfg = create_gauss_config(lmax, mmax)
        # allocate complex spectral coefficients
        sh = allocate_complex_spectral(cfg)
        # Random but stable seed
        srand = MersenneTwister(42)
        for i in eachindex(sh)
            sh[i] = randn(srand) + randn(srand)*im
        end
        spatial = synthesize_complex(cfg, sh)
        rec = analyze_complex(cfg, spatial)
        num = maximum(abs.(rec .- sh))
        den = maximum(abs.(sh)) + eps()
        @test num / den < 1e-6
        destroy_config(cfg)
    end
end

@testset "Complex transforms: single Y_l^m" begin
    cfg = create_gauss_config(8, 8)
    for (l, m) in ((2,1), (3,-2), (5,0))
        field = create_complex_test_field(cfg, l, m)
        coeffs = analyze_complex(cfg, field)
        # Find index of (l,m)
        target_idx = nothing
        for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
            if ll == l && mm == m
                target_idx = idx
                break
            end
        end
        @test target_idx !== nothing
        dominant = maximum(abs.(coeffs))
        @test abs(coeffs[target_idx]) / (dominant + eps()) > 0.9
    end
    destroy_config(cfg)
end
