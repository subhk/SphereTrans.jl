using Test
using SHTnsKit

@testset "Spectral ops (complex basis)" begin
    cfg = create_gauss_config(8, 8)
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

