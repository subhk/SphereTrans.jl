using Test
using SHTnsKit

@testset "Complex rotation: Z-rotations" begin
    cfg = create_gauss_config(6, 6)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    # Build random coefficients for a fixed l
    l = 4
    coeffs = zeros(ComplexF64, n)
    # Fill m from -l..l, others zero
    for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
        if ll == l
            coeffs[idx] = 0.1 * (ll + mm) + 0.2im
        end
    end
    alpha = 0.37
    # Copy and rotate with beta=gamma=0 (pure Z rotation by alpha)
    coeffs_rot = copy(coeffs)
    rotate_complex!(cfg, coeffs_rot; alpha=alpha, beta=0.0, gamma=0.0)
    # Expected: c'_{l,m} = e^{-i m α} c_{l,m}
    for (idx, (ll, mm)) in enumerate(SHTnsKit._cplx_lm_indices(cfg))
        if ll == l
            @test coeffs_rot[idx] ≈ cis(-alpha*mm) * coeffs[idx]
        end
    end
    destroy_config(cfg)
end

