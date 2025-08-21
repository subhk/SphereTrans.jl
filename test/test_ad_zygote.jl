using Test
using SHTnsKit

@testset "Zygote AD (skip if unavailable)" begin
    try
        using Zygote
    catch
        @info "Zygote not available; skipping Zygote AD tests"
        return
    end

    # Real-basis gradient matches adjoint formula
    cfg = create_gauss_config(8, 8)
    sh = rand(get_nlm(cfg))
    target = rand(get_nlat(cfg), get_nphi(cfg))
    function loss_real(shv)
        s = synthesize(cfg, shv)
        d = s .- target
        return 0.5 * sum(abs2, d)
    end
    val, back = Zygote.pullback(loss_real, sh)
    g = back(1.0)[1]
    # Adjoint: grad = analyze(cfg, synthesize(cfg, sh) - target)
    s = synthesize(cfg, sh)
    g_ref = analyze(cfg, s .- target)
    @test maximum(abs.(g .- g_ref)) / (maximum(abs.(g_ref)) + eps()) < 1e-8

    # Complex canonical gradient also matches adjoint transform
    shc = allocate_complex_spectral(cfg)
    target_c = ComplexF64.(rand(get_nlat(cfg), get_nphi(cfg)))
    function loss_cplx(shv)
        s = cplx_sh_to_spat(cfg, shv)
        d = s .- target_c
        return 0.5 * sum(abs2, d)
    end
    valc, backc = Zygote.pullback(loss_cplx, shc)
    gc = backc(1.0)[1]
    s_c = cplx_sh_to_spat(cfg, shc)
    g_refc = cplx_spat_to_sh(cfg, s_c .- target_c)
    @test maximum(abs.(gc .- g_refc)) / (maximum(abs.(g_refc)) + eps()) < 1e-8
    destroy_config(cfg)
end

