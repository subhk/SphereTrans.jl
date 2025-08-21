using Test
using SHTnsKit

@testset "ForwardDiff AD (skip if unavailable)" begin
    try
        using ForwardDiff
    catch
        @info "ForwardDiff not available; skipping ForwardDiff AD tests"
        return
    end

    # Small config due to naive DFT/IDFT fallbacks
    cfg = create_gauss_config(6, 6)
    sh = rand(get_nlm(cfg))
    target = rand(get_nlat(cfg), get_nphi(cfg))
    function loss_fd(shv)
        s = synthesize(cfg, shv)
        return 0.5 * sum(abs2, s .- target) |> real
    end
    g = ForwardDiff.gradient(loss_fd, sh)
    # Reference gradient from adjoint formula
    s = synthesize(cfg, sh)
    g_ref = analyze(cfg, s .- target)
    @test maximum(abs.(g .- g_ref)) / (maximum(abs.(g_ref)) + eps()) < 1e-6
    destroy_config(cfg)
end

