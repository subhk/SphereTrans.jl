using SHTnsKit

println("=== AD examples with Zygote and ForwardDiff ===")

println("-- Zygote (real-basis) --")
try
    using Zygote
    # Small config for demo
    cfg = create_gauss_config(8, 8)
    # Loss: L(sh) = 0.5 * || synthesize(cfg, sh) - target ||^2
    target = rand(get_nlat(cfg), get_nphi(cfg))
    function loss_real(sh::AbstractVector)
        s = synthesize(cfg, sh)
        d = s .- target
        return 0.5 * sum(abs2, d)
    end
    sh0 = rand(get_nlm(cfg))
    val, back = Zygote.pullback(loss_real, sh0)
    g = back(1.0f0)[1]
    println("Zygote real: loss=", val, ", ‖∇‖=", norm(g))
    destroy_config(cfg)
catch e
    @warn "Zygote example failed" exception=(e, catch_backtrace())
end

println("-- Zygote (complex canonical) --")
try
    using Zygote
    cfg = create_gauss_config(8, 8)
    target = ComplexF64.(rand(get_nlat(cfg), get_nphi(cfg)))
    function loss_cplx(sh::AbstractVector{ComplexF64})
        s = cplx_sh_to_spat(cfg, sh)
        d = s .- target
        return 0.5 * sum(abs2, d)
    end
    shc0 = allocate_complex_spectral(cfg)
    val, back = Zygote.pullback(loss_cplx, shc0)
    g = back(1.0f0)[1]
    println("Zygote complex: loss=", val, ", ‖∇‖=", norm(g))
    destroy_config(cfg)
catch e
    @warn "Zygote complex example failed" exception=(e, catch_backtrace())
end

println("-- ForwardDiff (real-basis) --")
try
    using ForwardDiff
    cfg = create_gauss_config(6, 6)
    target = rand(get_nlat(cfg), get_nphi(cfg))
    function loss_fd(sh::AbstractVector)
        s = synthesize(cfg, sh)
        # ensure real scalar result for ForwardDiff
        return 0.5 * sum(abs2, s .- target) |> real
    end
    sh0 = rand(get_nlm(cfg))
    g = ForwardDiff.gradient(loss_fd, sh0)
    println("ForwardDiff real: ‖∇‖=", norm(g))
    destroy_config(cfg)
catch e
    @warn "ForwardDiff example failed" exception=(e, catch_backtrace())
end

println("Done.")

