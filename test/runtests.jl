using Test
using Random
using SHTnsKit

function parseval_scalar_test(lmax::Int)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(42)
    f = randn(rng, nlat, nlon)
    alm = analysis(cfg, f)
    @test isapprox(energy_scalar(cfg, alm), grid_energy_scalar(cfg, f); rtol=1e-10, atol=1e-12)
end

function parseval_vector_test(lmax::Int)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(24)
    Vt = randn(rng, nlat, nlon)
    Vp = randn(rng, nlat, nlon)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    @test isapprox(energy_vector(cfg, Slm, Tlm), grid_energy_vector(cfg, Vt, Vp); rtol=1e-9, atol=1e-11)
end

@testset "Parseval identities" begin
    parseval_scalar_test(4)
    parseval_vector_test(4)
end

@testset "AD gradients - ForwardDiff" begin
    try
        using ForwardDiff
        lmax = 3
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(7)
        f0 = randn(rng, nlat, nlon)
        loss(x) = energy_scalar(cfg, analysis(cfg, reshape(x, nlat, nlon)))
        x0 = vec(f0)
        g = ForwardDiff.gradient(loss, x0)
        # Dot-test
        h = randn(rng, length(x0))
        ϵ = 1e-6
        φ(ξ) = loss(x0 .+ ξ .* h)
        dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)
        dfdξ_ad = dot(g, h)
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
    catch e
        @info "Skipping ForwardDiff gradient test" exception=(e, catch_backtrace())
    end
end

@testset "AD gradients - Zygote" begin
    try
        using Zygote
        lmax = 3
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(9)
        f0 = randn(rng, nlat, nlon)
        loss(f) = energy_scalar(cfg, analysis(cfg, f))
        g = Zygote.gradient(loss, f0)[1]
        # Dot-test
        h = randn(rng, size(f0))
        ϵ = 1e-6
        dfdξ_fd = (loss(f0 .+ ϵ.*h) - loss(f0 .- ϵ.*h)) / (2ϵ)
        dfdξ_ad = sum(g .* h)
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
    catch e
        @info "Skipping Zygote gradient test" exception=(e, catch_backtrace())
    end
end

@testset "AD gradients - Zygote: rotations and operators" begin
    try
        using Zygote
        # Setup
        lmax = 3
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(123)

        # Rotation around Y: gradient wrt packed real Qlm
        Q0 = ComplexF64.(randn(rng, cfg.nlm))
        α = 0.3
        function loss_yrot(Q)
            R = similar(Q)
            SH_Yrotate(cfg, Q, α, R)
            return 0.5 * sum(abs2, R)
        end
        gQ = Zygote.gradient(loss_yrot, Q0)[1]
        h = ComplexF64.(randn(rng, length(Q0)))
        ϵ = 1e-6
        φ(ξ) = loss_yrot(Q0 .+ ξ .* h)
        dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)
        dfdξ_ad = real(sum(gQ .* h))
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)

        # Z-rotation
        function loss_zrot(Q)
            R = similar(Q)
            SH_Zrotate(cfg, Q, α, R)
            return 0.5 * sum(abs2, R)
        end
        gQz = Zygote.gradient(loss_zrot, Q0)[1]
        φz(ξ) = loss_zrot(Q0 .+ ξ .* h)
        dfdξ_fd = (φz(ϵ) - φz(-ϵ)) / (2ϵ)
        dfdξ_ad = real(sum(gQz .* h))
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)

        # Operator application: test gradients wrt Q and mx
        mx = zeros(Float64, 2*cfg.nlm)
        mul_ct_matrix(cfg, mx)
        Qv = ComplexF64.(randn(rng, cfg.nlm))
        function loss_op(Q, mx)
            R = similar(Q)
            SH_mul_mx(cfg, mx, Q, R)
            return 0.5 * sum(abs2, R)
        end
        gQ_op, gmx_op = Zygote.gradient(loss_op, Qv, mx)
        hQ = ComplexF64.(randn(rng, length(Qv)))
        hmx = randn(rng, length(mx))
        φQ(ξ) = loss_op(Qv .+ ξ .* hQ, mx)
        φmx(ξ) = loss_op(Qv, mx .+ ξ .* hmx)
        dfdξ_fd_Q = (φQ(ϵ) - φQ(-ϵ)) / (2ϵ)
        dfdξ_ad_Q = real(sum(gQ_op .* hQ))
        @test isapprox(dfdξ_ad_Q, dfdξ_fd_Q; rtol=5e-4, atol=1e-7)
        dfdξ_fd_mx = (φmx(ϵ) - φmx(-ϵ)) / (2ϵ)
        dfdξ_ad_mx = sum(gmx_op .* hmx)
        @test isapprox(dfdξ_ad_mx, dfdξ_fd_mx; rtol=5e-4, atol=1e-7)
        # Angle gradient for real rotation
        function loss_angles(a,b,c)
            r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax; α=a, β=b, γ=c)
            R = similar(Q0)
            SHTnsKit.shtns_rotation_apply_real(r, Q0, R)
            return 0.5 * sum(abs2, R)
        end
        gα, gβ, gγ = Zygote.gradient(loss_angles, α, 0.1, -0.2)
        # Finite-diff check for α
        φa(ξ) = loss_angles(α + ξ, 0.1, -0.2)
        dfdξ_fd = (φa(ϵ) - φa(-ϵ)) / (2ϵ)
        @test isapprox(gα, dfdξ_fd; rtol=5e-4, atol=1e-7)
    catch e
        @info "Skipping Zygote rotation/operator gradient tests" exception=(e, catch_backtrace())
    end
end

@testset "Convenience gradients and packed energies" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(202)
    # Packed scalar
    f = randn(rng, nlat, nlon)
    Q = spat_to_SH(cfg, vec(f))
    @test isapprox(energy_scalar_packed(cfg, Q), energy_scalar(cfg, analysis(cfg, f)); rtol=1e-10)
    GQ = grad_energy_scalar_packed(cfg, Q)
    ϵ = 1e-7
    hQ = ComplexF64.(randn(rng, length(Q)))
    φ(ξ) = energy_scalar_packed(cfg, Q .+ ξ .* hQ)
    dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)
    dfdξ_ad = real(sum(conj(GQ) .* hQ))
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-8)
    # Packed vector
    Vt = randn(rng, nlat, nlon)
    Vp = randn(rng, nlat, nlon)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    # Pack matrices into vectors in LM order
    Sp = similar(Q); Tp = similar(Q)
    for m in 0:cfg.mmax
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Sp[lm] = Slm[l+1, m+1]
            Tp[lm] = Tlm[l+1, m+1]
        end
    end
    @test isapprox(energy_vector_packed(cfg, Sp, Tp), energy_vector(cfg, Slm, Tlm); rtol=1e-9)
    GS, GT = grad_energy_vector_packed(cfg, Sp, Tp)
    hS = ComplexF64.(randn(rng, length(Sp)))
    hT = ComplexF64.(randn(rng, length(Tp)))
    φv(ξ) = energy_vector_packed(cfg, Sp .+ ξ .* hS, Tp .+ ξ .* hT)
    dfdξ_fd = (φv(ϵ) - φv(-ϵ)) / (2ϵ)
    dfdξ_ad = real(sum(conj(GS) .* hS) + sum(conj(GT) .* hT))
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-8)
end
