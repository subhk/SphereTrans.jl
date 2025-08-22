using Test
using SHTnsKit

@testset "Point Evaluation Functions" begin
    # Create a test configuration
    lmax, mmax = 10, 8
    nlat, nphi = 32, 64
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "sh_to_point" begin
        # Test with a simple Y_0^0 coefficient (constant function)
        qlm = zeros(Complex{Float64}, cfg.nlm)
        qlm[1] = sh00_1(cfg)  # Set (0,0) coefficient to represent unity
        
        # Evaluate at different points
        points = [
            (1.0, 0.0),      # North pole
            (-1.0, 0.0),     # South pole  
            (0.0, 0.0),      # Equator, 0° longitude
            (0.0, π/2),      # Equator, 90° longitude
        ]
        
        for (cost, phi) in points
            val = sh_to_point(cfg, qlm, cost, phi)
            @test abs(val - 1.0) < 1e-12  # Should be unity everywhere
        end
    end
    
    @testset "sh_to_point_cplx" begin
        # Test with complex coefficients
        nlm_cplx = (lmax + 1)^2  # Full complex expansion
        alm = zeros(Complex{Float64}, nlm_cplx)
        
        # Set Y_0^0 coefficient
        alm[1] = 1.0 + 0.5im
        
        val = sh_to_point_cplx(cfg, alm, 0.0, 0.0)  # Equator
        @test abs(real(val) - 1.0) < 1e-12
        @test abs(imag(val) - 0.5) < 1e-12
    end
    
    @testset "sh_to_grad_point" begin
        # Test gradient of Y_1^0 = sqrt(3/(4π)) * cos(θ)
        qlm = zeros(Complex{Float64}, cfg.nlm)
        
        # Find index for (1,0)
        idx_10 = findfirst(i -> cfg.lm_indices[i] == (1, 0), 1:cfg.nlm)
        if idx_10 !== nothing
            qlm[idx_10] = sh10_ct(cfg)
            
            # Evaluate gradient at equator (θ = π/2, cos θ = 0)
            cost, phi = 0.0, 0.0
            gr, gt, gp = sh_to_grad_point(cfg, qlm, cost, phi)
            
            @test abs(gr) < 1e-12  # No radial component on sphere
            @test abs(gt) > 1e-12  # Should have θ component
            @test abs(gp) < 1e-12  # No φ component for m=0
        end
    end
    
    @testset "shqst_to_point" begin
        # Test 3D vector evaluation
        qlm = zeros(Complex{Float64}, cfg.nlm)
        slm = zeros(Complex{Float64}, cfg.nlm)
        tlm = zeros(Complex{Float64}, cfg.nlm)
        
        # Set some test coefficients
        qlm[1] = 1.0  # Radial component
        
        cost, phi = 0.0, 0.0  # Equator
        vr, vt, vp = shqst_to_point(cfg, qlm, slm, tlm, cost, phi)
        
        @test abs(vr - real(sh00_1(cfg))) < 1e-10  # Should match radial component
        @test abs(vt) < 1e-12  # No horizontal components with zero S,T
        @test abs(vp) < 1e-12
    end
end

@testset "Special Value Functions" begin
    lmax, mmax = 5, 5  
    nlat, nphi = 16, 32
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "Basic special values" begin
        # Test sh00_1 - should be 1/√(4π) for orthonormal
        val = sh00_1(cfg)
        expected = 1/sqrt(4π)
        @test abs(val - expected) < 1e-12
        
        # Test sh10_ct - should be √(3/(4π)) for orthonormal  
        val = sh10_ct(cfg)
        expected = sqrt(3/(4π))
        @test abs(val - expected) < 1e-12
        
        # Test sh11_st - should be -√(3/(8π)) for orthonormal
        val = sh11_st(cfg)
        expected = -sqrt(3/(8π))
        @test abs(val - expected) < 1e-12
    end
    
    @testset "shlm_e1" begin
        # Test unit energy coefficients
        for l in 0:min(3, lmax)
            for m in 0:min(l, mmax)
                val = shlm_e1(cfg, l, m)
                @test val > 0  # Should be positive
                @test isfinite(val)  # Should be finite
            end
        end
    end
    
    @testset "gauss_weights" begin
        weights = gauss_weights(cfg)
        @test length(weights) == cfg.nlat
        @test all(w -> w > 0, weights)  # All weights should be positive
        
        # Weights should sum to 2 for integration over cos θ from -1 to 1
        @test abs(sum(weights) - 2.0) < 1e-12
    end
end

@testset "Legendre Polynomial Evaluation" begin
    lmax = 8
    cfg = create_gauss_config(Float64, lmax, lmax, 16, 32)
    
    @testset "legendre_sphPlm_array" begin
        # Test at x = 0 (equator)
        x = 0.0
        m = 0
        yl = legendre_sphPlm_array(cfg, lmax, m, x)
        
        @test length(yl) == lmax - m + 1
        @test isfinite(yl[1])  # P_0^0(0) should be finite
        
        # Test orthogonality property (simplified test)
        # P_l^0(0) = 0 for odd l, nonzero for even l
        for (i, l) in enumerate(m:lmax)
            if l % 2 == 1  # Odd l
                @test abs(yl[i]) < 1e-12
            end
        end
    end
    
    @testset "legendre_sphPlm_deriv_array" begin
        x = 0.5  # cos(60°)
        sint = sqrt(1 - x^2)
        m = 1
        
        if lmax >= m
            yl, dyl = legendre_sphPlm_deriv_array(cfg, lmax, m, x, sint)
            
            @test length(yl) == length(dyl)
            @test length(yl) == lmax - m + 1
            @test all(isfinite, yl)
            @test all(isfinite, dyl)
        end
    end
end