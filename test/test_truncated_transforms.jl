using Test
using SHTnsKit

@testset "Truncated Transforms" begin
    lmax, mmax = 8, 6
    nlat, nphi = 24, 32
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "Scalar truncated transforms" begin
        # Create test data with known spectral content
        qlm_full = randn(Complex{Float64}, cfg.nlm)
        vr = allocate_spatial(cfg)
        
        # Test different truncation levels  
        for ltr in [2, 4, 6, lmax]
            qlm_truncated = zeros(Complex{Float64}, cfg.nlm)
            
            # Forward truncated transform
            sh_to_spat_l(cfg, qlm_full, vr, ltr)
            spat_to_sh_l(cfg, vr, qlm_truncated, ltr)
            
            # Check that only coefficients with l ≤ ltr are non-zero
            for (idx, (l, m)) in enumerate(cfg.lm_indices)
                if l <= ltr
                    # These coefficients might be non-zero
                    @test length(qlm_truncated) >= idx
                else
                    # These should be exactly zero
                    @test abs(qlm_truncated[idx]) < 1e-14
                end
            end
            
            # Test round-trip accuracy for included modes
            sh_to_spat_l(cfg, qlm_truncated, vr, ltr)
            qlm_roundtrip = zeros(Complex{Float64}, cfg.nlm)  
            spat_to_sh_l(cfg, vr, qlm_roundtrip, ltr)
            
            # Coefficients with l ≤ ltr should be preserved
            for (idx, (l, m)) in enumerate(cfg.lm_indices)
                if l <= ltr
                    @test abs(qlm_roundtrip[idx] - qlm_truncated[idx]) < 1e-12
                end
            end
        end
    end
    
    @testset "Vector truncated transforms" begin
        # Test vector transforms with truncation
        slm_full = randn(Complex{Float64}, cfg.nlm)
        tlm_full = randn(Complex{Float64}, cfg.nlm)
        vt = allocate_spatial(cfg)
        vp = allocate_spatial(cfg)
        
        ltr = 4
        
        # Forward synthesis
        sphtor_to_spat_l(cfg, slm_full, tlm_full, vt, vp, ltr)
        
        # Backward analysis  
        slm_trunc = zeros(Complex{Float64}, cfg.nlm)
        tlm_trunc = zeros(Complex{Float64}, cfg.nlm)
        spat_to_sphtor_l(cfg, vt, vp, slm_trunc, tlm_trunc, ltr)
        
        # Check truncation property
        for (idx, (l, m)) in enumerate(cfg.lm_indices)
            if l > ltr
                @test abs(slm_trunc[idx]) < 1e-14
                @test abs(tlm_trunc[idx]) < 1e-14
            end
        end
    end
    
    @testset "3D vector truncated transforms" begin
        # Test QST transforms with truncation
        qlm = randn(Complex{Float64}, cfg.nlm)
        slm = randn(Complex{Float64}, cfg.nlm)
        tlm = randn(Complex{Float64}, cfg.nlm)
        
        vr = allocate_spatial(cfg)
        vt = allocate_spatial(cfg)  
        vp = allocate_spatial(cfg)
        
        ltr = 5
        
        # Forward synthesis
        shqst_to_spat_l(cfg, qlm, slm, tlm, vr, vt, vp, ltr)
        
        # Backward analysis
        qlm_out = zeros(Complex{Float64}, cfg.nlm)
        slm_out = zeros(Complex{Float64}, cfg.nlm)
        tlm_out = zeros(Complex{Float64}, cfg.nlm)
        
        spat_to_shqst_l(cfg, vr, vt, vp, qlm_out, slm_out, tlm_out, ltr)
        
        # Check that high-l modes are zero
        for (idx, (l, m)) in enumerate(cfg.lm_indices)
            if l > ltr
                @test abs(qlm_out[idx]) < 1e-14
                @test abs(slm_out[idx]) < 1e-14  
                @test abs(tlm_out[idx]) < 1e-14
            end
        end
    end
    
    @testset "Gradient truncated transform" begin
        # Test gradient computation with truncation
        slm = randn(Complex{Float64}, cfg.nlm)
        gt = allocate_spatial(cfg)
        gp = allocate_spatial(cfg)
        
        ltr = 3
        
        sh_to_grad_spat_l(cfg, slm, gt, gp, ltr)
        
        # Compare with full spheroidal synthesis (should be equivalent)
        gt_ref = allocate_spatial(cfg)
        gp_ref = allocate_spatial(cfg) 
        zero_tlm = zeros(Complex{Float64}, cfg.nlm)
        
        sphtor_to_spat_l(cfg, slm, zero_tlm, gt_ref, gp_ref, ltr)
        
        @test norm(gt - gt_ref) < 1e-12
        @test norm(gp - gp_ref) < 1e-12
    end
    
    @testset "Performance of truncated transforms" begin
        # Truncated transforms should be faster than full transforms
        # This is a basic sanity check
        
        qlm = randn(Complex{Float64}, cfg.nlm)
        vr = allocate_spatial(cfg)
        
        # Time full transform
        t_full = @elapsed begin
            for _ in 1:10
                sh_to_spat!(cfg, qlm, vr)
            end
        end
        
        # Time truncated transform (low truncation)
        ltr = 2
        t_trunc = @elapsed begin
            for _ in 1:10
                sh_to_spat_l(cfg, qlm, vr, ltr)
            end
        end
        
        # Truncated should generally be faster (though overhead might dominate for small problems)
        @test t_trunc <= 2 * t_full  # Allow some overhead
    end
    
    @testset "Edge cases" begin
        qlm = randn(Complex{Float64}, cfg.nlm)
        vr = allocate_spatial(cfg)
        
        # Test ltr = 0 (only l=0 mode)
        sh_to_spat_l(cfg, qlm, vr, 0)
        
        # Result should be constant (only Y_0^0 contribution)  
        mean_val = sum(vr) / length(vr)
        @test maximum(abs.(vr .- mean_val)) < 1e-12
        
        # Test ltr = lmax (should be equivalent to full transform)
        vr_full = allocate_spatial(cfg)
        vr_trunc = allocate_spatial(cfg)
        
        sh_to_spat!(cfg, qlm, vr_full)
        sh_to_spat_l(cfg, qlm, vr_trunc, lmax)
        
        @test norm(vr_full - vr_trunc) < 1e-12
    end
end

@testset "Truncation Consistency" begin
    lmax, mmax = 6, 4
    nlat, nphi = 18, 24
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "Analysis-synthesis consistency" begin  
        # Create spatial data with limited spectral content
        qlm_orig = zeros(Complex{Float64}, cfg.nlm)
        
        # Set only low-l coefficients
        ltr = 3
        for (idx, (l, m)) in enumerate(cfg.lm_indices)
            if l <= ltr
                qlm_orig[idx] = randn(Complex{Float64})
            end
        end
        
        # Synthesize to spatial
        vr = allocate_spatial(cfg)
        sh_to_spat!(cfg, qlm_orig, vr)
        
        # Analyze with different truncation levels
        for ltr_test in [1, 2, 3, 4]
            qlm_test = zeros(Complex{Float64}, cfg.nlm)
            spat_to_sh_l(cfg, vr, qlm_test, ltr_test)
            
            # Check that recovered coefficients match original (up to ltr_test)
            for (idx, (l, m)) in enumerate(cfg.lm_indices)
                if l <= min(ltr, ltr_test)
                    @test abs(qlm_test[idx] - qlm_orig[idx]) < 1e-12
                elseif l <= ltr_test  
                    # Original was zero, test should be close to zero  
                    @test abs(qlm_test[idx]) < 1e-12
                else
                    # Above truncation limit, should be exactly zero
                    @test abs(qlm_test[idx]) < 1e-14
                end
            end
        end
    end
end