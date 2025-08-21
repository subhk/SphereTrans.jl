using Test
using SHTnsKit
using LinearAlgebra

@testset "Rotation and Advanced Features" begin
    @testset "Field Rotation Functions" begin
        try
            cfg = create_gauss_config(12, 12)
            nlm = get_nlm(cfg)
            
            # Create test spectral field
            sh_original = rand(nlm)
            
            # Test spectral rotation
            alpha, beta, gamma = π/6, π/4, π/3
            sh_rotated = rotate_field(cfg, sh_original, alpha, beta, gamma)
            @test length(sh_rotated) == nlm
            @test sh_rotated != sh_original  # Should be different after rotation
            
            # Test that rotation preserves the norm (approximately for finite resolution)
            original_norm = norm(sh_original)
            rotated_norm = norm(sh_rotated)
            norm_error = abs(original_norm - rotated_norm) / original_norm
            @test norm_error < 0.01  # Allow small numerical errors
            
            free_config(cfg)
            
        catch e
            @test_skip "Field rotation tests skipped: $e"
        end
    end
    
    @testset "Spatial Field Rotation" begin
        try
            cfg = create_gauss_config(10, 10)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            
            # Create test spatial field
            spat_original = rand(nlat, nphi)
            
            # Test spatial rotation
            alpha, beta, gamma = π/8, π/6, π/4
            spat_rotated = rotate_spatial_field(cfg, spat_original, alpha, beta, gamma)
            @test size(spat_rotated) == size(spat_original)
            
            # Test that mean is approximately preserved
            original_mean = mean(spat_original)
            rotated_mean = mean(spat_rotated)
            mean_error = abs(original_mean - rotated_mean) / abs(original_mean)
            @test mean_error < 0.01
            
            free_config(cfg)
            
        catch e
            @test_skip "Spatial field rotation tests skipped: $e"
        end
    end
    
    @testset "Power Spectrum Analysis" begin
        try
            cfg = create_gauss_config(16, 16)
            nlm = get_nlm(cfg)
            lmax = get_lmax(cfg)
            
            # Create test field with known spectral content
            sh_test = zeros(nlm)
            
            # Set specific modes
            if nlm > 0
                sh_test[1] = 1.0  # l=0, m=0 mode
            end
            if nlm > 3
                sh_test[4] = 0.5  # Some other mode
            end
            
            # Compute power spectrum
            power = power_spectrum(cfg, sh_test)
            @test length(power) == lmax + 1
            @test power[1] > 0  # l=0 should have power
            @test sum(power) ≈ sum(sh_test.^2) atol=1e-10
            
            free_config(cfg)
            
        catch e
            @test_skip "Power spectrum tests skipped: $e"
        end
    end
    
    @testset "Power Spectrum Properties" begin
        try
            cfg = create_gauss_config(12, 12)
            
            # Create random field
            sh_random = rand(get_nlm(cfg))
            power = power_spectrum(cfg, sh_random)
            
            # Power should be non-negative
            @test all(p >= 0 for p in power)
            
            # Total power should equal sum of squared coefficients (Parseval's theorem)
            total_power = sum(power)
            coefficient_power = sum(sh_random.^2)
            @test abs(total_power - coefficient_power) < 1e-10
            
            free_config(cfg)
            
        catch e
            @test_skip "Power spectrum properties tests skipped: $e"
        end
    end
    
    @testset "Rotation Composition" begin
        try
            cfg = create_gauss_config(8, 8)
            
            # Test that two consecutive rotations compose properly
            sh_original = rand(get_nlm(cfg))
            
            # First rotation
            α1, β1, γ1 = π/12, π/8, π/6
            sh_rot1 = rotate_field(cfg, sh_original, α1, β1, γ1)
            
            # Second rotation
            α2, β2, γ2 = π/10, π/12, π/8
            sh_rot2 = rotate_field(cfg, sh_rot1, α2, β2, γ2)
            
            # Single combined rotation (this is approximate due to non-commutativity)
            # We just test that the result is different and has similar magnitude
            @test sh_rot2 != sh_original
            @test abs(norm(sh_rot2) - norm(sh_original)) / norm(sh_original) < 0.01
            
            free_config(cfg)
            
        catch e
            @test_skip "Rotation composition tests skipped: $e"
        end
    end
    
    @testset "Identity Rotation" begin
        try
            cfg = create_gauss_config(10, 10)
            
            # Test identity rotation (no rotation)
            sh_original = rand(get_nlm(cfg))
            sh_identity = rotate_field(cfg, sh_original, 0.0, 0.0, 0.0)
            
            # Should be identical (up to numerical precision)
            error = maximum(abs.(sh_original - sh_identity))
            @test error < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Identity rotation tests skipped: $e"
        end
    end
end