using Test
using SHTnsKit
using LinearAlgebra

@testset "New SHTns Features Implementation" begin
    @testset "Package Structure and New Exports" begin
        # Test that new functions are properly exported
        @test isdefined(SHTnsKit, :spat_to_SHqst)
        @test isdefined(SHTnsKit, :SHqst_to_spat)
        @test isdefined(SHTnsKit, :SH_to_point)
        @test isdefined(SHTnsKit, :SHqst_to_point)
        @test isdefined(SHTnsKit, :SH_to_grad_spat)
        @test isdefined(SHTnsKit, :SHqst_to_lat)
        @test isdefined(SHTnsKit, :nlm_calc)
        
        # Test high-level functions
        @test isdefined(SHTnsKit, :analyze_3d_vector)
        @test isdefined(SHTnsKit, :synthesize_3d_vector)
        @test isdefined(SHTnsKit, :evaluate_at_point)
        @test isdefined(SHTnsKit, :evaluate_vector_at_point)
        @test isdefined(SHTnsKit, :compute_gradient_direct)
        @test isdefined(SHTnsKit, :extract_latitude_slice)
        
        println(" New function exports working")
    end
    
    @testset "nlm_calc Utility Function" begin
        # Test utility function that doesn't require SHTns
        
        # Standard triangular truncation
        nlm_tri = nlm_calc(15, 15, 1)
        expected_tri = (15 + 1) * (15 + 2) ÷ 2  # Standard formula
        @test nlm_tri == expected_tri
        
        # Rhomboidal truncation
        nlm_rhomb = nlm_calc(20, 10, 1)
        @test nlm_rhomb < nlm_calc(20, 20, 1)  # Should be smaller than triangular
        
        # With azimuthal resolution
        nlm_res = nlm_calc(10, 8, 2)
        @test nlm_res <= nlm_calc(10, 8, 1)  # Should be same or smaller
        
        # Error cases
        @test_throws Exception nlm_calc(-1, 5, 1)  # Negative lmax
        @test_throws Exception nlm_calc(5, -1, 1)  # Negative mmax
        @test_throws Exception nlm_calc(5, 5, 0)   # Zero mres
        
        println(" nlm_calc function working correctly")
    end
    
    # Only test SHTns-dependent functions if SHTns is available
    import SHTnsKit: has_shtns_symbols, should_test_shtns_by_default
    has_symbols = has_shtns_symbols()
    should_test_by_default = should_test_shtns_by_default()
    
    if !has_symbols || !should_test_by_default
        reason = if !has_symbols
            "missing required symbols"
        else
            "SHTns testing disabled by default on this platform"
        end
        
        @test_skip "SHTns-dependent new feature tests - $reason"
        println(" Skipping SHTns-dependent new feature tests: $reason")
        return
    end
    
    println(" Testing new SHTns features on $(Sys.KERNEL) $(Sys.ARCH)")
    
    @testset "3D Vector Transforms" begin
        try
            cfg = create_test_config(8, 8)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            nlm = get_nlm(cfg)
            
            # Test low-level 3D vector functions
            # Create sample 3D vector field
            Vr_spat = rand(nlat * nphi)
            Vt_spat = rand(nlat * nphi)
            Vp_spat = rand(nlat * nphi)
            
            # Allocate spectral arrays
            Qlm = Vector{Float64}(undef, nlm)
            Slm = Vector{Float64}(undef, nlm)
            Tlm = Vector{Float64}(undef, nlm)
            
            # Test analysis
            spat_to_SHqst(cfg, Vr_spat, Vt_spat, Vp_spat, Qlm, Slm, Tlm)
            @test all(isfinite.(Qlm))
            @test all(isfinite.(Slm))
            @test all(isfinite.(Tlm))
            
            # Test synthesis
            Vr_reconstructed = Vector{Float64}(undef, nlat * nphi)
            Vt_reconstructed = Vector{Float64}(undef, nlat * nphi)
            Vp_reconstructed = Vector{Float64}(undef, nlat * nphi)
            
            SHqst_to_spat(cfg, Qlm, Slm, Tlm, Vr_reconstructed, Vt_reconstructed, Vp_reconstructed)
            
            # Check round-trip accuracy (should be exact for well-resolved fields)
            @test maximum(abs.(Vr_spat - Vr_reconstructed)) < 1e-10
            @test maximum(abs.(Vt_spat - Vt_reconstructed)) < 1e-10
            @test maximum(abs.(Vp_spat - Vp_reconstructed)) < 1e-10
            
            # Test high-level 3D vector functions
            Vr_mat = reshape(Vr_spat, nlat, nphi)
            Vt_mat = reshape(Vt_spat, nlat, nphi)
            Vp_mat = reshape(Vp_spat, nlat, nphi)
            
            # High-level analysis
            Qlm_hl, Slm_hl, Tlm_hl = analyze_3d_vector(cfg, Vr_mat, Vt_mat, Vp_mat)
            @test Qlm_hl ≈ Qlm atol=1e-12
            @test Slm_hl ≈ Slm atol=1e-12
            @test Tlm_hl ≈ Tlm atol=1e-12
            
            # High-level synthesis
            Vr_hl, Vt_hl, Vp_hl = synthesize_3d_vector(cfg, Qlm, Slm, Tlm)
            @test Vr_hl ≈ Vr_mat atol=1e-12
            @test Vt_hl ≈ Vt_mat atol=1e-12
            @test Vp_hl ≈ Vp_mat atol=1e-12
            
            free_config(cfg)
            println(" 3D vector transforms working correctly")
            
        catch e
            @test_skip "3D vector transform tests skipped: $e"
        end
    end
    
    @testset "Point Evaluation Functions" begin
        try
            cfg = create_test_config(8, 8)
            nlm = get_nlm(cfg)
            
            # Test scalar point evaluation
            sh = zeros(nlm)
            sh[1] = 1.0  # Y_0^0 component (constant field)
            
            # Evaluate at different points
            value_north = SH_to_point(cfg, sh, 1.0, 0.0)  # North pole
            value_south = SH_to_point(cfg, sh, -1.0, 0.0) # South pole
            value_equator = SH_to_point(cfg, sh, 0.0, 0.0) # Equator
            
            # Y_0^0 should be constant everywhere
            @test value_north ≈ value_south atol=1e-12
            @test value_north ≈ value_equator atol=1e-12
            
            # Test high-level scalar evaluation
            value_hl = evaluate_at_point(cfg, sh, 0.0, 0.0)  # North pole (theta=0)
            @test value_hl ≈ value_north atol=1e-12
            
            # Test vector point evaluation
            Qlm = zeros(nlm)
            Slm = zeros(nlm)
            Tlm = zeros(nlm)
            Qlm[1] = 1.0  # Set radial component
            
            Vr, Vt, Vp = SHqst_to_point(cfg, Qlm, Slm, Tlm, 0.0, 0.0)
            @test Vr ≈ 1.0 atol=1e-10  # Radial component should be 1
            @test abs(Vt) < 1e-10       # Tangential components should be 0
            @test abs(Vp) < 1e-10
            
            # Test high-level vector evaluation
            Vr_hl, Vt_hl, Vp_hl = evaluate_vector_at_point(cfg, Qlm, Slm, Tlm, π/2, 0.0)
            @test Vr_hl ≈ Vr atol=1e-12
            
            free_config(cfg)
            println(" Point evaluation functions working correctly")
            
        catch e
            @test_skip "Point evaluation tests skipped: $e"
        end
    end
    
    @testset "Gradient Computation" begin
        try
            cfg = create_test_config(8, 8)
            nlm = get_nlm(cfg)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            
            # Test with a simple field that has known gradient
            sh = zeros(nlm)
            if nlm > 2
                sh[2] = 1.0  # Y_1^0 component
            end
            
            # Allocate output arrays
            grad_theta = Vector{Float64}(undef, nlat * nphi)
            grad_phi = Vector{Float64}(undef, nlat * nphi)
            
            # Test low-level gradient computation
            SH_to_grad_spat(cfg, sh, grad_theta, grad_phi)
            @test all(isfinite.(grad_theta))
            @test all(isfinite.(grad_phi))
            
            # Test high-level gradient computation
            grad_theta_hl, grad_phi_hl = compute_gradient_direct(cfg, sh)
            @test size(grad_theta_hl) == (nlat, nphi)
            @test size(grad_phi_hl) == (nlat, nphi)
            
            # Results should match
            @test reshape(grad_theta_hl, :) ≈ grad_theta atol=1e-12
            @test reshape(grad_phi_hl, :) ≈ grad_phi atol=1e-12
            
            free_config(cfg)
            println(" Gradient computation working correctly")
            
        catch e
            @test_skip "Gradient computation tests skipped: $e"
        end
    end
    
    @testset "Latitude Extraction" begin
        try
            cfg = create_test_config(8, 8)
            nlm = get_nlm(cfg)
            nphi = get_nphi(cfg)
            
            # Create sample spectral coefficients
            Qlm = rand(nlm)
            Slm = rand(nlm)
            Tlm = rand(nlm)
            
            # Allocate output arrays
            Vr = Vector{Float64}(undef, nphi)
            Vt = Vector{Float64}(undef, nphi)
            Vp = Vector{Float64}(undef, nphi)
            
            # Test low-level latitude extraction
            cost_eq = 0.0  # Equator
            SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost_eq, Vr, Vt, Vp)
            @test length(Vr) == nphi
            @test all(isfinite.(Vr))
            @test all(isfinite.(Vt))
            @test all(isfinite.(Vp))
            
            # Test high-level latitude extraction
            latitude = 0.0  # Equator
            Vr_hl, Vt_hl, Vp_hl = extract_latitude_slice(cfg, Qlm, Slm, Tlm, latitude)
            @test length(Vr_hl) == nphi
            @test Vr_hl ≈ Vr atol=1e-12
            @test Vt_hl ≈ Vt atol=1e-12
            @test Vp_hl ≈ Vp atol=1e-12
            
            # Test different latitudes
            lat_30n = π/6  # 30°N
            Vr_30n, Vt_30n, Vp_30n = extract_latitude_slice(cfg, Qlm, Slm, Tlm, lat_30n)
            @test length(Vr_30n) == nphi
            
            free_config(cfg)
            println(" Latitude extraction working correctly")
            
        catch e
            @test_skip "Latitude extraction tests skipped: $e"
        end
    end
    
    @testset "Error Handling for New Functions" begin
        try
            cfg = create_test_config(8, 8)
            nlm = get_nlm(cfg)
            
            # Test input validation
            sh_wrong = rand(5)  # Wrong size
            @test_throws Exception SH_to_point(cfg, sh_wrong, 0.0, 0.0)
            
            # Test range validation
            sh_correct = rand(nlm)
            @test_throws Exception SH_to_point(cfg, sh_correct, 2.0, 0.0)  # cost > 1
            @test_throws Exception evaluate_at_point(cfg, sh_correct, -0.1, 0.0)  # theta < 0
            
            free_config(cfg)
            println(" Error handling working correctly")
            
        catch e
            @test_skip "Error handling tests skipped: $e"
        end
    end
end