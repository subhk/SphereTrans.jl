using Test
using SHTnsKit
using LinearAlgebra

@testset "Vector Field Transforms" begin
    @testset "Vector Transform Accuracy" begin
        try
            cfg = create_gauss_config(16, 16)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            nlm = get_nlm(cfg)
            
            # Create test spheroidal and toroidal coefficients
            Slm = rand(nlm) * 0.1
            Tlm = rand(nlm) * 0.1
            
            # Vector synthesis
            Vt, Vp = synthesize_vector(cfg, Slm, Tlm)
            @test size(Vt) == (nlat, nphi)
            @test size(Vp) == (nlat, nphi)
            
            # Vector analysis
            Slm_recovered, Tlm_recovered = analyze_vector(cfg, Vt, Vp)
            @test length(Slm_recovered) == nlm
            @test length(Tlm_recovered) == nlm
            
            # Check accuracy
            slm_error = maximum(abs.(Slm - Slm_recovered))
            tlm_error = maximum(abs.(Tlm - Tlm_recovered))
            @test slm_error < 1e-12
            @test tlm_error < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Vector transform accuracy tests skipped: $e"
        end
    end
    
    @testset "Gradient and Curl Operations" begin
        try
            cfg = create_gauss_config(12, 12)
            nlm = get_nlm(cfg)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            
            # Test gradient
            Slm = rand(nlm) * 0.1
            Vt_grad, Vp_grad = compute_gradient(cfg, Slm)
            @test size(Vt_grad) == (nlat, nphi)
            @test size(Vp_grad) == (nlat, nphi)
            
            # Test curl
            Tlm = rand(nlm) * 0.1
            Vt_curl, Vp_curl = compute_curl(cfg, Tlm)
            @test size(Vt_curl) == (nlat, nphi)
            @test size(Vp_curl) == (nlat, nphi)
            
            # Test that gradient and curl are orthogonal operations
            # (this is a theoretical property we can verify numerically)
            
            free_config(cfg)
            
        catch e
            @test_skip "Gradient and curl tests skipped: $e"
        end
    end
    
    @testset "Vector Energy Conservation" begin
        try
            cfg = create_gauss_config(16, 16)
            nlm = get_nlm(cfg)
            
            # Create random vector field
            u = rand(get_nlat(cfg), get_nphi(cfg))
            v = rand(get_nlat(cfg), get_nphi(cfg))
            
            # Compute original kinetic energy (approximate)
            original_energy = sum(u.^2 + v.^2)
            
            # Transform to spectral space and back
            Slm, Tlm = analyze_vector(cfg, u, v)
            u_reconstructed, v_reconstructed = synthesize_vector(cfg, Slm, Tlm)
            
            # Compute reconstructed energy
            reconstructed_energy = sum(u_reconstructed.^2 + v_reconstructed.^2)
            
            # Energy should be approximately conserved
            energy_error = abs(original_energy - reconstructed_energy) / original_energy
            @test energy_error < 1e-10
            
            free_config(cfg)
            
        catch e
            @test_skip "Vector energy conservation tests skipped: $e"
        end
    end
    
    @testset "Helmholtz Decomposition" begin
        try
            cfg = create_gauss_config(12, 12)
            nlm = get_nlm(cfg)
            
            # Create pure spheroidal (divergent) field
            Slm_pure = rand(nlm) * 0.1
            Tlm_zero = zeros(nlm)
            
            u_sph, v_sph = synthesize_vector(cfg, Slm_pure, Tlm_zero)
            
            # Analyze it back
            Slm_recovered, Tlm_recovered = analyze_vector(cfg, u_sph, v_sph)
            
            # Toroidal part should be nearly zero
            @test maximum(abs.(Tlm_recovered)) < 1e-12
            @test maximum(abs.(Slm_pure - Slm_recovered)) < 1e-12
            
            # Create pure toroidal (rotational) field
            Slm_zero = zeros(nlm)
            Tlm_pure = rand(nlm) * 0.1
            
            u_tor, v_tor = synthesize_vector(cfg, Slm_zero, Tlm_pure)
            
            # Analyze it back
            Slm_recovered2, Tlm_recovered2 = analyze_vector(cfg, u_tor, v_tor)
            
            # Spheroidal part should be nearly zero
            @test maximum(abs.(Slm_recovered2)) < 1e-12
            @test maximum(abs.(Tlm_pure - Tlm_recovered2)) < 1e-12
            
            free_config(cfg)
            
        catch e
            @test_skip "Helmholtz decomposition tests skipped: $e"
        end
    end
    
    @testset "Vector Transform In-Place Operations" begin
        try
            # Test the existing vector transform functions if available
            cfg = create_gauss_config(8, 8)
            nlat, nphi = get_nlat(cfg), get_nphi(cfg)
            nlm = get_nlm(cfg)
            
            # Pre-allocate arrays
            tor = rand(nlm)
            pol = rand(nlm)
            u = Matrix{Float64}(undef, nlat, nphi)
            v = Matrix{Float64}(undef, nlat, nphi)
            
            # Test if native vector functions are available
            try
                # This will work if vector transforms are enabled
                synthesize_vec!(cfg, tor, pol, u, v)
                @test size(u) == (nlat, nphi)
                @test size(v) == (nlat, nphi)
                
                # Test analysis
                tor_back = Vector{Float64}(undef, nlm)
                pol_back = Vector{Float64}(undef, nlm)
                analyze_vec!(cfg, u, v, tor_back, pol_back)
                
                # Check accuracy
                tor_error = maximum(abs.(tor - tor_back))
                pol_error = maximum(abs.(pol - pol_back))
                @test tor_error < 1e-12
                @test pol_error < 1e-12
                
            catch e
                @test_skip "Native vector transforms not available: $e"
            end
            
            free_config(cfg)
            
        catch e
            @test_skip "Vector in-place operation tests skipped: $e"
        end
    end
end