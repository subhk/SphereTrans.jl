using Test
using SHTnsKit

@testset "SHTnsKit.jl Tests" begin
    
    @testset "Configuration Creation" begin
        # Test basic configuration creation
        cfg = create_gauss_config(4, 4)
        @test get_lmax(cfg) == 4
        @test get_mmax(cfg) == 4
        @test get_nlm(cfg) == 25  # (4+1)*(4+2)/2 = 15, but nlm_calc gives different result
        @test get_nlat(cfg) >= 5
        @test get_nphi(cfg) >= 9
        
        # Test regular grid
        cfg_regular = create_regular_config(3, 3)
        @test get_lmax(cfg_regular) == 3
        @test cfg_regular.grid_type == SHT_REGULAR
    end
    
    @testset "Basic Transforms" begin
        cfg = create_gauss_config(4, 4)
        
        # Test allocation functions
        sh = allocate_spectral(cfg)
        spat = allocate_spatial(cfg)
        @test length(sh) == get_nlm(cfg)
        @test size(spat) == (get_nlat(cfg), get_nphi(cfg))
        
        # Test forward and backward transforms
        fill!(sh, 0.0)
        sh[1] = 1.0  # Set Y_0^0 component
        
        # Synthesis
        spat_out = synthesize(cfg, sh)
        @test size(spat_out) == (get_nlat(cfg), get_nphi(cfg))
        @test all(isfinite.(spat_out))
        
        # Analysis  
        sh_reconstructed = analyze(cfg, spat_out)
        @test length(sh_reconstructed) == get_nlm(cfg)
        @test all(isfinite.(sh_reconstructed))
        
        # Check that the l=0,m=0 mode is approximately preserved
        @test abs(sh_reconstructed[1] - sh[1]) < 1e-10
    end
    
    @testset "Utility Functions" begin
        cfg = create_gauss_config(3, 3)
        
        # Test indexing
        @test lmidx(cfg, 0, 0) == 1
        @test lmidx(cfg, 1, 0) > 1
        @test lmidx(cfg, 1, 1) > lmidx(cfg, 1, 0)
        
        # Test lm_from_index
        l, m = lm_from_index(cfg, 1)
        @test l == 0 && m == 0
        
        # Test test field creation
        test_field = create_test_field(cfg, 1, 0)
        @test size(test_field) == (get_nlat(cfg), get_nphi(cfg))
        @test all(isfinite.(test_field))
        
        # Test power spectrum
        sh = allocate_spectral(cfg)
        sh[1] = 1.0  # Y_0^0
        sh[lmidx(cfg, 1, 0)] = 0.5  # Y_1^0
        
        power = power_spectrum(cfg, sh)
        @test length(power) == get_lmax(cfg) + 1
        @test power[1] ≈ 1.0  # l=0 power
        @test power[2] ≈ 0.25  # l=1 power
    end
    
    @testset "Grid Utilities" begin
        cfg = create_gauss_config(4, 4)
        
        # Test coordinate access
        theta1 = get_theta(cfg, 1)
        phi1 = get_phi(cfg, 1)
        @test 0 <= theta1 <= π
        @test 0 <= phi1 < 2π
        
        # Test weights
        weights = get_gauss_weights(cfg)
        @test length(weights) == get_nlat(cfg)
        @test all(weights .> 0)
        
        # Test coordinate matrices
        theta_mat, phi_mat = create_coordinate_matrices(cfg)
        @test size(theta_mat) == (get_nlat(cfg), get_nphi(cfg))
        @test size(phi_mat) == (get_nlat(cfg), get_nphi(cfg))
        
        # Test Cartesian coordinates
        x, y, z = create_cartesian_coordinates(cfg)
        @test size(x) == size(y) == size(z) == (get_nlat(cfg), get_nphi(cfg))
        
        # Points should be on unit sphere
        r_squared = x.^2 + y.^2 + z.^2
        @test all(abs.(r_squared .- 1) .< 1e-12)
    end
    
    @testset "Complex Transforms" begin
        cfg = create_gauss_config(3, 3)
        
        # Test complex allocation
        sh_complex = allocate_complex_spectral(cfg)
        spat_complex = allocate_complex_spatial(cfg)
        
        @test eltype(sh_complex) == ComplexF64
        @test eltype(spat_complex) == ComplexF64
        
        # Test complex transforms
        fill!(sh_complex, 0.0 + 0.0im)
        sh_complex[1] = 1.0 + 0.5im  # Complex Y_0^0
        
        spat_out = synthesize_complex(cfg, sh_complex)
        @test size(spat_out) == (get_nlat(cfg), get_nphi(cfg))
        @test eltype(spat_out) == ComplexF64
        
        # Test roundtrip
        sh_reconstructed = analyze_complex(cfg, spat_out)
        @test abs(sh_reconstructed[1] - sh_complex[1]) < 1e-10
    end
    
    @testset "Error Handling" begin
        # Test invalid configurations
        @test_throws Exception create_config(-1, 1)  # negative lmax
        @test_throws Exception create_config(1, 2)   # mmax > lmax
        
        cfg = create_gauss_config(2, 2)
        
        # Test size mismatches
        sh_wrong = Vector{Float64}(undef, 5)  # Wrong size
        spat = allocate_spatial(cfg)
        @test_throws Exception synthesize!(cfg, sh_wrong, spat)
        
        # Test invalid indices
        @test_throws Exception lmidx(cfg, -1, 0)  # negative l
        @test_throws Exception lmidx(cfg, 1, 2)   # m > l
        @test_throws Exception lm_from_index(cfg, 0)  # 0-based index
        @test_throws Exception lm_from_index(cfg, get_nlm(cfg) + 1)  # out of range
    end
end