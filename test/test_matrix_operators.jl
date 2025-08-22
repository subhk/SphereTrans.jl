using Test
using SHTnsKit
using LinearAlgebra

@testset "Matrix Operators" begin
    lmax, mmax = 6, 4
    nlat, nphi = 20, 32  
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "cos(θ) multiplication matrix" begin
        matrix = mul_ct_matrix(cfg)
        
        @test size(matrix) == (cfg.nlm, cfg.nlm)
        @test issymmetric(matrix)  # cos(θ) operator should be symmetric
        
        # Test that matrix couples only l±1 terms
        for i in 1:cfg.nlm, j in 1:cfg.nlm  
            l_i, m_i = cfg.lm_indices[i]
            l_j, m_j = cfg.lm_indices[j]
            
            if abs(l_i - l_j) > 1 || m_i != m_j
                @test abs(matrix[i, j]) < 1e-12  # Should be zero
            end
        end
        
        # Test application to Y_0^0 (constant function)
        qlm_in = zeros(Complex{Float64}, cfg.nlm)
        qlm_in[1] = 1.0  # Y_0^0 coefficient
        
        qlm_out = similar(qlm_in)
        sh_mul_mx(cfg, matrix, qlm_in, qlm_out)
        
        # cos(θ) * Y_0^0 should give Y_1^0 component  
        idx_10 = findfirst(i -> cfg.lm_indices[i] == (1, 0), 1:cfg.nlm)
        if idx_10 !== nothing
            @test abs(real(qlm_out[idx_10])) > 1e-12
            @test abs(imag(qlm_out[idx_10])) < 1e-12
        end
    end
    
    @testset "sin(θ)d/dθ matrix" begin
        matrix = st_dt_matrix(cfg)
        
        @test size(matrix) == (cfg.nlm, cfg.nlm)
        
        # Test that matrix couples only l±1 terms with same m
        for i in 1:cfg.nlm, j in 1:cfg.nlm
            l_i, m_i = cfg.lm_indices[i] 
            l_j, m_j = cfg.lm_indices[j]
            
            if abs(l_i - l_j) > 1 || m_i != m_j
                @test abs(matrix[i, j]) < 1e-12
            end
        end
    end
    
    @testset "Laplacian operator" begin
        matrix = laplacian_matrix(cfg)
        
        @test size(matrix) == (cfg.nlm, cfg.nlm)
        @test isdiag(matrix)  # Laplacian should be diagonal in SH basis
        
        # Check eigenvalues are -l(l+1)
        for i in 1:cfg.nlm
            l, m = cfg.lm_indices[i]
            expected_eigenvalue = -l * (l + 1)
            @test abs(matrix[i, i] - expected_eigenvalue) < 1e-12
        end
        
        # Test application function
        qlm = randn(Complex{Float64}, cfg.nlm)
        result = apply_laplacian(cfg, qlm)
        
        @test length(result) == length(qlm)
        
        # Verify result matches matrix application  
        result_matrix = matrix * qlm
        @test norm(result - result_matrix) < 1e-12
    end
    
    @testset "Matrix-vector multiplication" begin  
        # Test general matrix-vector multiplication
        matrix = randn(cfg.nlm, cfg.nlm)
        qlm_in = randn(Complex{Float64}, cfg.nlm)
        qlm_out = zeros(Complex{Float64}, cfg.nlm)
        
        sh_mul_mx(cfg, matrix, qlm_in, qlm_out)
        
        # Compare with direct matrix multiplication
        expected = matrix * qlm_in
        @test norm(qlm_out - expected) < 1e-12
    end
    
    @testset "Convenience functions" begin
        qlm = randn(Complex{Float64}, cfg.nlm)
        
        # Test cos(θ) operator
        result_ct = apply_costheta_operator(cfg, qlm)
        @test length(result_ct) == length(qlm)
        @test result_ct !== qlm  # Should be different array
        
        # Test sin(θ)d/dθ operator
        result_st = apply_sintdtheta_operator(cfg, qlm)
        @test length(result_st) == length(qlm) 
        @test result_st !== qlm
        
        # Results should be different (unless input has special symmetry)
        @test norm(result_ct - result_st) > 1e-12
    end
end

@testset "Operator Properties" begin
    lmax, mmax = 4, 4
    nlat, nphi = 16, 24
    cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
    
    @testset "Matrix sparsity" begin
        ct_matrix = mul_ct_matrix(cfg)
        st_matrix = st_dt_matrix(cfg)
        
        # Count non-zero elements - should be sparse
        ct_nnz = count(x -> abs(x) > 1e-12, ct_matrix)
        st_nnz = count(x -> abs(x) > 1e-12, st_matrix)
        
        total_elements = cfg.nlm^2
        
        # Matrices should be much sparser than full
        @test ct_nnz < total_elements / 3
        @test st_nnz < total_elements / 3
    end
    
    @testset "Operator relations" begin
        # Test some basic operator relationships
        ct_matrix = mul_ct_matrix(cfg)
        lapl_matrix = laplacian_matrix(cfg)
        
        # Laplacian should commute with cos(θ) (both are rotationally invariant)
        # This is a simplified test - full commutation might need more careful setup
        qlm = randn(Complex{Float64}, cfg.nlm)
        
        result1 = apply_laplacian(cfg, apply_costheta_operator(cfg, qlm))
        result2 = apply_costheta_operator(cfg, apply_laplacian(cfg, qlm))
        
        # They should be close (exact commutation depends on boundary conditions)
        @test norm(result1 - result2) / norm(result1) < 0.1
    end
end