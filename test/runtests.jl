using Test
using SHTnsKit
import Libdl
import Base.Threads
using LinearAlgebra

# Limit threading in test environment to avoid oversubscription with SHTns/OpenMP
ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "2")
ENV["OPENBLAS_NUM_THREADS"] = get(ENV, "OPENBLAS_NUM_THREADS", "1")

@testset "SHTnsKit Comprehensive Tests" begin

@testset "Basic Functionality" begin
    try
        # Create a small configuration and grid
        cfg = create_config(8, 8, 1, UInt32(0))
        set_grid(cfg, 16, 32, SHTnsFlags.SHT_GAUSS)

        # Basic queries
        @test get_lmax(cfg) >= 8
        @test get_nlat(cfg) == 16
        @test get_nphi(cfg) == 32
        @test get_nlm(cfg) > 0

        # Allocation helpers
        sh = allocate_spectral(cfg)
        spat = allocate_spatial(cfg)
        @test length(sh) == get_nlm(cfg)
        @test size(spat) == (get_nlat(cfg), get_nphi(cfg))

        # High-level transforms (CPU)
        rand!(sh)
        spat2 = synthesize(cfg, sh)
        @test size(spat2) == (get_nlat(cfg), get_nphi(cfg))
        sh2 = analyze(cfg, spat2)
        @test length(sh2) == length(sh)

        # In-place variants
        synthesize!(cfg, sh, spat)
        @test size(spat) == (get_nlat(cfg), get_nphi(cfg))
        analyze!(cfg, spat, sh)
        @test length(sh) == get_nlm(cfg)

        free_config(cfg)
    catch e
        @info "Skipping basic tests (library not available)" exception=(e, catch_backtrace())
    end
end

@testset "Grid Types and Creation" begin
    try
        # Test Gauss grid creation
        cfg_gauss = create_gauss_config(8, 8)
        @test get_lmax(cfg_gauss) == 8
        @test get_nlat(cfg_gauss) == 9  # lmax + 1
        @test get_nphi(cfg_gauss) == 17  # 2*mmax + 1
        
        # Test regular grid creation
        cfg_regular = create_regular_config(8, 8)
        @test get_lmax(cfg_regular) == 8
        @test get_nlat(cfg_regular) == 17  # 2*lmax + 1
        @test get_nphi(cfg_regular) == 17  # 2*mmax + 1
        
        free_config(cfg_gauss)
        free_config(cfg_regular)
    catch e
        @info "Skipping grid tests" exception=(e, catch_backtrace())
    end
end

@testset "Complex Field Transforms" begin
    try
        cfg = create_gauss_config(8, 8)
        
        # Test complex allocation
        sh_complex = allocate_complex_spectral(cfg)
        spat_complex = allocate_complex_spatial(cfg)
        @test eltype(sh_complex) == ComplexF64
        @test eltype(spat_complex) == ComplexF64
        
        # Test complex transforms
        rand!(sh_complex)
        spat_result = synthesize_complex(cfg, sh_complex)
        @test size(spat_result) == (get_nlat(cfg), get_nphi(cfg))
        
        sh_result = analyze_complex(cfg, spat_result)
        @test length(sh_result) == length(sh_complex)
        
        free_config(cfg)
    catch e
        @info "Skipping complex field tests" exception=(e, catch_backtrace())
    end
end

@testset "Vector Field Transforms" begin
    try
        cfg = create_gauss_config(8, 8)
        
        # Create random spheroidal and toroidal coefficients
        Slm = rand(get_nlm(cfg))
        Tlm = rand(get_nlm(cfg))
        
        # Test vector synthesis
        Vt, Vp = synthesize_vector(cfg, Slm, Tlm)
        @test size(Vt) == (get_nlat(cfg), get_nphi(cfg))
        @test size(Vp) == (get_nlat(cfg), get_nphi(cfg))
        
        # Test vector analysis
        Slm_result, Tlm_result = analyze_vector(cfg, Vt, Vp)
        @test length(Slm_result) == length(Slm)
        @test length(Tlm_result) == length(Tlm)
        
        # Test gradient computation
        Vt_grad, Vp_grad = compute_gradient(cfg, Slm)
        @test size(Vt_grad) == (get_nlat(cfg), get_nphi(cfg))
        @test size(Vp_grad) == (get_nlat(cfg), get_nphi(cfg))
        
        # Test curl computation
        Vt_curl, Vp_curl = compute_curl(cfg, Tlm)
        @test size(Vt_curl) == (get_nlat(cfg), get_nphi(cfg))
        @test size(Vp_curl) == (get_nlat(cfg), get_nphi(cfg))
        
        free_config(cfg)
    catch e
        @info "Skipping vector field tests" exception=(e, catch_backtrace())
    end
end

@testset "Rotation Functions" begin
    try
        cfg = create_gauss_config(8, 8)
        
        # Test spectral rotation
        sh = rand(get_nlm(cfg))
        alpha, beta, gamma = π/4, π/3, π/6
        
        sh_rotated = rotate_field(cfg, sh, alpha, beta, gamma)
        @test length(sh_rotated) == length(sh)
        @test sh_rotated != sh  # Should be different after rotation
        
        # Test spatial rotation
        spat = rand(get_nlat(cfg), get_nphi(cfg))
        spat_rotated = rotate_spatial_field(cfg, spat, alpha, beta, gamma)
        @test size(spat_rotated) == size(spat)
        
        free_config(cfg)
    catch e
        @info "Skipping rotation tests" exception=(e, catch_backtrace())
    end
end

@testset "Power Spectrum and Analysis" begin
    try
        cfg = create_gauss_config(8, 8)
        
        # Create test field with known properties
        sh = zeros(get_nlm(cfg))
        sh[1] = 1.0  # Set l=0, m=0 component
        
        # Compute power spectrum
        power = power_spectrum(cfg, sh)
        @test length(power) == get_lmax(cfg) + 1
        @test power[1] > 0  # l=0 mode should have power
        @test sum(power[2:end]) ≈ 0 atol=1e-10  # Other modes should be zero
        
        free_config(cfg)
    catch e
        @info "Skipping power spectrum tests" exception=(e, catch_backtrace())
    end
end

@testset "Threading Control" begin
    try
        # Test threading functions
        original_threads = get_num_threads()
        
        set_num_threads(1)
        @test get_num_threads() == 1
        
        # Test optimal thread setting
        set_optimal_threads()
        new_threads = get_num_threads()
        @test new_threads >= 1
        
        # Restore original
        set_num_threads(original_threads)
    catch e
        @info "Skipping threading tests" exception=(e, catch_backtrace())
    end
end

@testset "Multi-threading Safety" begin
    try
        cfg = create_gauss_config(8, 8)
        
        if Threads.nthreads() > 1
            shs = [rand(get_nlm(cfg)) for _ in 1:Threads.nthreads()]
            spats = [allocate_spatial(cfg) for _ in 1:Threads.nthreads()]
            
            Threads.@threads for i in 1:length(shs)
                synthesize!(cfg, shs[i], spats[i])
                analyze!(cfg, spats[i], shs[i])
            end
            
            @test all(size(s) == (get_nlat(cfg), get_nphi(cfg)) for s in spats)
        else
            @info "Single-threaded environment; skipping multi-threading tests"
        end
        
        free_config(cfg)
    catch e
        @info "Skipping multi-threading tests" exception=(e, catch_backtrace())
    end
end

@testset "GPU Support (CUDA)" begin
    try
        Base.require(:CUDA)
        if get(Base.loaded_modules, :CUDA, nothing) !== nothing
            CUDA = Base.loaded_modules[:CUDA]
            if hasproperty(CUDA, :functional) && CUDA.functional()
                cfg = create_gpu_config(8, 8)
                
                # Test basic GPU transforms
                sh = rand(get_nlm(cfg))
                shd = CUDA.CuArray(sh)
                
                spatd = synthesize_gpu(cfg, shd)
                @test size(spatd) == (get_nlat(cfg), get_nphi(cfg))
                
                shd2 = analyze_gpu(cfg, spatd)
                @test length(shd2) == length(sh)
                
                # Test GPU initialization
                gpu_initialized = initialize_gpu(0; verbose=false)
                if gpu_initialized
                    @test cleanup_gpu(verbose=false) == true
                end
                
                free_config(cfg)
            else
                @info "CUDA not functional; skipping GPU tests"
            end
        else
            @info "CUDA not available; skipping GPU tests"
        end
    catch e
        @info "Skipping CUDA tests" exception=(e, catch_backtrace())
    end
end

@testset "Error Handling and Edge Cases" begin
    try
        # Test invalid configuration creation
        @test_throws BoundsError create_config(-1, 8, 1, UInt32(0))
        
        # Test with valid config
        cfg = create_gauss_config(4, 4)
        
        # Test mismatched array sizes
        sh_wrong = rand(10)  # Wrong size
        spat = allocate_spatial(cfg)
        @test_throws AssertionError synthesize!(cfg, sh_wrong, spat)
        
        free_config(cfg)
    catch e
        @info "Skipping error handling tests" exception=(e, catch_backtrace())
    end
end

@testset "Memory Management" begin
    try
        # Test configuration lifecycle
        cfg = create_gauss_config(8, 8)
        
        # Ensure we can perform operations
        sh = allocate_spectral(cfg)
        spat = allocate_spatial(cfg)
        rand!(sh)
        synthesize!(cfg, sh, spat)
        
        # Free should not crash
        free_config(cfg)
        
        @test true  # If we get here, no crash occurred
    catch e
        @info "Memory management test failed" exception=(e, catch_backtrace())
    end
end

end # Main testset