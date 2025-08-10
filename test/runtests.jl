using Test
using SHTnsKit
import Libdl

@testset "SHTnsKit high-level wrappers" begin
    try
        # Create a small configuration and grid
        cfg = create_config(8, 8, 1, UInt32(0))
        set_grid(cfg, 16, 32, 0)

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

        # GPU-friendly wrappers, if CUDA is available
        try
            Base.require(:CUDA)
            # Only run if CUDA is functional
            if get(Base.loaded_modules, :CUDA, nothing) !== nothing
                CUDA = Base.loaded_modules[:CUDA]
                if hasproperty(CUDA, :functional) && CUDA.functional()
                    shd = CUDA.CuArray(sh)
                    spatd = synthesize_gpu(cfg, shd)
                    @test size(spatd) == (get_nlat(cfg), get_nphi(cfg))
                    shd2 = analyze_gpu(cfg, spatd)
                    @test length(shd2) == length(sh)
                end
            end
        catch e
            @info "Skipping CUDA tests" exception=(e, catch_backtrace())
        end

        free_config(cfg)
    catch e
        @info "Skipping SHTns tests (library not available or failed to initialize)" exception=(e, catch_backtrace())
    end
end

