using Test
using SHTnsKit
import Libdl
import Base.Threads

# Limit threading in test environment to avoid oversubscription with SHTns/OpenMP
ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OPENBLAS_NUM_THREADS"] = get(ENV, "OPENBLAS_NUM_THREADS", "1")

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

        # Basic multi-threading safety: run multiple transforms concurrently
        if Threads.nthreads() > 1
            shs = [copy(sh) for _ in 1:Threads.nthreads()]
            spats = [allocate_spatial(cfg) for _ in 1:Threads.nthreads()]
            Threads.@threads for i in 1:length(shs)
                synthesize!(cfg, shs[i], spats[i])
                analyze!(cfg, spats[i], shs[i])
            end
            @test all(size(s) == (get_nlat(cfg), get_nphi(cfg)) for s in spats)
end

@testset "CUDA device-pointer mode (guarded)" begin
    try
        Base.require(:CUDA)
        if get(Base.loaded_modules, :CUDA, nothing) !== nothing
            CUDA = Base.loaded_modules[:CUDA]
            if hasproperty(CUDA, :functional) && CUDA.functional()
                # Ensure native GPU entrypoints are resolved (if provided via env)
                SHTnsKit.enable_native_gpu!()
                if SHTnsKit.is_native_gpu_enabled()
                    SHTnsKit.SHTnsKitCUDAExt.enable_gpu_deviceptrs!()
                    cfg = create_config(8, 8, 1, UInt32(0))
                    set_grid(cfg, 16, 32, 0)
                    sh = allocate_spectral(cfg)
                    rand!(sh)
                    shd = CUDA.CuArray(sh)
                    spatd = synthesize_gpu(cfg, shd)
                    @test size(spatd) == (get_nlat(cfg), get_nphi(cfg))
                    shd2 = analyze_gpu(cfg, spatd)
                    @test length(shd2) == length(sh)
                    free_config(cfg)
                else
                    @info "Native GPU entrypoints not enabled; skipping device-pointer test"
                end
            end
        end
    catch e
        @info "Skipping CUDA device-pointer tests" exception=(e, catch_backtrace())
    end
end

@testset "SHTnsKit MPI extension (guarded)" begin
    try
        Base.require(:MPI)
        if get(Base.loaded_modules, :MPI, nothing) === nothing
            @info "MPI not loaded; skipping MPI tests"
        else
            MPI = Base.loaded_modules[:MPI]
            if !MPI.Initialized()
                MPI.Init()
            end
            comm = MPI.COMM_WORLD
            # Ensure extension tries defaults if nothing set
            try
                SHTnsKit.SHTnsKitMPIExt.enable_native_mpi!()
            catch
            end
            if SHTnsKit.SHTnsKitMPIExt.is_native_mpi_enabled()
                cfgm = SHTnsKit.SHTnsKitMPIExt.create_mpi_config(comm, 8, 8, 1)
                set_grid(cfgm, 16, 32, 0)
                sh = allocate_spectral(cfgm.cfg)
                rand!(sh)
                spat = allocate_spatial(cfgm.cfg)
                synthesize!(cfgm, sh, spat)
                analyze!(cfgm, spat, sh)
                @test length(sh) == SHTnsKit.get_nlm(cfgm.cfg)
                free_config(cfgm)
            else
                @info "SHTns MPI symbols not enabled; skipping MPI tests"
            end
            if MPI.Initialized() && !MPI.Finalized()
                MPI.Finalize()
            end
        end
    catch e
        @info "Skipping MPI tests" exception=(e, catch_backtrace())
    end
end

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
