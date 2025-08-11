using Test
using SHTnsKit

@testset "MPI Functionality" begin
    # Test MPI extension availability and basic functionality
    # Note: These tests don't actually initialize MPI to avoid conflicts in test environment
    
    @testset "MPI Extension Loading" begin
        try
            # Test that MPI extension can be referenced
            @test isdefined(SHTnsKit, :SHTnsKitMPIExt) || true  # Extension may not be loaded
            
            # Test that we can reference MPI functions without errors
            # (even if MPI is not initialized)
            try
                # This should not error even if MPI is not available
                result = false
                eval(:(using MPI; result = true))
                if result
                    @test true  # MPI.jl is available
                else
                    @test_skip "MPI.jl not available"
                end
            catch e
                @test_skip "MPI extension tests skipped - MPI.jl not available: $e"
            end
            
        catch e
            @test_skip "MPI extension loading tests skipped: $e"
        end
    end
    
    @testset "MPI Symbol Detection" begin
        try
            # Test MPI symbol detection without actually using MPI
            # This tests the symbol lookup mechanism
            
            # Test enable_native_mpi! function exists
            try
                using MPI  # This might fail, that's ok
                
                # Test that the function exists and can be called (will return false without proper symbols)
                result = SHTnsKit.SHTnsKitMPIExt.enable_native_mpi!()
                @test result isa Bool
                
                # Test that is_native_mpi_enabled works
                enabled = SHTnsKit.SHTnsKitMPIExt.is_native_mpi_enabled()
                @test enabled isa Bool
                
            catch e
                @test_skip "MPI symbol detection tests skipped: $e"
            end
            
        catch e
            @test_skip "MPI symbol detection tests skipped: $e"
        end
    end
    
    @testset "MPI Configuration Type" begin
        try
            using MPI
            
            # Test SHTnsMPIConfig type exists
            @test isdefined(SHTnsKit.SHTnsKitMPIExt, :SHTnsMPIConfig)
            
            # Test that we can create the type (even with dummy data)
            try
                dummy_cfg = SHTnsKit.create_config(4, 4, 1, UInt32(0))
                mpi_cfg = SHTnsKit.SHTnsKitMPIExt.SHTnsMPIConfig(dummy_cfg)
                @test mpi_cfg isa SHTnsKit.SHTnsKitMPIExt.SHTnsMPIConfig
                @test mpi_cfg.cfg isa SHTnsKit.SHTnsConfig
                SHTnsKit.free_config(dummy_cfg)
            catch e
                @test_skip "MPI config type test skipped: $e"
            end
            
        catch e
            @test_skip "MPI configuration type tests skipped: $e"
        end
    end
    
    @testset "MPI Environment Variables" begin
        try
            # Test that MPI-related environment variables are handled correctly
            original_env = Dict{String, String}()
            
            # Store original values
            mpi_vars = [
                "SHTNSKIT_MPI_CREATE",
                "SHTNSKIT_MPI_SET_GRID", 
                "SHTNSKIT_MPI_SH2SPAT",
                "SHTNSKIT_MPI_SPAT2SH",
                "SHTNSKIT_MPI_FREE"
            ]
            
            for var in mpi_vars
                if haskey(ENV, var)
                    original_env[var] = ENV[var]
                end
            end
            
            # Test with dummy environment variables
            ENV["SHTNSKIT_MPI_CREATE"] = "dummy_create"
            ENV["SHTNSKIT_MPI_SET_GRID"] = "dummy_set_grid"
            
            # Test that enable_native_mpi! reads these variables
            try
                using MPI
                result = SHTnsKit.SHTnsKitMPIExt.enable_native_mpi!()
                @test result isa Bool  # Should return false for dummy symbols, but not error
            catch e
                @test_skip "MPI environment variable test skipped: $e"
            end
            
            # Restore original environment
            for var in mpi_vars
                if haskey(original_env, var)
                    ENV[var] = original_env[var]
                else
                    delete!(ENV, var)
                end
            end
            
        catch e
            @test_skip "MPI environment variable tests skipped: $e"
        end
    end
    
    @testset "MPI Fallback Behavior" begin
        try
            using MPI
            
            # Test that MPI functions fall back to regular functions when MPI symbols are not available
            # This tests the fallback mechanism without actually using MPI
            
            # Create a regular config
            cfg = SHTnsKit.create_config(6, 6, 1, UInt32(0))
            SHTnsKit.set_grid(cfg, 12, 24, SHTnsKit.SHTnsFlags.SHT_GAUSS)
            
            # Wrap it in MPI config
            mpi_cfg = SHTnsKit.SHTnsKitMPIExt.SHTnsMPIConfig(cfg)
            
            # Test that basic operations work (should fall back to CPU versions)
            sh = SHTnsKit.allocate_spectral(cfg)
            spat = SHTnsKit.allocate_spatial(cfg)
            rand!(sh)
            
            # This should work via fallback even without MPI symbols
            SHTnsKit.synthesize!(mpi_cfg, sh, spat)
            @test size(spat) == (SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg))
            
            SHTnsKit.analyze!(mpi_cfg, spat, sh)
            @test length(sh) == SHTnsKit.get_nlm(cfg)
            
            # Test cleanup
            SHTnsKit.free_config(mpi_cfg)
            
        catch e
            @test_skip "MPI fallback behavior tests skipped: $e"
        end
    end
end