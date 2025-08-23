"""
Parallel SHTnsKit Example

This example demonstrates how to use SHTnsKit with MPI for parallel
spherical harmonic transforms.

To run this example:
1. Install required packages:
   julia> using Pkg
   julia> Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])

2. Run with MPI:
   mpiexec -n 4 julia --project=. parallel_example.jl

3. Or run serially (will use fallback implementations):
   julia --project=. parallel_example.jl
"""

using SHTnsKit
using LinearAlgebra

# Optional MPI packages - will use extensions if available
try
    using MPI
    using PencilArrays  
    using PencilFFTs
    println("Parallel packages loaded successfully")
    MPI.Init()
    PARALLEL_AVAILABLE = true
catch e
    println("WARNING: Parallel packages not available: $e")
    println("  Running in serial mode")
    PARALLEL_AVAILABLE = false
end

function run_parallel_example()
    println("="^60)
    println("SHTnsKit Parallel Example")
    println("="^60)
    
    # Problem setup
    lmax, mmax = 20, 16
    nlat, nphi = 48, 64
    T = Float64
    
    println("Problem size: lmax=$lmax, mmax=$mmax, grid=$(nlat)×$(nphi)")
    
    # Create configuration
    cfg = create_gauss_config(T, lmax, mmax, nlat, nphi)
    println("Created SHT configuration with $(cfg.nlm) spectral coefficients")
    
    if PARALLEL_AVAILABLE
        # Get MPI info
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm) 
        
        println("MPI Info: rank $rank of $size processes")
        
        if rank == 0
            println("\n--- Testing Parallel Configuration ---")
        end
        
        # Test parallel configuration creation
        try
            pcfg = create_parallel_config(cfg, comm)
            if rank == 0
                println("Created parallel configuration")
                println("  Local l range: $(pcfg.local_l_range)")
                println("  Local m range: $(pcfg.local_m_range)")
            end
            
            # Test parallel operators
            if rank == 0
                println("\n--- Testing Parallel Operators ---")
            end
            
            # Create test data
            sh_coeffs = randn(Complex{T}, cfg.nlm)
            result = similar(sh_coeffs)
            
            # Test Laplacian (no communication)
            parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)
            if rank == 0
                println("Parallel Laplacian operator")
            end
            
            # Test cos(θ) operator (requires communication)
            parallel_apply_operator(pcfg, :costheta, sh_coeffs, result)
            if rank == 0
                println("Parallel cos(θ) operator")
            end
            
            # Test memory-efficient transforms
            if rank == 0
                println("\n--- Testing Parallel Transforms ---")
            end
            
            spatial_data = allocate_spatial(cfg)
            
            # Parallel synthesis
            memory_efficient_parallel_transform!(pcfg, :synthesis, sh_coeffs, spatial_data)
            if rank == 0
                println("Parallel spherical harmonic synthesis")
            end
            
            # Parallel analysis  
            sh_result = similar(sh_coeffs)
            memory_efficient_parallel_transform!(pcfg, :analysis, spatial_data, sh_result)
            if rank == 0
                println("Parallel spherical harmonic analysis")
                
                # Check roundtrip error
                error = maximum(abs.(sh_coeffs - sh_result))
                println("  Roundtrip error: $error")
            end
            
        catch e
            if rank == 0
                println("ERROR: Parallel functionality error: $e")
                println("  This may be due to missing package versions or MPI setup")
            end
        end
        
        # Performance comparison
        if rank == 0
            println("\n--- Performance Analysis ---")
            
            optimal_procs = optimal_process_count(cfg)
            println("Recommended number of processes: $optimal_procs")
            
            perf_model = parallel_performance_model(cfg, size)
            println("Performance model:")
            println("  Serial time estimate: $(perf_model.serial_time*1000:.2f) ms")
            println("  Parallel time estimate: $(perf_model.parallel_time*1000:.2f) ms") 
            println("  Expected speedup: $(perf_model.speedup:.2f)x")
            println("  Parallel efficiency: $(perf_model.efficiency*100:.1f)%")
        end
        
    else
        println("\n--- Serial Mode (No MPI) ---")
        
        # Test auto configuration (should fall back to serial)
        try
            auto_cfg = auto_parallel_config(cfg)
            println("Auto configuration created (serial fallback)")
        catch e
            println("Auto configuration correctly reports MPI not available")
        end
        
        # Show what would be optimal if parallel was available
        optimal_procs = optimal_process_count(cfg)
        println("Would recommend $optimal_procs processes if MPI was available")
        
        perf_model = parallel_performance_model(cfg, 1)
        println("Serial performance estimate: $(perf_model.serial_time*1000:.2f) ms")
    end
    
    # Test basic SHTnsKit functionality regardless of MPI
    println("\n--- Basic SHTnsKit Functionality ---")
    
    # Create test data
    sh_coeffs = randn(Complex{T}, cfg.nlm)
    spatial_data = allocate_spatial(cfg)
    
    # Forward transform
    sh_to_spat!(cfg, sh_coeffs, spatial_data)
    println("Spherical harmonic synthesis")
    
    # Backward transform
    sh_result = similar(sh_coeffs)
    spat_to_sh!(cfg, spatial_data, sh_result)
    println("Spherical harmonic analysis")
    
    # Check accuracy
    error = maximum(abs.(sh_coeffs - sh_result))
    println("Roundtrip error: $error")
    
    # Test matrix operators
    laplacian_result = similar(sh_coeffs)
    apply_laplacian!(cfg, sh_coeffs, laplacian_result)
    println("Laplacian operator")
    
    costheta_result = similar(sh_coeffs)
    apply_costheta_operator!(cfg, sh_coeffs, costheta_result)
    println("cos(θ) operator")
    
    println("\n" * "="^60)
    if PARALLEL_AVAILABLE
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if rank == 0
            println("Parallel example completed successfully!")
        end
    else
        println("Serial example completed successfully!")
        println("Install MPI, PencilArrays, and PencilFFTs for parallel functionality")
    end
    println("="^60)
end

function performance_benchmark()
    """Run a performance benchmark comparing serial vs parallel (if available)"""
    
    if !PARALLEL_AVAILABLE
        println("Performance benchmark requires MPI packages")
        return
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank != 0
        return  # Only run benchmark on root process
    end
    
    println("\n" * "="^60)
    println("Performance Benchmark")
    println("="^60)
    
    # Test different problem sizes
    test_sizes = [(10, 8), (20, 16), (30, 24)]
    
    for (lmax, mmax) in test_sizes
        nlat, nphi = 2*lmax + 2, 4*lmax + 1
        cfg = create_gauss_config(Float64, lmax, mmax, nlat, nphi)
        
        println("\nProblem: lmax=$lmax, nlm=$(cfg.nlm)")
        
        # Create test data
        sh_coeffs = randn(Complex{Float64}, cfg.nlm)
        spatial_data = allocate_spatial(cfg)
        
        # Serial timing
        serial_time = @elapsed begin
            for _ in 1:10
                sh_to_spat!(cfg, sh_coeffs, spatial_data)
                spat_to_sh!(cfg, spatial_data, sh_coeffs)
            end
        end
        serial_time /= 10  # Average
        
        # Parallel timing
        try
            pcfg = create_parallel_config(cfg, comm)
            parallel_time = @elapsed begin
                for _ in 1:10
                    memory_efficient_parallel_transform!(pcfg, :synthesis, sh_coeffs, spatial_data)
                    memory_efficient_parallel_transform!(pcfg, :analysis, spatial_data, sh_coeffs)
                end
            end
            parallel_time /= 10  # Average
            
            speedup = serial_time / parallel_time
            efficiency = speedup / size * 100
            
            println("  Serial time:    $(serial_time*1000:.2f) ms")
            println("  Parallel time:  $(parallel_time*1000:.2f) ms")
            println("  Speedup:        $(speedup:.2f)x")
            println("  Efficiency:     $(efficiency:.1f)%")
            
        catch e
            println("  Parallel timing failed: $e")
        end
    end
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    run_parallel_example()
    
    # Optionally run performance benchmark
    if PARALLEL_AVAILABLE && length(ARGS) > 0 && ARGS[1] == "--benchmark"
        performance_benchmark()
    end
    
    # Finalize MPI if it was initialized
    if PARALLEL_AVAILABLE
        MPI.Finalize()
    end
end