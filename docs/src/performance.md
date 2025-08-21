# Performance Guide

This guide provides comprehensive information on optimizing SHTnsKit.jl performance for various computational scenarios.

## Understanding Performance Characteristics

### Transform Complexity

Spherical harmonic transforms have the following computational complexity:
- **Naive implementation**: O(L⁴) for degree L
- **SHTns optimized**: O(L³) with advanced algorithms
- **Memory usage**: O(L²) for spectral coefficients, O(L²) for spatial grid

### Performance Scaling

```julia
using SHTnsKit
using BenchmarkTools

function benchmark_transforms(lmax_values)
    results = []
    
    for lmax in lmax_values
        cfg = create_gauss_config(lmax, lmax)
        sh = rand(get_nlm(cfg))
        
        # Benchmark forward transform
        forward_time = @belapsed synthesize($cfg, $sh)
        
        # Benchmark backward transform
        spatial = synthesize(cfg, sh)
        backward_time = @belapsed analyze($cfg, $spatial)
        
        push!(results, (lmax=lmax, forward=forward_time, backward=backward_time))
        free_config(cfg)
    end
    
    return results
end

# Test scaling
lmax_range = [16, 32, 64, 128, 256]
results = benchmark_transforms(lmax_range)

for r in results
    println("lmax=$(r.lmax): forward=$(r.forward)s, backward=$(r.backward)s")
end
```

## Threading Optimization

### OpenMP Configuration

SHTnsKit uses OpenMP for multi-threading. Optimal performance requires proper thread configuration:

```julia
using SHTnsKit

# Check system capabilities
println("System threads: ", Sys.CPU_THREADS)
println("Current OpenMP threads: ", get_num_threads())

# Optimal thread setting
set_optimal_threads()
optimal_threads = get_num_threads()
println("Optimal threads: ", optimal_threads)

# Manual thread control
function benchmark_threading(lmax=64)
    cfg = create_gauss_config(lmax, lmax)
    sh = rand(get_nlm(cfg))
    
    thread_counts = [1, 2, 4, 8, min(16, Sys.CPU_THREADS)]
    
    for nthreads in thread_counts
        set_num_threads(nthreads)
        time = @elapsed begin
            for i in 1:10
                synthesize(cfg, sh)
            end
        end
        
        speedup = (thread_counts[1] > 0) ? 
                  time / benchmark_threading_baseline : 1.0
        println("$nthreads threads: $(time/10)s per transform")
    end
    
    free_config(cfg)
end

benchmark_threading()
```

### Thread Affinity and NUMA

For high-performance computing:

```bash
# Set thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# NUMA-aware execution
export OMP_NUM_THREADS=16
numactl --interleave=all julia script.jl
```

### Avoiding Oversubscription

```julia
# Prevent thread oversubscription with other libraries
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["FFTW_NUM_THREADS"] = "1"

# Set SHTns threads to match physical cores
using Sys
set_num_threads(min(Sys.CPU_THREADS ÷ 2, 8))  # Account for hyperthreading
```

## Memory Optimization

### Pre-allocation Strategies

```julia
using SHTnsKit

cfg = create_gauss_config(64, 64)

# Method 1: Allocate once, reuse many times
sh_buffer = allocate_spectral(cfg)
spatial_buffer = allocate_spatial(cfg)

function process_many_fields_optimized(field_generator, n_fields)
    results = Float64[]
    
    for i in 1:n_fields
        # Generate field data (application-specific)
        fill!(spatial_buffer, 0.0)
        field_data = field_generator(i)
        spatial_buffer .= field_data
        
        # In-place transform
        analyze!(cfg, spatial_buffer, sh_buffer)
        
        # Process result (example: compute energy)
        energy = sum(abs2, sh_buffer)
        push!(results, energy)
    end
    
    return results
end

# vs Method 2: Allocate every time (slower)
function process_many_fields_naive(field_generator, n_fields)
    results = Float64[]
    
    for i in 1:n_fields
        field_data = field_generator(i)
        sh = analyze(cfg, field_data)  # Allocates new array
        energy = sum(abs2, sh)
        push!(results, energy)
    end
    
    return results
end

free_config(cfg)
```

### Memory Layout Optimization

```julia
# For batch processing, consider array-of-arrays vs array layout
using SHTnsKit

cfg = create_gauss_config(32, 32)
n_fields = 1000

# Layout 1: Array of arrays (better for random access)
spectral_data_aoa = [rand(get_nlm(cfg)) for _ in 1:n_fields]

# Layout 2: Single large array (better for streaming)
nlm = get_nlm(cfg)
spectral_data_flat = rand(nlm, n_fields)

# Process with different layouts
@time begin
    for i in 1:n_fields
        spatial = synthesize(cfg, spectral_data_aoa[i])
    end
end

@time begin
    for i in 1:n_fields
        spatial = synthesize(cfg, @view spectral_data_flat[:, i])
    end
end

free_config(cfg)
```

### Large Problem Memory Management

```julia
using SHTnsKit

function process_large_dataset(lmax=256, n_fields=10000)
    cfg = create_gauss_config(lmax, lmax)
    
    # For very large problems, process in chunks
    chunk_size = 100
    n_chunks = div(n_fields, chunk_size)
    
    results = Float64[]
    
    for chunk in 1:n_chunks
        # Process chunk
        chunk_results = Float64[]
        
        for i in 1:chunk_size
            # Generate field (don't store all at once)
            sh = rand(get_nlm(cfg))
            spatial = synthesize(cfg, sh)
            
            # Compute result
            result = mean(spatial)
            push!(chunk_results, result)
        end
        
        # Store chunk results
        append!(results, chunk_results)
        
        # Force garbage collection between chunks
        GC.gc()
    end
    
    free_config(cfg)
    return results
end
```

## GPU Acceleration

### CUDA Setup and Optimization

```julia
using SHTnsKit
using CUDA

if CUDA.functional()
    # GPU memory management
    function optimize_gpu_memory()
        # Check available memory
        total_mem = CUDA.device!(0) do
            CUDA.total_memory()
        end
        
        free_mem = CUDA.device!(0) do  
            CUDA.free_memory()
        end
        
        println("GPU Memory - Total: $(total_mem÷1024^3)GB, Free: $(free_mem÷1024^3)GB")
        
        # Set memory pool size
        CUDA.memory_pool_limit!(div(free_mem, 2))  # Use half of free memory
    end
    
    optimize_gpu_memory()
    
    # GPU performance benchmarking
    function benchmark_gpu_vs_cpu(lmax=128)
        cfg_cpu = create_gauss_config(lmax, lmax)
        cfg_gpu = create_gpu_config(lmax, lmax)
        
        # Initialize GPU
        initialize_gpu(0, verbose=false)
        
        # Data
        sh = rand(get_nlm(cfg_cpu))
        sh_gpu = CuArray(sh)
        
        # CPU benchmark
        cpu_time = @elapsed begin
            for i in 1:10
                spatial = synthesize(cfg_cpu, sh)
            end
        end
        
        # GPU benchmark (with data transfer)
        gpu_time_with_transfer = @elapsed begin
            for i in 1:10
                sh_gpu_i = CuArray(sh)
                spatial_gpu = synthesize_gpu(cfg_gpu, sh_gpu_i)
                spatial_cpu = Array(spatial_gpu)
            end
        end
        
        # GPU benchmark (no data transfer)
        gpu_time_no_transfer = @elapsed begin
            for i in 1:10
                spatial_gpu = synthesize_gpu(cfg_gpu, sh_gpu)
            end
        end
        
        println("Performance Comparison (lmax=$lmax):")
        println("CPU: $(cpu_time/10)s per transform")
        println("GPU (with transfer): $(gpu_time_with_transfer/10)s per transform") 
        println("GPU (no transfer): $(gpu_time_no_transfer/10)s per transform")
        
        speedup = cpu_time / gpu_time_no_transfer
        println("GPU speedup: $(speedup)x")
        
        cleanup_gpu(verbose=false)
        free_config(cfg_cpu)
        free_config(cfg_gpu)
    end
    
    benchmark_gpu_vs_cpu()
    
else
    println("CUDA not available")
end
```

### GPU Memory Staging

```julia
using CUDA, SHTnsKit

function gpu_batch_processing(spectral_data::Vector{Vector{Float64}})
    if !CUDA.functional()
        error("CUDA not available")
    end
    
    cfg = create_gpu_config(32, 32)
    initialize_gpu(0, verbose=false)
    
    # Stage data in GPU memory
    batch_size = 32
    n_batches = div(length(spectral_data), batch_size)
    
    results = []
    
    for batch in 1:n_batches
        # Upload batch to GPU
        gpu_batch = []
        for i in 1:batch_size
            idx = (batch-1)*batch_size + i
            if idx <= length(spectral_data)
                push!(gpu_batch, CuArray(spectral_data[idx]))
            end
        end
        
        # Process on GPU
        batch_results = []
        for sh_gpu in gpu_batch
            spatial_gpu = synthesize_gpu(cfg, sh_gpu)
            # Do processing on GPU if possible
            result = CUDA.sum(spatial_gpu)  # Example operation
            push!(batch_results, Array(result)[1])
        end
        
        append!(results, batch_results)
        
        # Clear GPU memory for next batch
        CUDA.reclaim()
    end
    
    cleanup_gpu(verbose=false)
    free_config(cfg)
    
    return results
end
```

## Algorithm-Specific Optimizations

### Transform Direction Optimization

```julia
using SHTnsKit

cfg = create_gauss_config(64, 64)

# Forward transforms (synthesis) are generally faster than backward (analysis)
# Plan your algorithm to minimize analysis operations

function optimize_transform_direction()
    sh = rand(get_nlm(cfg))
    spatial = rand(get_nlat(cfg), get_nphi(cfg))
    
    # Forward transform timing
    forward_time = @elapsed begin
        for i in 1:100
            synthesize(cfg, sh)
        end
    end
    
    # Backward transform timing
    backward_time = @elapsed begin
        for i in 1:100
            analyze(cfg, spatial)
        end
    end
    
    println("Forward: $(forward_time/100)s")
    println("Backward: $(backward_time/100)s") 
    println("Ratio: $(backward_time/forward_time)")
end

optimize_transform_direction()
free_config(cfg)
```

### Grid Type Selection

```julia
using SHTnsKit

function compare_grid_types(lmax=32)
    # Gauss grids: optimal for accuracy, fewer points
    cfg_gauss = create_gauss_config(lmax, lmax)
    
    # Regular grids: more points, but uniform spacing
    cfg_regular = create_regular_config(lmax, lmax)
    
    println("Grid Comparison (lmax=$lmax):")
    println("Gauss: $(get_nlat(cfg_gauss)) × $(get_nphi(cfg_gauss)) = $(get_nlat(cfg_gauss)*get_nphi(cfg_gauss)) points")
    println("Regular: $(get_nlat(cfg_regular)) × $(get_nphi(cfg_regular)) = $(get_nlat(cfg_regular)*get_nphi(cfg_regular)) points")
    
    # Benchmark both
    sh = rand(get_nlm(cfg_gauss))  # Same spectral resolution
    
    gauss_time = @elapsed begin
        for i in 1:50
            synthesize(cfg_gauss, sh)
        end
    end
    
    regular_time = @elapsed begin  
        for i in 1:50
            synthesize(cfg_regular, sh)
        end
    end
    
    println("Gauss time: $(gauss_time/50)s")
    println("Regular time: $(regular_time/50)s")
    println("Gauss is $(regular_time/gauss_time)x faster")
    
    free_config(cfg_gauss)
    free_config(cfg_regular)
end

compare_grid_types()
```

## Vector Field Performance

```julia
using SHTnsKit

cfg = create_gauss_config(48, 48)

# Vector transforms are more expensive than scalar
function benchmark_vector_vs_scalar()
    # Scalar data
    sh_scalar = rand(get_nlm(cfg))
    spatial_scalar = rand(get_nlat(cfg), get_nphi(cfg))
    
    # Vector data  
    S_lm = rand(get_nlm(cfg))
    T_lm = rand(get_nlm(cfg))
    Vθ = rand(get_nlat(cfg), get_nphi(cfg))
    Vφ = rand(get_nlat(cfg), get_nphi(cfg))
    
    # Scalar benchmarks
    scalar_synth = @elapsed begin
        for i in 1:20
            synthesize(cfg, sh_scalar)
        end
    end
    
    scalar_analysis = @elapsed begin
        for i in 1:20
            analyze(cfg, spatial_scalar)
        end
    end
    
    # Vector benchmarks
    vector_synth = @elapsed begin
        for i in 1:20
            synthesize_vector(cfg, S_lm, T_lm)
        end
    end
    
    vector_analysis = @elapsed begin
        for i in 1:20
            analyze_vector(cfg, Vθ, Vφ)
        end  
    end
    
    println("Transform Performance Comparison:")
    println("Scalar synthesis: $(scalar_synth/20)s")
    println("Vector synthesis: $(vector_synth/20)s ($(vector_synth/scalar_synth)x slower)")
    println("Scalar analysis: $(scalar_analysis/20)s")
    println("Vector analysis: $(vector_analysis/20)s ($(vector_analysis/scalar_analysis)x slower)")
end

benchmark_vector_vs_scalar()
free_config(cfg)
```

## Distributed Computing Performance

### MPI Scaling

```julia
using SHTnsKit
# using MPI  # Uncomment if MPI is available

function mpi_scaling_example()
    # This would run across multiple nodes
    # Pseudo-code for illustration
    
    println("MPI Performance Considerations:")
    println("- Communication overhead increases with more processes")
    println("- Optimal process count depends on problem size")  
    println("- Network topology affects scaling")
    println("- Load balancing crucial for irregular problems")
end

# MPI performance best practices:
# 1. Minimize communication
# 2. Overlap computation and communication  
# 3. Use appropriate process topology
# 4. Consider hybrid MPI+OpenMP
```

## Performance Monitoring and Profiling

### Built-in Benchmarking

```julia
using SHTnsKit
using Profile
using BenchmarkTools

cfg = create_gauss_config(64, 64)

function profile_transforms()
    sh = rand(get_nlm(cfg))
    
    # Detailed benchmarking
    forward_bench = @benchmark synthesize($cfg, $sh)
    println("Forward transform statistics:")
    println("  Median: $(median(forward_bench.times))ns")
    println("  Mean: $(mean(forward_bench.times))ns") 
    println("  Std: $(std(forward_bench.times))ns")
    
    # Memory allocation tracking
    spatial = synthesize(cfg, sh)
    backward_bench = @benchmark analyze($cfg, $spatial)
    
    println("Backward transform statistics:")
    println("  Median: $(median(backward_bench.times))ns")
    println("  Allocations: $(backward_bench.memory) bytes")
end

profile_transforms()

# Julia profiling
function profile_detailed()
    sh = rand(get_nlm(cfg))
    
    Profile.clear()
    @profile begin
        for i in 1:100
            spatial = synthesize(cfg, sh)
        end
    end
    
    Profile.print()
end

free_config(cfg)
```

### Custom Performance Metrics

```julia
using SHTnsKit

function performance_report(cfg, n_runs=100)
    # Warm up
    sh_test = rand(get_nlm(cfg))
    for i in 1:5
        synthesize(cfg, sh_test)
    end
    
    # Collect metrics
    times = Float64[]
    
    for i in 1:n_runs
        sh = rand(get_nlm(cfg))
        
        time = @elapsed begin
            spatial = synthesize(cfg, sh)
        end
        
        push!(times, time)
    end
    
    # Statistics
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)
    
    # Compute derived metrics
    lmax = get_lmax(cfg)
    operations_per_sec = 1.0 / mean_time
    points_per_sec = (get_nlat(cfg) * get_nphi(cfg)) / mean_time
    
    println("Performance Report (lmax=$lmax, $n_runs runs):")
    println("  Mean time: $(mean_time*1000)ms (±$(std_time*1000)ms)")
    println("  Min/Max: $(min_time*1000)ms / $(max_time*1000)ms")
    println("  Transforms/sec: $(operations_per_sec)")
    println("  Points/sec: $(points_per_sec)")
    println("  Grid efficiency: $(get_nlm(cfg)/(get_nlat(cfg)*get_nphi(cfg)))")
end

cfg = create_gauss_config(32, 32)
performance_report(cfg)
free_config(cfg)
```

## Optimization Checklist

### Before Optimization
- [ ] Profile your code to identify bottlenecks
- [ ] Understand your problem's computational characteristics
- [ ] Measure baseline performance

### Threading Optimization  
- [ ] Set `OMP_NUM_THREADS` appropriately
- [ ] Use `set_optimal_threads()` for automatic tuning
- [ ] Disable threading in other libraries (BLAS, FFTW)
- [ ] Consider NUMA topology for large systems

### Memory Optimization
- [ ] Pre-allocate buffers for repeated operations
- [ ] Use in-place transforms when possible
- [ ] Process data in chunks for large datasets
- [ ] Monitor memory usage and fragmentation

### Algorithm Optimization
- [ ] Minimize backward transforms (analysis)
- [ ] Choose appropriate grid type (Gauss vs regular)
- [ ] Batch operations when possible
- [ ] Cache frequently used configurations

### GPU Optimization (if applicable)
- [ ] Verify CUDA functionality
- [ ] Manage GPU memory carefully
- [ ] Minimize CPU-GPU data transfers
- [ ] Use appropriate batch sizes

### System-Level Optimization
- [ ] Use high-performance BLAS library
- [ ] Enable CPU optimizations (AVX, etc.)
- [ ] Consider process/thread affinity
- [ ] Monitor system resource utilization

### Performance Validation
- [ ] Compare with baseline measurements
- [ ] Verify numerical accuracy after optimization
- [ ] Test with realistic problem sizes
- [ ] Document performance characteristics

## Common Performance Pitfalls

1. **Thread Oversubscription**: Too many threads can hurt performance
2. **Memory Allocation**: Repeated allocation in inner loops
3. **Wrong Grid Type**: Regular grids when Gauss would suffice
4. **Unnecessary Transforms**: Computing both directions when only one needed
5. **GPU Overhead**: Using GPU for small problems where setup cost dominates
6. **MPI Communication**: Excessive communication for small local problems

Following these guidelines will help you achieve optimal performance for your specific SHTnsKit.jl applications.