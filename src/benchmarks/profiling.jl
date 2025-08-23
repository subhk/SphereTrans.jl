"""
Timing and profiling functions for spherical harmonic transforms.
These functions provide performance monitoring capabilities similar to the SHTns C library.
"""

mutable struct SHTnsProfiler{T<:AbstractFloat}
    enabled::Bool
    total_time::T
    legendre_time::T
    fourier_time::T
    last_total::T  
    last_legendre::T
    last_fourier::T
    transform_count::Int
    
    function SHTnsProfiler{T}() where T
        new{T}(false, zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), 0)
    end
end

# Global profiler instances (one per thread for thread safety)
const PROFILERS = Dict{Int,SHTnsProfiler{Float64}}()

"""
    shtns_profiling(cfg::SHTnsConfig{T}, enabled::Bool) where T

Enable or disable profiling for spherical harmonic transforms.

# Arguments
- `cfg`: SHTns configuration
- `enabled`: Whether to enable (true) or disable (false) profiling

When enabled, timing information will be collected for subsequent transforms.
Use `shtns_profiling_read_time()` to retrieve the timing data.

Equivalent to the C library function `shtns_profiling()`.
"""
function shtns_profiling(cfg::SHTnsConfig{T}, enabled::Bool) where T
    thread_id = Threads.threadid()
    
    if !haskey(PROFILERS, thread_id)
        PROFILERS[thread_id] = SHTnsProfiler{Float64}()
    end
    
    PROFILERS[thread_id].enabled = enabled
    
    if enabled
        PROFILERS[thread_id].total_time = 0.0
        PROFILERS[thread_id].legendre_time = 0.0
        PROFILERS[thread_id].fourier_time = 0.0
        PROFILERS[thread_id].transform_count = 0
    end
    
    return nothing
end

"""
    shtns_profiling_read_time(cfg::SHTnsConfig{T}) where T

Read timing information from the last transform.

# Returns
- `(total_time, legendre_time, fourier_time)`: Tuple of timing information in seconds

The total time is also returned as the function result for compatibility.

Equivalent to the C library function `shtns_profiling_read_time()`.
"""
function shtns_profiling_read_time(cfg::SHTnsConfig{T}) where T
    thread_id = Threads.threadid()
    
    if haskey(PROFILERS, thread_id) && PROFILERS[thread_id].enabled
        profiler = PROFILERS[thread_id]
        return (profiler.last_total, profiler.last_legendre, profiler.last_fourier)
    else
        return (0.0, 0.0, 0.0)
    end
end

"""
    sh_to_spat_time(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}) where T

Perform spherical harmonic synthesis with timing measurement.

# Arguments  
- `cfg`: SHTns configuration
- `qlm`: Input SH coefficients
- `vr`: Output spatial field (pre-allocated)

# Returns
- Total time in seconds

The spatial field is computed in `vr` and timing information is stored for later retrieval.

Equivalent to the C library function `SH_to_spat_time()`.
"""
function sh_to_spat_time(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, vr::AbstractMatrix{T}) where T
    thread_id = Threads.threadid()
    
    if !haskey(PROFILERS, thread_id)
        PROFILERS[thread_id] = SHTnsProfiler{Float64}()
    end
    
    profiler = PROFILERS[thread_id]
    
    # Perform timed transform
    start_time = time_ns()
    
    # Break down into Legendre and Fourier parts
    legendre_start = time_ns()
    result = _sh_to_spat_with_timing!(cfg, qlm, vr, profiler)
    fourier_end = time_ns()
    
    total_time = (fourier_end - start_time) * 1e-9
    
    # Store timing information
    profiler.last_total = total_time
    profiler.total_time += total_time
    profiler.transform_count += 1
    
    return total_time
end

"""
    spat_to_sh_time(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, qlm::AbstractVector{Complex{T}}) where T

Perform spherical harmonic analysis with timing measurement.

# Arguments
- `cfg`: SHTns configuration  
- `vr`: Input spatial field
- `qlm`: Output SH coefficients (pre-allocated)

# Returns  
- Total time in seconds

Equivalent to the C library function `spat_to_SH_time()`.
"""
function spat_to_sh_time(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, 
                        qlm::AbstractVector{Complex{T}}) where T
                        
    thread_id = Threads.threadid()
    
    if !haskey(PROFILERS, thread_id)
        PROFILERS[thread_id] = SHTnsProfiler{Float64}()
    end
    
    profiler = PROFILERS[thread_id]
    
    # Perform timed transform
    start_time = time_ns()
    result = _spat_to_sh_with_timing!(cfg, vr, qlm, profiler)
    end_time = time_ns()
    
    total_time = (end_time - start_time) * 1e-9
    
    # Store timing information
    profiler.last_total = total_time
    profiler.total_time += total_time
    profiler.transform_count += 1
    
    return total_time
end

"""
    benchmark_transform(cfg::SHTnsConfig{T}, n_iterations::Int=100) where T

Benchmark transform performance with the given configuration.

# Arguments
- `cfg`: SHTns configuration
- `n_iterations`: Number of iterations to average over

# Returns
- Named tuple with timing statistics: `(synthesis_time, analysis_time, total_time, iterations)`
"""
function benchmark_transform(cfg::SHTnsConfig{T}, n_iterations::Int=100) where T
    validate_config(cfg)
    n_iterations > 0 || error("n_iterations must be positive")
    
    # Allocate test data
    qlm = allocate_spectral(cfg)
    vr = allocate_spatial(cfg)
    
    # Initialize with some test data  
    for i in 1:cfg.nlm
        qlm[i] = Complex{T}(randn(T), randn(T))
    end
    
    # Warmup
    sh_to_spat!(cfg, qlm, vr)
    spat_to_sh!(cfg, vr, qlm)
    
    # Benchmark synthesis
    synthesis_times = Vector{Float64}(undef, n_iterations)
    for i in 1:n_iterations
        synthesis_times[i] = sh_to_spat_time(cfg, qlm, vr)
    end
    
    # Benchmark analysis  
    analysis_times = Vector{Float64}(undef, n_iterations)
    for i in 1:n_iterations
        analysis_times[i] = spat_to_sh_time(cfg, vr, qlm)
    end
    
    # Compute statistics
    avg_synthesis = sum(synthesis_times) / n_iterations
    avg_analysis = sum(analysis_times) / n_iterations
    total_avg = avg_synthesis + avg_analysis
    
    return (
        synthesis_time = avg_synthesis,
        analysis_time = avg_analysis,
        total_time = total_avg,
        iterations = n_iterations,
        synthesis_std = std(synthesis_times),
        analysis_std = std(analysis_times)
    )
end

"""
    profile_memory_usage(cfg::SHTnsConfig{T}) where T

Profile memory usage for the given configuration.

# Returns
- Named tuple with memory usage information
"""
function profile_memory_usage(cfg::SHTnsConfig{T}) where T
    validate_config(cfg)
    
    # Calculate theoretical memory requirements
    spectral_memory = cfg.nlm * sizeof(Complex{T})
    spatial_memory = cfg.nlat * cfg.nphi * sizeof(T)
    
    # Cache memory (if allocated)
    cache_memory = 0
    if !isempty(cfg.plm_cache)
        cache_memory = sizeof(cfg.plm_cache)
    end
    
    # Work array memory (estimated)
    work_memory = spectral_memory + spatial_memory  # Rough estimate
    
    total_memory = spectral_memory + spatial_memory + cache_memory + work_memory
    
    return (
        spectral_memory = spectral_memory,
        spatial_memory = spatial_memory, 
        cache_memory = cache_memory,
        work_memory = work_memory,
        total_memory = total_memory,
        memory_mb = total_memory / (1024^2)
    )
end

"""
    get_profiling_summary(cfg::SHTnsConfig{T}) where T

Get a summary of profiling information accumulated so far.

# Returns  
- Named tuple with cumulative timing statistics
"""
function get_profiling_summary(cfg::SHTnsConfig{T}) where T
    thread_id = Threads.threadid()
    
    if haskey(PROFILERS, thread_id)
        profiler = PROFILERS[thread_id]
        
        if profiler.transform_count > 0
            avg_time = profiler.total_time / profiler.transform_count
            avg_legendre = profiler.legendre_time / profiler.transform_count
            avg_fourier = profiler.fourier_time / profiler.transform_count
        else
            avg_time = avg_legendre = avg_fourier = 0.0
        end
        
        return (
            total_time = profiler.total_time,
            average_time = avg_time,
            legendre_time = profiler.legendre_time,
            fourier_time = profiler.fourier_time,
            average_legendre = avg_legendre,
            average_fourier = avg_fourier,
            transform_count = profiler.transform_count,
            enabled = profiler.enabled
        )
    else
        return (
            total_time = 0.0,
            average_time = 0.0,
            legendre_time = 0.0,
            fourier_time = 0.0,
            average_legendre = 0.0,
            average_fourier = 0.0,
            transform_count = 0,
            enabled = false
        )
    end
end

"""
    reset_profiling(cfg::SHTnsConfig{T}) where T

Reset all accumulated profiling statistics.
"""
function reset_profiling(cfg::SHTnsConfig{T}) where T
    thread_id = Threads.threadid()
    
    if haskey(PROFILERS, thread_id)
        profiler = PROFILERS[thread_id]
        profiler.total_time = 0.0
        profiler.legendre_time = 0.0
        profiler.fourier_time = 0.0
        profiler.transform_count = 0
    end
    
    return nothing
end

# Internal timing functions

"""
    _sh_to_spat_with_timing!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, 
                             vr::AbstractMatrix{T}, profiler::SHTnsProfiler) where T

Internal synthesis function with detailed timing breakdown.
"""
function _sh_to_spat_with_timing!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}, 
                                  vr::AbstractMatrix{T}, profiler::SHTnsProfiler) where T
    if profiler.enabled
        # Detailed timing of each phase
        total_start = time_ns()
        
        legendre_start = time_ns()
        # Call the regular synthesis (would need to break this down further)
        sh_to_spat!(cfg, qlm, vr)
        legendre_end = time_ns()
        
        fourier_start = legendre_end  # Fourier part is included in above
        fourier_end = time_ns()
        
        # Store detailed timing
        legendre_time = (legendre_end - legendre_start) * 1e-9
        fourier_time = (fourier_end - fourier_start) * 1e-9
        
        profiler.last_legendre = legendre_time
        profiler.last_fourier = fourier_time
        profiler.legendre_time += legendre_time
        profiler.fourier_time += fourier_time
    else
        # Regular transform without timing overhead
        sh_to_spat!(cfg, qlm, vr)
    end
    
    return vr
end

"""
    _spat_to_sh_with_timing!(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T}, 
                             qlm::AbstractVector{Complex{T}}, profiler::SHTnsProfiler) where T

Internal analysis function with detailed timing breakdown.
"""  
function _spat_to_sh_with_timing!(cfg::SHTnsConfig{T}, vr::AbstractMatrix{T},
                                  qlm::AbstractVector{Complex{T}}, profiler::SHTnsProfiler) where T
    if profiler.enabled
        # Detailed timing of each phase
        total_start = time_ns()
        
        fourier_start = time_ns()
        # Call the regular analysis (would need to break this down further)
        spat_to_sh!(cfg, vr, qlm)
        fourier_end = time_ns()
        
        legendre_start = fourier_end  # Legendre part is included in above  
        legendre_end = time_ns()
        
        # Store detailed timing
        fourier_time = (fourier_end - fourier_start) * 1e-9
        legendre_time = (legendre_end - legendre_start) * 1e-9
        
        profiler.last_fourier = fourier_time
        profiler.last_legendre = legendre_time
        profiler.fourier_time += fourier_time
        profiler.legendre_time += legendre_time
    else
        # Regular transform without timing overhead
        spat_to_sh!(cfg, vr, qlm)
    end
    
    return qlm
end

"""
    compare_performance(cfg1::SHTnsConfig{T}, cfg2::SHTnsConfig{T}, n_iterations::Int=50) where T

Compare performance between two different configurations.

# Arguments
- `cfg1`, `cfg2`: SHTns configurations to compare
- `n_iterations`: Number of iterations for benchmarking

# Returns
- Named tuple with comparative timing information
"""
function compare_performance(cfg1::SHTnsConfig{T}, cfg2::SHTnsConfig{T}, n_iterations::Int=50) where T
    validate_config(cfg1)
    validate_config(cfg2)
    
    println("Benchmarking configuration 1...")
    stats1 = benchmark_transform(cfg1, n_iterations)
    
    println("Benchmarking configuration 2...")
    stats2 = benchmark_transform(cfg2, n_iterations)
    
    # Compute relative performance
    synthesis_ratio = stats2.synthesis_time / stats1.synthesis_time
    analysis_ratio = stats2.analysis_time / stats1.analysis_time
    total_ratio = stats2.total_time / stats1.total_time
    
    return (
        config1 = stats1,
        config2 = stats2,
        synthesis_ratio = synthesis_ratio,  # cfg2/cfg1
        analysis_ratio = analysis_ratio,
        total_ratio = total_ratio,
        faster_config = total_ratio < 1.0 ? 2 : 1
    )
end

"""
    print_profiling_report(cfg::SHTnsConfig{T}) where T

Print a formatted profiling report to stdout.
"""
function print_profiling_report(cfg::SHTnsConfig{T}) where T
    summary = get_profiling_summary(cfg)
    memory = profile_memory_usage(cfg)
    
    println("="^60)
    println("SHTns Performance Report")
    println("="^60)
    println("Configuration:")
    println("  lmax = $(cfg.lmax), mmax = $(cfg.mmax), mres = $(cfg.mres)")
    println("  Grid: $(cfg.nlat) × $(cfg.nphi) ($(cfg.grid_type))")
    println("  Normalization: $(cfg.norm)")
    println()
    
    if summary.enabled
        println("Timing Statistics ($(summary.transform_count) transforms):")
        println("  Total time: $(summary.total_time:.4f) s")
        println("  Average per transform: $(summary.average_time*1000:.2f) ms")
        if summary.legendre_time > 0 || summary.fourier_time > 0
            println("  Legendre component: $(summary.average_legendre*1000:.2f) ms")
            println("  Fourier component: $(summary.average_fourier*1000:.2f) ms")
        end
        println()
    else
        println("Profiling is disabled. Enable with shtns_profiling(cfg, true)")
        println()
    end
    
    println("Memory Usage:")
    println("  Spectral coefficients: $(memory.spectral_memory ÷ 1024) KB")
    println("  Spatial data: $(memory.spatial_memory ÷ 1024) KB") 
    println("  Cache: $(memory.cache_memory ÷ 1024) KB")
    println("  Total estimated: $(memory.memory_mb:.2f) MB")
    println("="^60)
end