"""
SIMD-optimized matrix operations for high-performance single-node computations.

This module provides vectorized implementations of spherical harmonic operators
using Julia's SIMD capabilities and manual loop optimizations.
"""

using Base.Threads
using LinearAlgebra

"""
    simd_apply_laplacian!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}) where T

SIMD-vectorized Laplacian application with manual loop unrolling.
Processes multiple (l,m) coefficients simultaneously using SIMD lanes.
"""
function simd_apply_laplacian!(cfg::SHTnsConfig{T}, qlm::AbstractVector{Complex{T}}) where T
    nlm = cfg.nlm
    lm_indices = cfg.lm_indices
    
    # Vectorized processing using Julia's built-in SIMD
    # Process in chunks for better cache utilization
    chunk_size = 8  # Process 8 elements at a time
    n_chunks = nlm ÷ chunk_size
    
    @inbounds for chunk in 1:n_chunks
        base_idx = (chunk - 1) * chunk_size + 1
        
        # Vectorized loop - Julia will SIMD-optimize this
        @simd for i in 0:(chunk_size-1)
            idx = base_idx + i
            l, _ = lm_indices[idx]
            eigenval = -T(l * (l + 1))
            qlm[idx] *= eigenval
        end
    end
    
    # Handle remainder elements
    @inbounds for i in (n_chunks * chunk_size + 1):nlm
        l, _ = lm_indices[i]
        qlm[i] *= -T(l * (l + 1))
    end
    
    return qlm
end

"""
    threaded_apply_costheta_operator!(cfg::SHTnsConfig{T}, 
                                     qlm_in::AbstractVector{Complex{T}}, 
                                     qlm_out::AbstractVector{Complex{T}}) where T

Multi-threaded cos(θ) operator with NUMA-aware work distribution.
"""
function threaded_apply_costheta_operator!(cfg::SHTnsConfig{T}, 
                                          qlm_in::AbstractVector{Complex{T}}, 
                                          qlm_out::AbstractVector{Complex{T}}) where T
    nlm = cfg.nlm
    lm_indices = cfg.lm_indices
    nthreads = Threads.nthreads()
    
    # Initialize output
    fill!(qlm_out, zero(Complex{T}))
    
    # Thread-local intermediate results to avoid false sharing
    thread_results = [zeros(Complex{T}, nlm) for _ in 1:nthreads]
    
    # Divide work among threads, each thread handles a range of output indices
    @threads for tid in 1:nthreads
        thread_start = ((tid - 1) * nlm) ÷ nthreads + 1
        thread_end = (tid * nlm) ÷ nthreads
        local_result = thread_results[tid]
        
        @inbounds for idx_out in thread_start:thread_end
            l_out, m_out = lm_indices[idx_out]
            
            # Compute coupling contributions for this output index
            for idx_in in 1:nlm
                l_in, m_in = lm_indices[idx_in]
                
                # cos(θ) couples same m, adjacent l
                if m_out == m_in && abs(l_in - l_out) == 1
                    coeff = _costheta_coupling_coefficient(cfg, l_out, l_in, m_out)
                    if abs(coeff) > eps(T)
                        local_result[idx_out] += coeff * qlm_in[idx_in]
                    end
                end
            end
        end
    end
    
    # Combine thread results (no conflicts since each thread wrote to different indices)
    @inbounds for tid in 1:nthreads
        thread_start = ((tid - 1) * nlm) ÷ nthreads + 1
        thread_end = (tid * nlm) ÷ nthreads
        
        for idx in thread_start:thread_end
            qlm_out[idx] = thread_results[tid][idx]
        end
    end
    
    return qlm_out
end

"""
    vectorized_sparse_matvec!(A::SparseMatrixCSC{T}, 
                              x_real::Vector{T}, x_imag::Vector{T},
                              y_real::Vector{T}, y_imag::Vector{T}) where T

Vectorized sparse matrix-vector multiplication for complex vectors.
Processes real and imaginary parts separately for better memory layout.
"""
function vectorized_sparse_matvec!(A::SparseMatrixCSC{T}, 
                                   x_real::Vector{T}, x_imag::Vector{T},
                                   y_real::Vector{T}, y_imag::Vector{T}) where T
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    
    fill!(y_real, zero(T))
    fill!(y_imag, zero(T))
    
    @inbounds for col in 1:n
        x_real_col = x_real[col]
        x_imag_col = x_imag[col]
        
        # Process non-zeros in this column with SIMD-friendly loop
        col_range = nzrange(A, col)
        
        @simd for nz_idx in col_range
            row = rows[nz_idx]
            val = vals[nz_idx]
            y_real[row] += val * x_real_col
            y_imag[row] += val * x_imag_col
        end
    end
    
    return (y_real, y_imag)
end

"""
    cache_optimized_coupling_computation!(cfg::SHTnsConfig{T}, 
                                         coupling_matrix::SparseMatrixCSC{T}) where T

Cache-optimized computation of coupling coefficients with data locality optimization.
Pre-computes and stores coupling matrices in cache-friendly formats.
"""
function cache_optimized_coupling_computation!(cfg::SHTnsConfig{T}, 
                                              coupling_matrix::SparseMatrixCSC{T}) where T
    lm_indices = cfg.lm_indices
    nlm = cfg.nlm
    
    # Group computations by m-value for better cache locality
    m_groups = Dict{Int, Vector{Int}}()
    for (idx, (l, m)) in enumerate(lm_indices)
        if !haskey(m_groups, m)
            m_groups[m] = Int[]
        end
        push!(m_groups[m], idx)
    end
    
    # Process each m-group independently (better cache reuse)
    for (m_val, indices) in m_groups
        # All indices in this group have the same m value
        # cos(θ) operator only couples within same m
        
        for idx_out in indices
            l_out, _ = lm_indices[idx_out]
            
            # Find coupling partners (same m, adjacent l)
            for idx_in in indices
                l_in, _ = lm_indices[idx_in]
                
                if abs(l_in - l_out) == 1
                    coeff = _costheta_coupling_coefficient(cfg, l_out, l_in, m_val)
                    if abs(coeff) > eps(T)
                        coupling_matrix[idx_out, idx_in] = coeff
                    end
                end
            end
        end
    end
    
    return coupling_matrix
end

"""
    prefetch_optimized_matvec!(A::SparseMatrixCSC{T}, x::Vector{Complex{T}}, 
                               y::Vector{Complex{T}}) where T

Matrix-vector multiplication with manual prefetching for better cache performance.
"""
function prefetch_optimized_matvec!(A::SparseMatrixCSC{T}, x::Vector{Complex{T}}, 
                                   y::Vector{Complex{T}}) where T
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    
    fill!(y, zero(Complex{T}))
    
    @inbounds for col in 1:n
        x_val = x[col]
        
        # Prefetch next column's data (if exists)
        if col < n
            Base.llvmcall("""
                %ptr = inttoptr i64 %0 to i8*
                call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 1)
                ret void
                """, Cvoid, Tuple{Ptr{Complex{T}}}, pointer(x, col + 1))
        end
        
        for idx in nzrange(A, col)
            row = rows[idx]
            val = vals[idx]
            
            # Prefetch next row data
            if idx < length(nzrange(A, col))
                next_row = rows[idx + 1]
                Base.llvmcall("""
                    %ptr = inttoptr i64 %0 to i8*
                    call void @llvm.prefetch(i8* %ptr, i32 1, i32 3, i32 1)
                    ret void
                    """, Cvoid, Tuple{Ptr{Complex{T}}}, pointer(y, next_row))
            end
            
            y[row] += val * x_val
        end
    end
    
    return y
end

"""
    auto_simd_dispatch(cfg::SHTnsConfig{T}, op::Symbol, qlm_in, qlm_out) where T

Automatically choose the best SIMD implementation based on problem characteristics.
"""
function auto_simd_dispatch(cfg::SHTnsConfig{T}, op::Symbol, qlm_in::AbstractVector{Complex{T}}, 
                           qlm_out::AbstractVector{Complex{T}}) where T
    nlm = cfg.nlm
    nthreads = Threads.nthreads()
    
    if op === :laplacian
        # Laplacian is always best with SIMD
        return simd_apply_laplacian!(cfg, qlm_in)
    elseif op === :costheta
        # Choose based on problem size and thread availability
        if nlm > 1000 && nthreads > 1
            return threaded_apply_costheta_operator!(cfg, qlm_in, qlm_out)
        elseif nlm > 100
            # Use sparse matrix with vectorized multiply
            matrix = mul_ct_matrix(cfg)
            x_real = [real(x) for x in qlm_in]
            x_imag = [imag(x) for x in qlm_in]
            y_real = zeros(T, nlm)
            y_imag = zeros(T, nlm)
            
            vectorized_sparse_matvec!(matrix, x_real, x_imag, y_real, y_imag)
            
            @inbounds @simd for i in 1:nlm
                qlm_out[i] = complex(y_real[i], y_imag[i])
            end
            
            return qlm_out
        else
            # Small problems: use direct computation
            return apply_costheta_operator_direct!(cfg, qlm_in, qlm_out)
        end
    else
        error("Unknown operator: $op")
    end
end

"""
    benchmark_simd_variants(cfg::SHTnsConfig{T}) where T

Benchmark different SIMD implementations to choose the best one dynamically.
"""
function benchmark_simd_variants(cfg::SHTnsConfig{T}) where T
    nlm = cfg.nlm
    qlm_test = [complex(randn(T), randn(T)) for _ in 1:nlm]
    qlm_out = similar(qlm_test)
    
    results = Dict{String, Float64}()
    
    # Benchmark SIMD Laplacian
    qlm_copy = copy(qlm_test)
    t_simd = @elapsed begin
        for _ in 1:100
            simd_apply_laplacian!(cfg, qlm_copy)
        end
    end
    results["simd_laplacian"] = t_simd / 100
    
    # Benchmark threaded cos(θ)
    if Threads.nthreads() > 1
        t_threaded = @elapsed begin
            for _ in 1:10
                threaded_apply_costheta_operator!(cfg, qlm_test, qlm_out)
            end
        end
        results["threaded_costheta"] = t_threaded / 10
    end
    
    # Benchmark vectorized sparse
    matrix = mul_ct_matrix(cfg)
    x_real = [real(x) for x in qlm_test]
    x_imag = [imag(x) for x in qlm_test]
    y_real = zeros(T, nlm)
    y_imag = zeros(T, nlm)
    
    t_vectorized = @elapsed begin
        for _ in 1:10
            vectorized_sparse_matvec!(matrix, x_real, x_imag, y_real, y_imag)
        end
    end
    results["vectorized_sparse"] = t_vectorized / 10
    
    return results
end

# Export SIMD-optimized functions
export simd_apply_laplacian!,
       threaded_apply_costheta_operator!,
       vectorized_sparse_matvec!,
       auto_simd_dispatch,
       benchmark_simd_variants