#!/usr/bin/env julia

"""
Parallel Spherical Harmonic Transform Roundtrip Test

This example demonstrates the distributed/parallel spherical harmonic transform
capabilities of SHTnsKit using MPI and PencilArrays. It tests the mathematical
accuracy of the forward and inverse transforms by performing roundtrip tests:
spatial field → spectral coefficients → spatial field.

The test validates three types of transforms:
1. Scalar fields (always tested)
2. Vector fields (--vector flag)  
3. QST 3D vector fields (--qst flag)

Usage:
  mpirun -np 4 julia parallel_roundtrip.jl [--vector] [--qst]
"""

using MPI           # Message Passing Interface for parallelization
using PencilArrays  # Distributed array framework for parallel computing
using PencilFFTs    # Distributed FFT operations
using SHTnsKit      # Spherical harmonic transforms

function main()
    # ===== MPI INITIALIZATION =====
    MPI.Init()
    comm = MPI.COMM_WORLD            # Global communicator
    rank = MPI.Comm_rank(comm)       # Process ID (0-based)

    # ===== PROBLEM SETUP =====
    lmax = 16                        # Maximum spherical harmonic degree
    nlat = lmax + 2                  # Latitude points (slightly over-resolved)
    nlon = 2*lmax + 1               # Longitude points (Nyquist limit)
    
    # Parse command line arguments for test selection
    do_vector = any(x -> x == "--vector", ARGS)  # Test vector transforms?
    do_qst = any(x -> x == "--qst", ARGS)        # Test Q,S,T transforms?

    # ===== SPHERICAL HARMONIC TRANSFORM CONFIGURATION =====
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    
    # ===== DISTRIBUTED ARRAY SETUP =====
    # Create a distributed pencil using PencilArrays 0.19 API
    # This distributes the spatial grid across MPI processes
    function _procgrid(p)
        # Choose near-square processor grid (pθ,pφ) for optimal load balancing
        # We want to minimize communication overhead by keeping the grid as square as possible
        best = (1, p); bestdiff = p-1
        for d in 1:p
            if p % d == 0
                d2 = div(p, d)
                if abs(d - d2) < bestdiff
                    best = (d, d2); bestdiff = abs(d - d2)
                end
            end
        end
        return best
    end
    
    p = MPI.Comm_size(comm)
    pθ, pφ = _procgrid(p)                                    # Decompose processes into 2D grid
    topo = Pencil((nlat, nlon), (pθ, pφ), comm)            # Create distributed pencil topology
    fθφ = PencilArrays.allocate(topo; eltype=Float64)       # Allocate distributed array
    fill!(fθφ, 0)
    
    # ===== TEST FIELD INITIALIZATION =====
    # Create deterministic-ish local test data for roundtrip validation
    # Each process fills its local portion with a known analytical function
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3 * (iθ + rank + 1)) + cos(0.2 * (iφ + 2))
    end
    # ===== SCALAR FIELD ROUNDTRIP TEST =====
    # Plan-based distributed analysis + synthesis roundtrip validation!
    # This tests: spatial field → spherical harmonic coefficients → spatial field
    
    # Create analysis plan for forward transform (spatial → spectral)
    aplan = SHTnsKit.DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)         # Spectral coefficient storage
    SHTnsKit.dist_analysis!(aplan, Alm, fθφ)               # Forward transform
    
    # Create synthesis plan for inverse transform (spectral → spatial) 
    spln = SHTnsKit.DistPlan(cfg, fθφ)
    fθφ_out = similar(fθφ)                                  # Output spatial field
    SHTnsKit.dist_synthesis!(spln, fθφ_out, PencilArray(Alm)) # Inverse transform
    
    # ===== ERROR ANALYSIS =====
    # Compare roundtrip result with original to validate mathematical accuracy
    fout = Array(fθφ_out); f0 = Array(fθφ)                # Convert to local arrays
    num = sum(abs2, fout .- f0); den = sum(abs2, f0) + eps() # Local error norms
    rel_local = sqrt(num / den)                             # Local relative error
    
    # Compute global error across all MPI processes
    rel_global = sqrt(MPI.Allreduce(num, +, comm) / MPI.Allreduce(den, +, comm))
    
    if rank == 0
        println("[scalar] rel_local≈$rel_local rel_global≈$rel_global")
    end

    # ===== VECTOR FIELD ROUNDTRIP TEST (OPTIONAL) =====
    if do_vector
        # Allocate distributed arrays for vector field components (Vθ, Vφ)
        Vtθφ = PencilArrays.allocate(topo; eltype=Float64); fill!(Vtθφ, 0)
        Vpθφ = PencilArrays.allocate(topo; eltype=Float64); fill!(Vpθφ, 0)
        
        # Initialize test vector field with analytical functions
        # Each component has different spatial variation for comprehensive testing
        for (iθ, iφ) in zip(eachindex(axes(Vtθφ,1)), eachindex(axes(Vtθφ,2)))
            Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)         # Linear variation in Vθ
            Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))         # Sinusoidal variation in Vφ
        end
        
        # ===== SPHEROIDAL/TOROIDAL DECOMPOSITION ROUNDTRIP =====
        # Plan-based distributed vector analysis + synthesis!
        # This tests: (Vθ,Vφ) → (S_lm, T_lm) → (Vθ,Vφ) roundtrip
        vplan = SHTnsKit.DistSphtorPlan(cfg, Vtθφ)
        Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)      # Spheroidal coefficients
        Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)      # Toroidal coefficients
        
        # Forward transform: spatial vector → spheroidal/toroidal coefficients
        SHTnsKit.dist_spat_to_SHsphtor!(vplan, Slm, Tlm, Vtθφ, Vpθφ)
        
        # Inverse transform: coefficients → spatial vector components
        Vt_out = similar(Vtθφ); Vp_out = similar(Vpθφ)
        SHTnsKit.dist_SHsphtor_to_spat!(vplan, Vt_out, Vp_out, Slm, Tlm)
        
        # ===== VECTOR ERROR ANALYSIS =====
        # Analyze errors separately for each vector component
        T1 = Array(Vt_out); P1 = Array(Vp_out)             # Roundtrip results
        T0 = Array(Vtθφ); P0 = Array(Vpθφ)                 # Original data
        
        # Local errors for each component
        num_t = sum(abs2, T1 .- T0); den_t = sum(abs2, T0) + eps()
        num_p = sum(abs2, P1 .- P0); den_p = sum(abs2, P0) + eps()
        rl_t = sqrt(num_t / den_t); rl_p = sqrt(num_p / den_p)
        
        # Global errors across all processes
        rg_t = sqrt(MPI.Allreduce(num_t, +, comm) / MPI.Allreduce(den_t, +, comm))
        rg_p = sqrt(MPI.Allreduce(num_p, +, comm) / MPI.Allreduce(den_p, +, comm))
        
        if rank == 0
            println("[vector] Vt rel_local≈$rl_t rel_global≈$rg_t; Vp rel_local≈$rl_p rel_global≈$rg_p")
        end
    end

    # ===== QST 3D VECTOR FIELD ROUNDTRIP TEST (OPTIONAL) =====
    if do_qst
        # Build simple synthetic 3D vector field (Vr, Vθ, Vφ)
        # This tests the full 3D vector transform including the radial component
        Vrθφ = PencilArrays.allocate(topo; eltype=Float64); fill!(Vrθφ, 0)
        Vtθφ = PencilArrays.allocate(topo; eltype=Float64); fill!(Vtθφ, 0)
        Vpθφ = PencilArrays.allocate(topo; eltype=Float64); fill!(Vpθφ, 0)
        
        # Initialize 3D vector field with different analytical functions for each component
        # This provides a comprehensive test of the Q-S-T decomposition
        for (iθ, iφ) in zip(eachindex(axes(Vrθφ,1)), eachindex(axes(Vrθφ,2)))
            Vrθφ[iθ, iφ] = 0.3*sin(0.1*(iθ+1)) + 0.2*cos(0.05*(iφ+1))  # Mixed trig in Vr
            Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)                     # Linear in Vθ  
            Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))                     # Process-dependent Vφ
        end
        
        # ===== Q-S-T DECOMPOSITION ROUNDTRIP =====
        # Plan-based distributed 3D vector analysis + synthesis!
        # This tests: (Vr,Vθ,Vφ) → (Q_lm, S_lm, T_lm) → (Vr,Vθ,Vφ) roundtrip
        qplan = SHTnsKit.DistQstPlan(cfg, Vrθφ)
        Qlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)      # Q coefficients (radial-like)
        Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)      # S coefficients (spheroidal)
        Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)      # T coefficients (toroidal)
        
        # Forward transform: 3D spatial vector → Q-S-T spectral coefficients
        SHTnsKit.dist_spat_to_SHqst!(qplan, Qlm, Slm, Tlm, Vrθφ, Vtθφ, Vpθφ)
        
        # Inverse transform: Q-S-T coefficients → 3D spatial vector components
        Vr_out = similar(Vrθφ); Vt_out = similar(Vtθφ); Vp_out = similar(Vpθφ)
        SHTnsKit.dist_SHqst_to_spat!(qplan, Vr_out, Vt_out, Vp_out, Qlm, Slm, Tlm)
        
        # ===== 3D VECTOR ERROR ANALYSIS =====
        # Analyze roundtrip errors for all three vector components
        r0 = Array(Vrθφ); r1 = Array(Vr_out)               # Radial component
        t0 = Array(Vtθφ); t1 = Array(Vt_out)               # Theta component
        p0 = Array(Vpθφ); p1 = Array(Vp_out)               # Phi component
        
        # Local error norms for each component
        num_r = sum(abs2, r1 .- r0); den_r = sum(abs2, r0) + eps()
        num_t = sum(abs2, t1 .- t0); den_t = sum(abs2, t0) + eps()
        num_p = sum(abs2, p1 .- p0); den_p = sum(abs2, p0) + eps()
        rl_r = sqrt(num_r / den_r); rl_t = sqrt(num_t / den_t); rl_p = sqrt(num_p / den_p)
        
        # Global error norms across all MPI processes
        rg_r = sqrt(MPI.Allreduce(num_r, +, comm) / MPI.Allreduce(den_r, +, comm))
        rg_t = sqrt(MPI.Allreduce(num_t, +, comm) / MPI.Allreduce(den_t, +, comm))
        rg_p = sqrt(MPI.Allreduce(num_p, +, comm) / MPI.Allreduce(den_p, +, comm))
        
        if rank == 0
            println("[qst] Vr rel_local≈$rl_r rel_global≈$rg_r; Vt rel_local≈$rl_t rel_global≈$rg_t; Vp rel_local≈$rl_p rel_global≈$rg_p")
        end
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
