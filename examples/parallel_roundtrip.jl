#!/usr/bin/env julia

using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    lmax = 16
    nlat = lmax + 2
    nlon = 2*lmax + 1
    do_vector = any(x -> x == "--vector", ARGS)
    do_qst = any(x -> x == "--qst", ARGS)

    cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    Pθφ = Pencil((:θ, :φ), (nlat, nlon); comm)

    # Scalar roundtrip (plan-based)
    fθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
    # Deterministic-ish local fill
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3 * (iθ + rank + 1)) + cos(0.2 * (iφ + 2))
    end
    # Plan-based distributed analysis + synthesis!
    aplan = SHTnsKit.DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    SHTnsKit.dist_analysis!(aplan, Alm, fθφ)
    spln = SHTnsKit.DistPlan(cfg, fθφ)
    fθφ_out = similar(fθφ)
    SHTnsKit.dist_synthesis!(spln, fθφ_out, PencilArray(Alm))
    fout = Array(fθφ_out); f0 = Array(fθφ)
    num = sum(abs2, fout .- f0); den = sum(abs2, f0) + eps()
    rel_local = sqrt(num / den)
    rel_global = sqrt(MPI.Allreduce(num, +, comm) / MPI.Allreduce(den, +, comm))
    if rank == 0
        println("[scalar] rel_local≈$rel_local rel_global≈$rel_global")
    end

    if do_vector
        Vtθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        Vpθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        for (iθ, iφ) in zip(eachindex(axes(Vtθφ,1)), eachindex(axes(Vtθφ,2)))
            Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)
            Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))
        end
        # Plan-based distributed vector analysis + synthesis!
        vplan = SHTnsKit.DistSphtorPlan(cfg, Vtθφ)
        Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        SHTnsKit.dist_spat_to_SHsphtor!(vplan, Slm, Tlm, Vtθφ, Vpθφ)
        Vt_out = similar(Vtθφ); Vp_out = similar(Vpθφ)
        SHTnsKit.dist_SHsphtor_to_spat!(vplan, Vt_out, Vp_out, Slm, Tlm)
        T1 = Array(Vt_out); P1 = Array(Vp_out)
        T0 = Array(Vtθφ); P0 = Array(Vpθφ)
        num_t = sum(abs2, T1 .- T0); den_t = sum(abs2, T0) + eps()
        num_p = sum(abs2, P1 .- P0); den_p = sum(abs2, P0) + eps()
        rl_t = sqrt(num_t / den_t); rl_p = sqrt(num_p / den_p)
        rg_t = sqrt(MPI.Allreduce(num_t, +, comm) / MPI.Allreduce(den_t, +, comm))
        rg_p = sqrt(MPI.Allreduce(num_p, +, comm) / MPI.Allreduce(den_p, +, comm))
        if rank == 0
            println("[vector] Vt rel_local≈$rl_t rel_global≈$rg_t; Vp rel_local≈$rl_p rel_global≈$rg_p")
        end
    end

    if do_qst
        # Build simple synthetic 3D field
        Vrθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        Vtθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        Vpθφ = PencilArrays.zeros(Pθφ; eltype=Float64)
        for (iθ, iφ) in zip(eachindex(axes(Vrθφ,1)), eachindex(axes(Vrθφ,2)))
            Vrθφ[iθ, iφ] = 0.3*sin(0.1*(iθ+1)) + 0.2*cos(0.05*(iφ+1))
            Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)
            Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))
        end
        qplan = SHTnsKit.DistQstPlan(cfg, Vrθφ)
        Qlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        SHTnsKit.dist_spat_to_SHqst!(qplan, Qlm, Slm, Tlm, Vrθφ, Vtθφ, Vpθφ)
        Vr_out = similar(Vrθφ); Vt_out = similar(Vtθφ); Vp_out = similar(Vpθφ)
        SHTnsKit.dist_SHqst_to_spat!(qplan, Vr_out, Vt_out, Vp_out, Qlm, Slm, Tlm)
        # Errors
        r0 = Array(Vrθφ); r1 = Array(Vr_out)
        t0 = Array(Vtθφ); t1 = Array(Vt_out)
        p0 = Array(Vpθφ); p1 = Array(Vp_out)
        num_r = sum(abs2, r1 .- r0); den_r = sum(abs2, r0) + eps()
        num_t = sum(abs2, t1 .- t0); den_t = sum(abs2, t0) + eps()
        num_p = sum(abs2, p1 .- p0); den_p = sum(abs2, p0) + eps()
        rl_r = sqrt(num_r / den_r); rl_t = sqrt(num_t / den_t); rl_p = sqrt(num_p / den_p)
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

abspath(PROGRAM_FILE) == @__FILE__ && main()
