using SHTnsKit
using Random

function bench_vector_pair(lmax::Int, reps::Int=5)
    cfg = create_gauss_config(lmax, lmax)
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    u = rand(nlat, nphi)
    v = rand(nlat, nphi)
    println("\n=== Vector: lmax=mmax=$lmax, grid=($nlat,$nphi) ===")
    # Warmup
    S_r, T_r = analyze_vector_real(cfg, u, v)
    u_rt, v_rt = synthesize_vector_real(cfg, S_r, T_r)
    S_c, T_c = cplx_analyze_vector(cfg, ComplexF64.(u), ComplexF64.(v))
    u_ct, v_ct = cplx_synthesize_vector(cfg, S_c, T_c)
    # Timings
    t1 = time_ns()
    for _ in 1:reps
        S_r, T_r = analyze_vector_real(cfg, u, v)
        u_rt, v_rt = synthesize_vector_real(cfg, S_r, T_r)
    end
    t2 = time_ns()
    for _ in 1:reps
        S_c, T_c = cplx_analyze_vector(cfg, ComplexF64.(u), ComplexF64.(v))
        u_ct, v_ct = cplx_synthesize_vector(cfg, S_c, T_c)
    end
    t3 = time_ns()
    println("real  vector roundtrip avg: ", round((t2 - t1)/1e9/reps, digits=6), " s")
    println("cplx  vector roundtrip avg: ", round((t3 - t2)/1e9/reps, digits=6), " s")
    destroy_config(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for l in (16, 24)
        bench_vector_pair(l, 5)
    end
end

