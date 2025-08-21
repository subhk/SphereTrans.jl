using SHTnsKit
using Random

function bench_pair(lmax::Int, reps::Int=5)
    cfg = create_gauss_config(lmax, lmax)
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    spat = rand(nlat, nphi)
    println("\n=== lmax=mmax=$lmax, grid=($nlat,$nphi) ===")
    # Warmup
    r = analyze_real(cfg, spat)
    spat2 = synthesize_real(cfg, r)
    c = cplx_spat_to_sh(cfg, ComplexF64.(spat))
    spatc2 = cplx_sh_to_spat(cfg, c)
    # Timings
    t1 = time_ns()
    for _ in 1:reps
        r = analyze_real(cfg, spat)
        spat2 = synthesize_real(cfg, r)
    end
    t2 = time_ns()
    for _ in 1:reps
        c = cplx_spat_to_sh(cfg, ComplexF64.(spat))
        spatc2 = cplx_sh_to_spat(cfg, c)
    end
    t3 = time_ns()
    println("real  roundtrip avg: ", round((t2 - t1)/1e9/reps, digits=6), " s")
    println("cplx  roundtrip avg: ", round((t3 - t2)/1e9/reps, digits=6), " s")
    destroy_config(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for l in (16, 24, 32)
        bench_pair(l, 5)
    end
end

