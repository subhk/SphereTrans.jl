using SHTnsKit
using Random

function bench(cfg; reps=5)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    rng = MersenneTwister(123)
    sh = [randn(rng) + randn(rng)*im for _ in 1:n]
    # Warmup
    cplx_sh_to_spat(cfg, sh)
    cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, sh))
    # Time synthesize only (heavier call), reps times
    t1 = time_ns()
    for _ in 1:reps
        cplx_sh_to_spat(cfg, sh)
    end
    t2 = time_ns()
    # Time analyze(synthesize())
    for _ in 1:reps
        cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, sh))
    end
    t3 = time_ns()
    return ((t2 - t1)/1e9/reps, (t3 - t2)/1e9/reps)
end

function run_case(lmax)
    cfg = create_gauss_config(lmax, lmax)
    println("\n=== lmax=mmax=$lmax, grid=($(get_nlat(cfg)),$(get_nphi(cfg))) ===")
    # Single-thread style: disable loop threading; FFTW threads = 1
    set_threading!(false)
    set_fft_threads(1)
    t_syn_st, t_an_st = bench(cfg)
    println("single-thread synthesize: ", round(t_syn_st, digits=6), " s")
    println("single-thread analyze(synthesize): ", round(t_an_st, digits=6), " s")

    # Multi-thread style: enable loop threading; FFTW threads = Threads.nthreads()
    set_threading!(true)
    set_fft_threads(Threads.nthreads())
    t_syn_mt, t_an_mt = bench(cfg)
    println("multi-thread synthesize: ", round(t_syn_mt, digits=6), " s  (x", round(t_syn_st/max(t_syn_mt,1e-12), digits=2), ")")
    println("multi-thread analyze(synthesize): ", round(t_an_mt, digits=6), " s  (x", round(t_an_st/max(t_an_mt,1e-12), digits=2), ")")
    destroy_config(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia threads: ", Threads.nthreads())
    for l in (24, 32, 40)
        run_case(l)
    end
end

