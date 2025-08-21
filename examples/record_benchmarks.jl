using SHTnsKit
using Random
using Printf

function bench(cfg; reps=5)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    rng = MersenneTwister(123)
    sh = [randn(rng) + randn(rng)*im for _ in 1:n]
    # Warmup
    cplx_sh_to_spat(cfg, sh)
    cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, sh))
    # synth
    t1 = time_ns(); for _ in 1:reps; cplx_sh_to_spat(cfg, sh); end; t2 = time_ns()
    # analyze(synthesize())
    t_an1 = time_ns(); for _ in 1:reps; cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, sh)); end; t_an2 = time_ns()
    return ((t2 - t1)/1e9/reps, (t_an2 - t_an1)/1e9/reps)
end

function run_case(lmax; reps=5)
    cfg = create_gauss_config(lmax, lmax)
    # single-thread style
    set_threading!(false)
    set_fft_threads(1)
    st_syn, st_an = bench(cfg; reps=reps)
    # multi-thread style
    set_threading!(true)
    set_fft_threads(Threads.nthreads())
    mt_syn, mt_an = bench(cfg; reps=reps)
    destroy_config(cfg)
    return st_syn, mt_syn, st_an, mt_an
end

function main()
    nT = Threads.nthreads()
    println("| lmax=mmax | Nthreads | synth ST (s) | synth MT (s) | speedup | analyze(syn) ST (s) | analyze(syn) MT (s) | speedup |")
    println("|-----------|----------|--------------|--------------|---------|---------------------|---------------------|---------|")
    for l in (24, 32, 40)
        st_syn, mt_syn, st_an, mt_an = run_case(l)
        s1 = st_syn; s2 = mt_syn; a1 = st_an; a2 = mt_an
        @printf("| %-9d | %-8d | %12.6f | %12.6f | %7.2f | %19.6f | %19.6f | %7.2f |\n",
                l, nT, s1, s2, s1/max(s2,1e-12), a1, a2, a1/max(a2,1e-12))
    end
end

abspath(PROGRAM_FILE) == @__FILE__ && main()

