using SHTnsKit
using Random

function profile_complex(lmax::Int, mmax::Int; reps=10)
    println("Config: lmax=$lmax, mmax=$mmax")
    cfg = create_gauss_config(lmax, mmax)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    rng = MersenneTwister(2025)
    sh = [randn(rng) + randn(rng)*im for _ in 1:n]
    spat = allocate_complex_spatial(cfg)
    coeffs = similar(sh)

    # Warmup
    synthesize_complex(cfg, sh)
    analyze_complex(cfg, synthesize_complex(cfg, sh))

    t1 = time_ns()
    for _ in 1:reps
        synthesize_complex(cfg, sh)
    end
    t2 = time_ns()
    for _ in 1:reps
        analyze_complex(cfg, synthesize_complex(cfg, sh))
    end
    t3 = time_ns()
    println("synthesize avg: ", round((t2 - t1)/1e9/reps, digits=6), " s")
    println("analyze(synthesize(.)) avg: ", round((t3 - t2)/1e9/reps, digits=6), " s")
    destroy_config(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    profile_complex(32, 32; reps=5)
    profile_complex(48, 32; reps=5)
end

