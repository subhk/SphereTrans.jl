using SHTnsKit
using Zygote
using Random

println("=== Zygote AD: complex vector example ===")

# Config (keep modest size for a quick demo)
cfg = create_gauss_config(8, 8)
n = length(SHTnsKit._cplx_lm_indices(cfg))
rng = MersenneTwister(42)

# Build a target vector field from random complex S,T
S_target = [0.3randn(rng) + 0.3im*randn(rng) for _ in 1:n]
T_target = [0.3randn(rng) + 0.3im*randn(rng) for _ in 1:n]
uθ_tar, uφ_tar = cplx_synthesize_vector(cfg, S_target, T_target)

# Loss: 0.5 * ||uθ(S,T)-uθ_tar||^2 + 0.5 * ||uφ(S,T)-uφ_tar||^2
function loss_vec(S::AbstractVector{<:Complex}, T::AbstractVector{<:Complex})
    uθ, uφ = cplx_synthesize_vector(cfg, S, T)
    dθ = uθ .- uθ_tar
    dφ = uφ .- uφ_tar
    return 0.5 * (sum(abs2, dθ) + sum(abs2, dφ))
end

# Gradient and one simple optimization loop
S = zeros(ComplexF64, n)
T = zeros(ComplexF64, n)

for iter in 1:10
    L, back = Zygote.pullback(loss_vec, S, T)
    gS, gT = back(1.0)
    η = 0.1
    S .-= η .* gS
    T .-= η .* gT
    println("iter=", iter, ", loss=", L)
end

destroy_config(cfg)
println("Done.")

