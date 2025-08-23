"""
    gausslegendre(n::Int)

Compute Gauss–Legendre nodes and weights for integrating functions on [-1, 1].
Returns `(x::Vector{Float64}, w::Vector{Float64})` with `length == n` where
`∫_{-1}^1 f(x) dx ≈ sum(w .* f.(x))`.
"""
function gausslegendre(n::Int)
    n > 0 || throw(ArgumentError("n must be positive"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    # Number of roots computed (symmetry)
    m = (n + 1) >>> 1
    for k in 1:m
        # Initial guess (Abramowitz & Stegun 10.18.10)
        z = cos(pi * (k - 0.25) / (n + 0.5))
        z1 = 0.0
        # Newton iterations
        for _ in 1:50
            pnm1 = 1.0
            pn = z
            # Compute Legendre P_n(z) using recurrence
            for l in 2:n
                pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
                pnm1, pn = pn, pnp1
            end
            # Derivative using stable relation
            pd = n * (z * pn - pnm1) / (z^2 - 1.0)
            z1 = z
            z -= pn / pd
            if abs(z - z1) < 1e-15
                break
            end
        end
        # Compute P_n and derivative at converged root for weights
        pnm1 = 1.0
        pn = z
        for l in 2:n
            pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
            pnm1, pn = pn, pnp1
        end
        pd = n * (z * pn - pnm1) / (z^2 - 1.0)

        x[k] = -z
        x[n - k + 1] = z
        wk = 2.0 / ((1.0 - z^2) * pd^2)
        w[k] = wk
        w[n - k + 1] = wk
    end
    return x, w
end

"""
    thetaphi_from_nodes(nlat::Int, nlon::Int)

Return `θ` and `φ` arrays where `θ ∈ [0, π]` (Gauss–Legendre nodes mapped) and
`φ ∈ [0, 2π)` equally spaced longitudes suitable for FFT-based azimuthal transforms.
"""
function thetaphi_from_nodes(nlat::Int, nlon::Int)
    x, w = gausslegendre(nlat)
    θ = acos.(x)
    φ = (2π / nlon) .* (0:(nlon-1))
    return θ, φ, x, w
end

