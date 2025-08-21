using Test
using SHTnsKit

# Build ZYZ rotation matrix
function rot_zyz(alpha, beta, gamma)
    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta), sin(beta)
    cg, sg = cos(gamma), sin(gamma)
    Rz1 = [ca -sa 0; sa ca 0; 0 0 1]
    Ry  = [cb 0 sb; 0 1 0; -sb 0 cb]
    Rz2 = [cg -sg 0; sg cg 0; 0 0 1]
    return Rz1 * Ry * Rz2
end

function sph_to_cart(theta, phi)
    s, c = sin(theta), cos(theta)
    sp, cp = sin(phi), cos(phi)
    return (s*cp, s*sp, c)
end

function cart_to_sph(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    r == 0 && return (0.0, 0.0)
    theta = acos(clamp(z/r, -1, 1))
    phi = atan(y, x)
    phi < 0 && (phi += 2π)
    return (theta, phi)
end

@testset "Spatial vs spectral rotation" begin
    cfg = create_gauss_config(10, 10)
    # use a denser grid to reduce interpolation error
    set_grid!(cfg, 128, 256)
    n = length(SHTnsKit._cplx_lm_indices(cfg))
    coeffs = zeros(ComplexF64, n)
    # Fill random coefficients
    for i in 1:n
        coeffs[i] = 0.1 * randn() + 0.1im * randn()
    end
    # Synthesize spatial field
    f = cplx_sh_to_spat(cfg, coeffs)
    # Spectral rotation
    coeffs_rot = copy(coeffs)
    α, β, γ = 0.31, 0.21, 0.17
    rotate_complex!(cfg, coeffs_rot; alpha=α, beta=β, gamma=γ)
    f_spec = cplx_sh_to_spat(cfg, coeffs_rot)

    # Spatial rotation: g(θ,φ) = f(R^{-1}·x(θ,φ))
    R = rot_zyz(α, β, γ)
    Rt = transpose(R)  # inverse for rotation matrix
    g = Matrix{ComplexF64}(undef, get_nlat(cfg), get_nphi(cfg))
    for i in 1:get_nlat(cfg)
        θ = get_theta(cfg, i)
        for j in 1:get_nphi(cfg)
            φ = get_phi(cfg, j)
            x, y, z = sph_to_cart(θ, φ)
            xr, yr, zr = Rt * [x, y, z]
            θp, φp = cart_to_sph(xr, yr, zr)
            g[i, j] = SHTnsKit.interpolate_to_point(cfg, f, θp, φp)
        end
    end

    # Compare complex fields
    num = maximum(abs.(f_spec .- g))
    den = maximum(abs.(f_spec)) + eps()
    @test num/den < 2e-2  # interpolation error dominates; sanity threshold
    destroy_config(cfg)
end
