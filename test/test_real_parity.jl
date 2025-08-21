using Test
using SHTnsKit

@testset "Real basis parity" begin
    cfg = create_gauss_config(6, 6)
    # Roundtrip real via real-basis
    spat = rand(get_nlat(cfg), get_nphi(cfg))
    r = analyze_real(cfg, spat)
    spat2 = synthesize_real(cfg, r)
    @test maximum(abs.(spat .- spat2)) / (maximum(abs.(spat)) + eps()) < 1e-6

    # Consistency with complex route (for real fields)
    c = cplx_spat_to_sh(cfg, ComplexF64.(spat))
    r2 = complex_to_real_coeffs(cfg, c)
    c2 = real_to_complex_coeffs(cfg, r2)
    # Complex reconstructed should match original complex coeffs for m>=0 (up to tiny numerical noise)
    @test maximum(abs.(c2 .- c)) / (maximum(abs.(c)) + eps()) < 1e-10
    destroy_config(cfg)
end

