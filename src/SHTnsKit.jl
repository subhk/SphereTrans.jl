
module SHTnsKit

using LinearAlgebra
using FFTW
using Base.Threads

include("fftutils.jl")
include("layout.jl")
include("gausslegendre.jl")
include("legendre.jl")
include("plan.jl")
include("normalization.jl")
include("config.jl")
include("transform.jl")
include("complex_packed.jl")
include("vector.jl")
include("operators.jl")
include("rotations.jl")
include("local.jl")
include("diagnostics.jl")
include("api_compat.jl")
include("parallel_dense.jl")

export SHTConfig, create_gauss_config, destroy_config
export analysis, synthesis
export SHTPlan, analysis!, synthesis!
export spat_to_SHsphtor!, SHsphtor_to_spat!
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm, LM_cplx_index
export spat_cplx_to_SH, SH_to_spat_cplx, SH_to_point_cplx
export spat_to_SHsphtor, SHsphtor_to_spat, SHsph_to_spat, SHtor_to_spat, SH_to_grad_spat
export spat_to_SHqst, SHqst_to_spat, spat_cplx_to_SHqst, SHqst_to_spat_cplx
export SHsphtor_to_spat_l, spat_to_SHsphtor_l, SHsph_to_spat_l, SHtor_to_spat_l
export spat_to_SHsphtor_ml, SHsphtor_to_spat_ml
export spat_to_SHqst_l, SHqst_to_spat_l, spat_to_SHqst_ml, SHqst_to_spat_ml
export SHsphtor_to_spat_cplx, spat_cplx_to_SHsphtor
export mul_ct_matrix, st_dt_matrix, SH_mul_mx
export SH_to_lat, SHqst_to_lat
export SH_Zrotate
export SH_Yrotate, SH_Yrotate90, SH_Xrotate90
export SHTRotation, shtns_rotation_create, shtns_rotation_destroy
export shtns_rotation_set_angles_ZYZ, shtns_rotation_set_angles_ZXZ
export shtns_rotation_wigner_d_matrix, shtns_rotation_apply_cplx, shtns_rotation_apply_real, shtns_rotation_set_angle_axis
export energy_scalar, energy_vector, enstrophy, vorticity_spectral, vorticity_grid
export grid_energy_scalar, grid_energy_vector, grid_enstrophy
export energy_scalar_l_spectrum, energy_scalar_m_spectrum
export energy_vector_l_spectrum, energy_vector_m_spectrum
export enstrophy_l_spectrum, enstrophy_m_spectrum
export energy_scalar_lm, energy_vector_lm, enstrophy_lm
export grad_energy_scalar_alm, grad_energy_vector_Slm_Tlm, grad_enstrophy_Tlm
export grad_grid_energy_scalar_field, grad_grid_energy_vector_fields, grad_grid_enstrophy_zeta
export energy_scalar_packed, grad_energy_scalar_packed
export energy_vector_packed, grad_energy_vector_packed
export loss_vorticity_grid, grad_loss_vorticity_Tlm, loss_and_grad_vorticity_Tlm
export prepare_plm_tables!, enable_plm_tables!, disable_plm_tables!

# Optional LoopVectorization-powered helpers (defined via extension)
export analysis_turbo, synthesis_turbo
export turbo_apply_laplacian!, benchmark_turbo_vs_simd

# AD convenience wrappers (populated via extensions)
export zgrad_scalar_energy, zgrad_vector_energy, zgrad_enstrophy_Tlm
export fdgrad_scalar_energy, fdgrad_vector_energy
export zgrad_rotation_angles_real, zgrad_rotation_angles_cplx
export dist_analysis, dist_synthesis
export dist_scalar_roundtrip!, dist_vector_roundtrip!
export DistPlan, dist_synthesis!
export DistAnalysisPlan, dist_analysis!
export DistSphtorPlan, dist_spat_to_SHsphtor!, dist_SHsphtor_to_spat!
export DistQstPlan, dist_spat_to_SHqst!, dist_SHqst_to_spat!
export dist_SH_to_lat, dist_SH_to_point, dist_SHqst_to_point
export dist_spat_to_SH_packed, dist_SH_packed_to_spat
export dist_spat_cplx_to_SH, dist_SH_to_spat_cplx
export dist_SHqst_to_lat

# Default fallbacks if extensions are not loaded
zgrad_scalar_energy(::SHTConfig, ::AbstractMatrix) = error("Zygote extension not loaded")
zgrad_vector_energy(::SHTConfig, ::AbstractMatrix, ::AbstractMatrix) = error("Zygote extension not loaded")
zgrad_enstrophy_Tlm(::SHTConfig, ::AbstractMatrix) = error("Zygote extension not loaded")
fdgrad_scalar_energy(::SHTConfig, ::AbstractMatrix) = error("ForwardDiff extension not loaded")
fdgrad_vector_energy(::SHTConfig, ::AbstractMatrix, ::AbstractMatrix) = error("ForwardDiff extension not loaded")
zgrad_rotation_angles_real(::SHTConfig, ::AbstractVector, ::Real, ::Real, ::Real) = error("Zygote extension not loaded")
zgrad_rotation_angles_cplx(::Integer, ::Integer, ::AbstractVector, ::Real, ::Real, ::Real) = error("Zygote extension not loaded")
dist_analysis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_roundtrip!(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_vector_roundtrip!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_lat(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_point(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_SHqst_to_point(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_spat_to_SH_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_packed_to_spat(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spat_cplx_to_SH(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_SH_to_spat_cplx(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHqst_to_lat(::SHTConfig, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

# LoopVectorization extension fallbacks
analysis_turbo(::SHTConfig, ::AbstractMatrix) = error("LoopVectorization extension not loaded")
synthesis_turbo(::SHTConfig, ::AbstractMatrix; real_output::Bool=true) = error("LoopVectorization extension not loaded")
turbo_apply_laplacian!(::SHTConfig, ::Any) = error("LoopVectorization extension not loaded")
benchmark_turbo_vs_simd(::SHTConfig; kwargs...) = error("LoopVectorization extension not loaded")
export shtns_verbose, shtns_print_version, shtns_get_build_info
export shtns_init, shtns_create, shtns_set_grid, shtns_set_grid_auto, shtns_create_with_grid
export shtns_use_threads, shtns_reset, shtns_destroy, shtns_unset_grid, shtns_robert_form
export sh00_1, sh10_ct, sh11_st, shlm_e1, shtns_gauss_wts
export shtns_print_cfg, legendre_sphPlm_array, legendre_sphPlm_deriv_array
export shtns_malloc, shtns_free, shtns_set_many

end # module SHTnsKit
