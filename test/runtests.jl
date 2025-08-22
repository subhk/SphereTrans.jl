using Test
using SHTnsKit

include("test_complex_transforms.jl")
include("test_vector_transforms.jl")
include("test_spectral_ops.jl")
include("test_complex_vector.jl")
include("test_real_parity.jl")
include("test_rotation.jl")
include("test_rotation_spatial.jl")
include("test_ad_zygote.jl")
include("test_ad_forwarddiff.jl")
include("test_point_evaluation.jl")
include("test_matrix_operators.jl")
include("test_truncated_transforms.jl")
include("test_profiling.jl")
