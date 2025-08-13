module SHTnsKit_jll

using JLLWrappers
using Artifacts
import JLLWrappers: @generate_main_file_header, @generate_main_file
export libshtns, libshtns_omp

# Load wrapper for current platform
if Sys.islinux() && Sys.ARCH === :x86_64
    include("wrappers/x86_64-linux-gnu.jl")
else
    error("Platform $(Base.BinaryPlatforms.triplet(Base.BinaryPlatforms.platform_key_abi())) not supported by SHTnsKit_jll")
end

end # module SHTnsKit_jll