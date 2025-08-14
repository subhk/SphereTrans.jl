module SHTnsKit_jll

using JLLWrappers
using Libdl
export libshtns, libshtns_omp

JLLWrappers.@generate_wrapper_header("SHTnsKit")
JLLWrappers.@declare_library_product(libshtns, "libshtns")
JLLWrappers.@declare_library_product(libshtns_omp, "libshtns_omp")

function __init__()
    JLLWrappers.@generate_init_header()
    # Use platform-appropriate extension
    libname = "lib/" * ("libshtns." * Libdl.dlext)
    JLLWrappers.@init_library_product(
        libshtns,
        libname,
        RTLD_LAZY | RTLD_DEEPBIND,
    )
    libname_omp = "lib/" * ("libshtns_omp." * Libdl.dlext)
    JLLWrappers.@init_library_product(
        libshtns_omp,
        libname_omp,
        RTLD_LAZY | RTLD_DEEPBIND,
    )
    JLLWrappers.@generate_init_footer()
end

end # module SHTnsKit_jll
