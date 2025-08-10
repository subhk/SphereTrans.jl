module SHTnsKitCUDAExt

using SHTnsKit
import CUDA

"""
CuArray-specialized GPU helpers. These methods are loaded only when CUDA is
available, via Julia package extensions. They currently stage data to host,
leverage CPU SHTns transforms, and copy results back to device.
"""

function SHTnsKit.synthesize_gpu(cfg::SHTnsKit.SHTnsConfig,
                                 sh_dev::CUDA.CuArray{<:Real,1})
    sh_host_any = Array(sh_dev)
    sh_host = sh_host_any isa Vector{Float64} ? sh_host_any : Float64.(sh_host_any)
    spat_host = SHTnsKit.synthesize(cfg, sh_host)
    nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)
    spat_dev = similar(sh_dev, Float64, nlat, nphi)
    copyto!(spat_dev, spat_host)
    return spat_dev
end

function SHTnsKit.analyze_gpu(cfg::SHTnsKit.SHTnsConfig,
                              spat_dev::CUDA.CuArray{<:Real,2})
    spat_host_any = Array(spat_dev)
    spat_host = spat_host_any isa Matrix{Float64} ? spat_host_any : Float64.(spat_host_any)
    sh_host = SHTnsKit.analyze(cfg, spat_host)
    nlm = SHTnsKit.get_nlm(cfg)
    sh_dev = similar(spat_dev, Float64, nlm)
    copyto!(sh_dev, sh_host)
    return sh_dev
end

end # module

