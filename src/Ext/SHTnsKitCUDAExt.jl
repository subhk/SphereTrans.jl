module SHTnsKitCUDAExt

using SHTnsKit
import CUDA
using Base.Threads: Atomic, atomic_load, atomic_store!

"""
CuArray-specialized GPU helpers. These methods are loaded only when CUDA is
available, via Julia package extensions. By default, they stage data to host,
leverage CPU SHTns transforms, and copy results back to device. If the SHTns
GPU entrypoints are enabled and expect device pointers, you can switch to a
zero-copy device-pointer path.
"""

const _use_device_ptrs = Atomic{Bool}(false)

"""Enable device-pointer mode for native SHTns GPU entrypoints."""
function enable_gpu_deviceptrs!()
    atomic_store!(_use_device_ptrs, true)
    try
        # Ensure native GPU entrypoints are resolved if available
        SHTnsKit.enable_native_gpu!()
    catch
    end
    return true
end

# Auto-enable device-pointer mode if requested via ENV
try
    if get(ENV, "SHTNSKIT_GPU_PTRKIND", "") == "device"
        enable_gpu_deviceptrs!()
    end
catch
end

function SHTnsKit.synthesize_gpu(cfg::SHTnsKit.SHTnsConfig,
                                 sh_dev::CUDA.CuArray{<:Real,1})
    # Fast path: native GPU entrypoint with device pointers
    if atomic_load(_use_device_ptrs) && (atomic_load(SHTnsKit._gpu_sh2spat_ptr) != C_NULL)
        shd64 = eltype(sh_dev) === Float64 ? sh_dev : CUDA.convert(CUDA.CuArray{Float64,1}, sh_dev)
        nlat, nphi = SHTnsKit.get_nlat(cfg), SHTnsKit.get_nphi(cfg)
        spat_dev = CUDA.CuArray{Float64}(undef, nlat, nphi)
        lk = try
            SHTnsKit._get_lock(cfg)
        catch
            nothing
        end
        if lk !== nothing; Base.lock(lk); end
        try
            ccall(atomic_load(SHTnsKit._gpu_sh2spat_ptr), Cvoid,
                  (Ptr{Cvoid}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}), cfg.ptr,
                  CUDA.device_pointer(shd64), CUDA.device_pointer(spat_dev))
        finally
            if lk !== nothing; Base.unlock(lk); end
        end
        return spat_dev
    end
    # Fallback: host staging
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
    # Fast path: native GPU entrypoint with device pointers
    if atomic_load(_use_device_ptrs) && (atomic_load(SHTnsKit._gpu_spat2sh_ptr) != C_NULL)
        spatd64 = eltype(spat_dev) === Float64 ? spat_dev : CUDA.convert(CUDA.CuArray{Float64,2}, spat_dev)
        nlm = SHTnsKit.get_nlm(cfg)
        sh_dev = CUDA.CuArray{Float64}(undef, nlm)
        lk = try
            SHTnsKit._get_lock(cfg)
        catch
            nothing
        end
        if lk !== nothing; Base.lock(lk); end
        try
            ccall(atomic_load(SHTnsKit._gpu_spat2sh_ptr), Cvoid,
                  (Ptr{Cvoid}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}), cfg.ptr,
                  CUDA.device_pointer(spatd64), CUDA.device_pointer(sh_dev))
        finally
            if lk !== nothing; Base.unlock(lk); end
        end
        return sh_dev
    end
    # Fallback: host staging
    spat_host_any = Array(spat_dev)
    spat_host = spat_host_any isa Matrix{Float64} ? spat_host_any : Float64.(spat_host_any)
    sh_host = SHTnsKit.analyze(cfg, spat_host)
    nlm = SHTnsKit.get_nlm(cfg)
    sh_dev = similar(spat_dev, Float64, nlm)
    copyto!(sh_dev, sh_host)
    return sh_dev
end

end # module
