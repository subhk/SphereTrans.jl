module SHTnsKitCUDAExt

using SHTnsKit
import CUDA
"""
CuArray-specialized GPU helpers. These methods are loaded only when CUDA is
available, via Julia package extensions. By default, they stage data to host,
leverage CPU SHTns transforms, and copy results back to device. If the SHTns
GPU entrypoints are enabled and expect device pointers, you can switch to a
zero-copy device-pointer path.
"""

# Thread-safe boolean flag for device pointer mode
const _device_ptr_lock = ReentrantLock()
const _use_device_ptrs = Ref{Bool}(false)

@inline function _load_device_flag()
    lock(_device_ptr_lock) do
        _use_device_ptrs[]
    end
end

@inline function _store_device_flag!(value::Bool)
    lock(_device_ptr_lock) do
        _use_device_ptrs[] = value
    end
end

"""Enable device-pointer mode for native SHTns GPU entrypoints."""
function enable_gpu_deviceptrs!()
    _store_device_flag!(true)
    try
        # Ensure native GPU entrypoints are resolved if available
        enable_native_gpu!()
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

function synthesize_gpu(cfg::SHTnsConfig,
                                 sh_dev::CUDA.CuArray{<:Real,1})
    # Fast path: native GPU entrypoint with device pointers
    if _load_device_flag() && (SHTnsKit._load_ptr(SHTnsKit._gpu_sh2spat_ptr) != C_NULL)
        shd64 = eltype(sh_dev) === Float64 ? sh_dev : CUDA.convert(CUDA.CuArray{Float64,1}, sh_dev)
        nlat, nphi = get_nlat(cfg), get_nphi(cfg)
        spat_dev = CUDA.CuArray{Float64}(undef, nlat, nphi)
        lk = try
            SHTnsKit._get_lock(cfg)
        catch
            nothing
        end
        if lk !== nothing; Base.lock(lk); end
        try
            ccall(SHTnsKit._load_ptr(SHTnsKit._gpu_sh2spat_ptr), Cvoid,
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
    spat_host = synthesize(cfg, sh_host)
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    spat_dev = similar(sh_dev, Float64, nlat, nphi)
    copyto!(spat_dev, spat_host)
    return spat_dev
end

function analyze_gpu(cfg::SHTnsConfig,
                              spat_dev::CUDA.CuArray{<:Real,2})
    # Fast path: native GPU entrypoint with device pointers
    if _load_device_flag() && (SHTnsKit._load_ptr(SHTnsKit._gpu_spat2sh_ptr) != C_NULL)
        spatd64 = eltype(spat_dev) === Float64 ? spat_dev : CUDA.convert(CUDA.CuArray{Float64,2}, spat_dev)
        nlm = get_nlm(cfg)
        sh_dev = CUDA.CuArray{Float64}(undef, nlm)
        lk = try
            SHTnsKit._get_lock(cfg)
        catch
            nothing
        end
        if lk !== nothing; Base.lock(lk); end
        try
            ccall(SHTnsKit._load_ptr(SHTnsKit._gpu_spat2sh_ptr), Cvoid,
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
    sh_host = analyze(cfg, spat_host)
    nlm = get_nlm(cfg)
    sh_dev = similar(spat_dev, Float64, nlm)
    copyto!(sh_dev, sh_host)
    return sh_dev
end

# === ENHANCED GPU FUNCTIONALITY ===

"""Enhanced GPU initialization with CUDA device management."""
function initialize_gpu(device_id::Integer = 0; verbose::Bool = false)
    try
        # Set CUDA device
        CUDA.device!(device_id)
        
        # Initialize SHTns GPU
        result = gpu_init(device_id)
        if result == 0
            verbose && @info "GPU initialized successfully on CUDA device $device_id"
            # Try to enable device pointer mode
            if get(ENV, "SHTNSKIT_AUTO_DEVICEPTRS", "true") == "true"
                enable_gpu_deviceptrs!()
                verbose && @info "Enabled device pointer mode for GPU transforms"
            end
            return true
        else
            verbose && @warn "SHTns GPU initialization failed with code $result"
            return false
        end
    catch e
        verbose && @warn "GPU initialization error: $e"
        return false
    end
end

"""GPU-based complex field synthesis."""
function synthesize_complex_gpu(cfg::SHTnsConfig, 
                                        sh_dev::CUDA.CuArray{<:Complex,1})
    # Convert to host, synthesize, copy back
    sh_host = Array(sh_dev)
    sh64 = sh_host isa Vector{ComplexF64} ? sh_host : ComplexF64.(sh_host)
    spat_host = synthesize_complex(cfg, sh64)
    
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    spat_dev = similar(sh_dev, ComplexF64, nlat, nphi)
    copyto!(spat_dev, spat_host)
    return spat_dev
end

"""GPU-based complex field analysis."""
function analyze_complex_gpu(cfg::SHTnsConfig,
                                     spat_dev::CUDA.CuArray{<:Complex,2})
    # Convert to host, analyze, copy back
    spat_host = Array(spat_dev)
    spat64 = spat_host isa Matrix{ComplexF64} ? spat_host : ComplexF64.(spat_host)
    sh_host = analyze_complex(cfg, spat64)
    
    nlm = get_nlm(cfg)
    sh_dev = similar(spat_dev, ComplexF64, nlm)
    copyto!(sh_dev, sh_host)
    return sh_dev
end

"""GPU-based vector field synthesis."""
function synthesize_vector_gpu(cfg::SHTnsConfig,
                                       Slm_dev::CUDA.CuArray{<:Real,1}, 
                                       Tlm_dev::CUDA.CuArray{<:Real,1})
    # Convert to host
    Slm_host = Array(Slm_dev)
    Tlm_host = Array(Tlm_dev)
    
    # Synthesize on host
    Vt_host, Vp_host = synthesize_vector(cfg, Slm_host, Tlm_host)
    
    # Copy back to device
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    Vt_dev = similar(Slm_dev, Float64, nlat, nphi)
    Vp_dev = similar(Slm_dev, Float64, nlat, nphi)
    copyto!(Vt_dev, Vt_host)
    copyto!(Vp_dev, Vp_host)
    
    return Vt_dev, Vp_dev
end

"""GPU-based vector field analysis."""
function analyze_vector_gpu(cfg::SHTnsConfig,
                                    Vt_dev::CUDA.CuArray{<:Real,2},
                                    Vp_dev::CUDA.CuArray{<:Real,2})
    # Convert to host
    Vt_host = Array(Vt_dev)
    Vp_host = Array(Vp_dev)
    
    # Analyze on host
    Slm_host, Tlm_host = analyze_vector(cfg, Vt_host, Vp_host)
    
    # Copy back to device
    nlm = get_nlm(cfg)
    Slm_dev = similar(Vt_dev, Float64, nlm)
    Tlm_dev = similar(Vt_dev, Float64, nlm)
    copyto!(Slm_dev, Slm_host)
    copyto!(Tlm_dev, Tlm_host)
    
    return Slm_dev, Tlm_dev
end

"""GPU-based field rotation."""
function rotate_field_gpu(cfg::SHTnsConfig, 
                                  sh_dev::CUDA.CuArray{<:Real,1},
                                  alpha::Real, beta::Real, gamma::Real)
    # Convert to host, rotate, copy back
    sh_host = Array(sh_dev)
    sh_rotated_host = rotate_field(cfg, sh_host, alpha, beta, gamma)
    
    sh_rotated_dev = similar(sh_dev, Float64)
    copyto!(sh_rotated_dev, sh_rotated_host)
    return sh_rotated_dev
end

"""GPU-based spatial field rotation."""
function rotate_spatial_field_gpu(cfg::SHTnsConfig,
                                          spat_dev::CUDA.CuArray{<:Real,2},
                                          alpha::Real, beta::Real, gamma::Real)
    # Convert to host, rotate, copy back
    spat_host = Array(spat_dev)
    spat_rotated_host = rotate_spatial_field(cfg, spat_host, alpha, beta, gamma)
    
    spat_rotated_dev = similar(spat_dev, Float64)
    copyto!(spat_rotated_dev, spat_rotated_host)
    return spat_rotated_dev
end

"""GPU power spectrum computation."""
function power_spectrum_gpu(cfg::SHTnsConfig, 
                                    sh_dev::CUDA.CuArray{<:Real,1})
    sh_host = Array(sh_dev)
    power_host = power_spectrum(cfg, sh_host)
    power_dev = CUDA.CuArray(power_host)
    return power_dev
end

"""GPU gradient computation."""
function compute_gradient_gpu(cfg::SHTnsConfig, 
                                      Slm_dev::CUDA.CuArray{<:Real,1})
    Slm_host = Array(Slm_dev)
    Vt_host, Vp_host = compute_gradient(cfg, Slm_host)
    
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    Vt_dev = similar(Slm_dev, Float64, nlat, nphi)
    Vp_dev = similar(Slm_dev, Float64, nlat, nphi)
    copyto!(Vt_dev, Vt_host)
    copyto!(Vp_dev, Vp_host)
    
    return Vt_dev, Vp_dev
end

"""GPU curl computation."""
function compute_curl_gpu(cfg::SHTnsConfig,
                                  Tlm_dev::CUDA.CuArray{<:Real,1})
    Tlm_host = Array(Tlm_dev)
    Vt_host, Vp_host = compute_curl(cfg, Tlm_host)
    
    nlat, nphi = get_nlat(cfg), get_nphi(cfg)
    Vt_dev = similar(Tlm_dev, Float64, nlat, nphi)
    Vp_dev = similar(Tlm_dev, Float64, nlat, nphi)
    copyto!(Vt_dev, Vt_host)
    copyto!(Vp_dev, Vp_host)
    
    return Vt_dev, Vp_dev
end


end # module
