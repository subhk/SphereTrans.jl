"""
Lightweight plan to reuse buffers and FFT plans for SHT transforms.
Reduces allocations and improves locality without increasing peak memory.
"""
struct SHTPlan
    cfg::SHTConfig
    P::Vector{Float64}
    dPdx::Vector{Float64}
    G::Vector{ComplexF64}
    Fθk::Matrix{ComplexF64}
    fft_plan::FFTW.cFFTWPlan
    ifft_plan::FFTW.cFFTWPlan
end

"""
    SHTPlan(cfg::SHTConfig)

Create a plan with reusable buffers and in-place FFTW plans along φ (dim=2).
"""
function SHTPlan(cfg::SHTConfig)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    G = Vector{ComplexF64}(undef, cfg.nlat)
    Fθk = Matrix{ComplexF64}(undef, cfg.nlat, cfg.nlon)
    fill!(Fθk, 0)
    fft_plan = FFTW.plan_fft!(Fθk, 2)
    ifft_plan = FFTW.plan_ifft!(Fθk, 2)
    return SHTPlan(cfg, P, dPdx, G, Fθk, fft_plan, ifft_plan)
end

"""
    analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)

In-place forward scalar SHT writing coefficients into `alm_out`.
"""
function analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f,1)==nlat || throw(DimensionMismatch("f first dim must be nlat"))
    size(f,2)==nlon || throw(DimensionMismatch("f second dim must be nlon"))
    size(alm_out,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))
    # Copy f into complex buffer and FFT along φ in-place
    @inbounds for i in 1:nlat, j in 1:nlon
        plan.Fθk[i,j] = f[i,j]
    end
    FFTW.fft!(plan.fft_plan)  # in-place along dim 2
    # Compute alm
    fill!(alm_out, 0)
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_row!(plan.P, cfg.x[i], lmax, m)
            Fi = plan.Fθk[i, col]
            wi = cfg.w[i]
            @inbounds for l in m:lmax
                alm_out[l+1, col] += (wi * plan.P[l+1]) * Fi
            end
        end
        @inbounds for l in m:lmax
            alm_out[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    # Convert to cfg normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmp = similar(alm_out)
        convert_alm_norm!(tmp, alm_out, cfg; to_internal=false)
        copyto!(alm_out, tmp)
    end
    return alm_out
end

"""
    synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output=true)

In-place inverse scalar SHT writing spatial field into `f_out`.
Streams m→k directly without building a (θ×m) intermediate.
"""
function synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f_out,1)==nlat || throw(DimensionMismatch("f_out first dim must be nlat"))
    size(f_out,2)==nlon || throw(DimensionMismatch("f_out second dim must be nlon"))
    size(alm,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))
    # Convert alm to internal normalization if needed
    alm_int = alm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmp = similar(alm)
        convert_alm_norm!(tmp, alm, cfg; to_internal=true)
        alm_int = tmp
    end
    # Zero Fourier buffer
    fill!(plan.Fθk, 0)
    lmax, mmax = cfg.lmax, cfg.mmax
    inv_scaleφ = nlon / (2π)
    # Stream over m, fill k-bins directly
    for m in 0:mmax
        col = m + 1
        # Build Gm(θ)
        for i in 1:nlat
            Plm_row!(plan.P, cfg.x[i], lmax, m)
            g = 0.0 + 0.0im
            @inbounds for l in m:lmax
                g += (cfg.Nlm[l+1, col] * plan.P[l+1]) * alm_int[l+1, col]
            end
            plan.G[i] = g
        end
        # Place positive m Fourier modes
        @inbounds for i in 1:nlat
            plan.Fθk[i, col] = inv_scaleφ * plan.G[i]
        end
        # Hermitian conjugate for negative m to ensure real output
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
            end
        end
    end
    # Inverse FFT along φ in-place
    FFTW.ifft!(plan.ifft_plan)
    # Write result
    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = real(plan.Fθk[i,j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = plan.Fθk[i,j]
        end
    end
    return f_out
end

