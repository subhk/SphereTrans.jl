##########
# PencilArray rotations
##########

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig, 
                            Alm_pencil::PencilArrays.PencilArray, alpha::Real)
    mloc = axes(Alm_pencil, 2)
    gl_m = PencilArrays.globalindices(Alm_pencil, 2)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        phase = cis(mval * alpha)
        Alm_pencil[:, jm] .*= phase
    end

    return Alm_pencil
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                            Alm_pencil::PencilArrays.PencilArray, alpha::Real,
                            R_pencil::PencilArrays.PencilArray)
    mloc = axes(Alm_pencil, 2)
    gl_m = PencilArrays.globalindices(Alm_pencil, 2)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        phase = cis(mval * alpha)
        @inbounds R_pencil[:, jm] .= phase .* Alm_pencil[:, jm]
    end
    return R_pencil
end

function SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg::SHTnsKit.SHTConfig, 
                                            Alm_pencil::PencilArrays.PencilArray, 
                                            beta::Real, 
                                            R_pencil::PencilArrays.PencilArray)

    lmax, mmax = cfg.lmax, cfg.mmax
    
    comm = PencilArrays.communicator(Alm_pencil)

    lloc = axes(Alm_pencil, 1)
    mloc = axes(Alm_pencil, 2)
    
    gl_l = PencilArrays.globalindices(Alm_pencil, 1)
    gl_m = PencilArrays.globalindices(Alm_pencil, 2)

    nm_local = length(mloc)
    counts_m = MPI.Allgather(nm_local, comm)
    displs_m = cumsum([0; counts_m[1:end-1]])
    a_full = Vector{ComplexF64}(undef, mmax + 1)

    for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        a_local = Array(view(Alm_pencil, il, :))
        MPI.Allgatherv!(a_local, a_full, counts_m, displs_m, comm)
        mm = min(lval, mmax)
        n2 = 2*lval + 1
        b = Vector{ComplexF64}(undef, n2); fill!(b, 0.0 + 0.0im)
        if lval >= 0
            k0 = SHTnsKit.norm_scale_from_orthonormal(lval, 0, cfg.norm)
            α0 = SHTnsKit.cs_phase_factor(0, true, cfg.cs_phase)
            b[0 + lval + 1] = (k0 * α0) * a_full[1]
        end

        for m in 1:mm
            km = SHTnsKit.norm_scale_from_orthonormal(lval, m, cfg.norm)
            αm = SHTnsKit.cs_phase_factor(m, true, cfg.cs_phase)
            a_int = (km * αm) * a_full[m+1]
            b[m + lval + 1] = a_int
            b[-m + lval + 1] = (-1.0)^m * conj(a_int)
        end

        dl = SHTnsKit.wigner_d_matrix(lval, float(beta))
        c = Vector{ComplexF64}(undef, n2)
        @inbounds for mi in -lval:lval
            acc = 0.0 + 0.0im
            for mp in -lval:lval
                acc += dl[mi + lval + 1, mp + lval + 1] * b[mp + lval + 1]
            end
            c[mi + lval + 1] = acc
        end

        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            if mval <= lval
                cm = c[mval + lval + 1]
                km = SHTnsKit.norm_scale_from_orthonormal(lval, mval, cfg.norm)
                αm = SHTnsKit.cs_phase_factor(mval, true, cfg.cs_phase)
                R_pencil[il, jm] = cm / (km * αm)
            else
                R_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end
    
    return R_pencil
end

function SHTnsKit.dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig,
                                  Alm_pencil::PencilArrays.PencilArray,
                                  beta::Real,
                                  R_pencil::PencilArrays.PencilArray)
    return SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg, Alm_pencil, beta, R_pencil)
end
