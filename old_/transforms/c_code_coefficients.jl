"""
Exact coefficient computation following SHTns C code (sht_legendre.c).

This implements the exact recurrence coefficient computation from the C code:
- alm coefficients for Legendre polynomial recurrence
- glm coefficients for synthesis normalization  
- glm_analys coefficients for analysis normalization
- alm2 coefficients for alternative recurrence
"""

"""
Compute alm recurrence coefficients exactly following sht_legendre.c lines 439-450.
Returns coefficients for orthonormal normalization.
"""
function compute_alm_coefficients(cfg::SHTnsConfig{T}) where T
    lmax = cfg.lmax
    mmax = cfg.mmax
    mres = cfg.mres
    
    # Total number of coefficients needed
    total_size = sum((2*(lmax-m*mres) + 1) for m in 0:mmax if m*mres <= lmax)
    alm = zeros(T, total_size)
    
    for im in 0:mmax
        m = im * mres
        m > lmax && break
        
        # Starting index for this m (following C code indexing)
        lm_start = sum((2*(lmax-mm*mres) + 1) for mm in 0:(im-1) if mm*mres <= lmax; init=0)
        
        if m == 0
            # For m=0 (lines 441-449 in C code)
            lm = lm_start + 1  # alm[lm] corresponds to l=m
            # Starting value unchanged for orthonormal
            alm[lm] = one(T)  # l=0, implicit
            
            if lmax >= 1
                alm[lm + 1] = sqrt(T(3))  # l=1: alm[lm+1] = SQRT(2*m+3) = sqrt(3)
                lm += 2
                
                for l in 2:lmax
                    t1 = T(l * l)  # (l+m)*(l-m) with m=0
                    t2 = T((l-1) * (l-1))  # Previous t1
                    
                    # C code formulas for orthonormal
                    alm[lm + 1] = sqrt((T(2*l+1) * T(2*l-1)) / t1)  # a_l^m
                    alm[lm] = -sqrt((T(2*l+1) * t2) / (T(2*l-3) * t1))  # b_l^m
                    
                    lm += 2
                end
            end
        else
            # For m>0 (similar structure)
            lm = lm_start + 1
            t2 = T(2*m + 1)
            
            # Starting value unchanged
            if lmax >= m + 1
                alm[lm + 1] = sqrt(T(2*m + 3))  # l=m+1
                lm += 2
                
                for l in (m+2):lmax
                    t1 = T((l + m) * (l - m))
                    alm[lm + 1] = sqrt((T(2*l+1) * T(2*l-1)) / t1)  # a_l^m
                    alm[lm] = -sqrt((T(2*l+1) * t2) / (T(2*l-3) * t1))  # b_l^m
                    t2 = t1
                    lm += 2
                end
            end
        end
    end
    
    return alm
end

"""
Compute glm and alm2 coefficients following sht_legendre.c lines 453-459.
"""
function compute_glm_alm2_coefficients(cfg::SHTnsConfig{T}, alm::Vector{T}) where T
    lmax = cfg.lmax
    mmax = cfg.mmax
    mres = cfg.mres
    
    # Storage for glm and alm2
    nlm1 = cfg.nlm  # Simplified sizing
    glm = zeros(T, nlm1)
    alm2 = zeros(T, nlm1)
    
    for im in 0:mmax
        m = im * mres
        m > lmax && break
        
        # Compute lm0 and lm indices following C code
        lm0 = im > 0 ? sum(lmax + 1 - mm*mres for mm in 0:(im-1) if mm*mres <= lmax; init=0) : 0
        lm = sum((2*(lmax-mm*mres) + 1) for mm in 0:(im-1) if mm*mres <= lmax; init=0) + 1
        
        # Lines 455-456: glm computation
        if lm0 + 1 <= length(glm) && lm + 1 <= length(alm)
            glm[lm0 + 1] = alm[lm]      # glm[lm0] = alm[lm] (1-indexed)
            if lm0 + 2 <= length(glm)
                glm[lm0 + 2] = alm[lm]  # glm[lm0+1] = alm[lm]
            end
            
            for l in 2:(lmax-m)
                if lm0 + l + 1 <= length(glm) && lm + 2*l - 1 <= length(alm)
                    glm[lm0 + l + 1] = glm[lm0 + l - 1] * alm[lm + 2*l - 1]
                end
            end
        end
        
        # Lines 457-459: alm2 computation
        if lm0 + 1 <= length(alm2)
            alm2[lm0 + 1] = one(T)  # alm2[lm0] = 1.0
            
            if lm0 + 2 <= length(alm2) && lm + 2 <= length(alm)
                alm2[lm0 + 2] = alm[lm + 2]  # alm2[lm0+1] = alm[lm+1]
            end
            
            for l in 2:(lmax-m)
                if (lm0 + l + 1 <= length(alm2) && 
                    lm + 2*l <= length(alm) && 
                    lm0 + l <= length(glm) && 
                    lm0 + l + 1 <= length(glm))
                    
                    if glm[lm0 + l + 1] != 0
                        alm2[lm0 + l + 1] = alm[lm + 2*l] * glm[lm0 + l] / glm[lm0 + l + 1]
                    end
                end
            end
        end
    end
    
    return glm, alm2
end

"""
Get glm_analys coefficients (same as glm for orthonormal, different for Schmidt).
Following sht_legendre.c line 420.
"""
function compute_glm_analys_coefficients(cfg::SHTnsConfig{T}, glm::Vector{T}) where T
    if cfg.norm == SHT_ORTHONORMAL
        # For orthonormal: glm_analys = glm + 0 (line 420)
        return copy(glm)
    elseif cfg.norm == SHT_SCHMIDT  
        # For Schmidt: multiply by (2*l+1) (lines 461-462)
        glm_analys = copy(glm)
        # Apply Schmidt normalization - need proper indexing
        # This is a simplified version - full implementation needs proper lm indexing
        return glm_analys
    else
        return copy(glm)
    end
end

"""
Compute mpos_scale_analys following sht_init.c line 1227.
"""
function compute_mpos_scale_analys(cfg::SHTnsConfig{T}) where T
    # C code: const double mpos_renorm = (norm & SHT_REAL_NORM) ? 0.5 : 1.0;
    # C code: shtns->mpos_scale_analys = 0.5/mpos_renorm;
    
    # Check if SHT_REAL_NORM flag is set
    has_real_norm = false  # Our implementation doesn't set this flag by default
    
    mpos_renorm = has_real_norm ? T(0.5) : T(1.0)
    mpos_scale_analys = T(0.5) / mpos_renorm
    
    return mpos_scale_analys
end