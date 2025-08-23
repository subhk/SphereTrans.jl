"""
SHTns-compatible layout utilities: mode counting and index mapping.
All functions assume non-negative orders `m` that are multiples of `mres`.
"""

"""
    nlm_calc(lmax::Integer, mmax::Integer, mres::Integer) -> Int

Number of packed coefficients for a real field (m ≥ 0 only).
"""
function nlm_calc(lmax::Integer, mmax::Integer, mres::Integer)
    lmax ≥ 0 && mmax ≥ 0 && mres ≥ 1 || throw(ArgumentError("invalid sizes"))
    mmax ≤ lmax || return 0
    s = 0
    for m in 0:mres:mmax
        s += (lmax - m + 1)
    end
    return s
end

"""
    nlm_cplx_calc(lmax::Integer, mmax::Integer, mres::Integer) -> Int

Number of coefficients for a complex field (both signs of m).
"""
function nlm_cplx_calc(lmax::Integer, mmax::Integer, mres::Integer)
    lmax ≥ 0 && mmax ≥ 0 && mres ≥ 1 || throw(ArgumentError("invalid sizes"))
    mmax ≤ lmax || return 0
    s = 0
    for m in -mmax:mres:mmax
        s += (lmax - abs(m) + 1)
    end
    return s
end

"""
    LM_index(lmax::Int, mres::Int, l::Int, m::Int) -> Int

Packed index `lm` for given `(l,m)` with `m ≥ 0` and multiple of `mres`.
Matches SHTns macro: LM(shtns, l, m).
"""
function LM_index(lmax::Int, mres::Int, l::Int, m::Int)
    m ≥ 0 || throw(ArgumentError("m must be ≥ 0 for packed layout"))
    (m % mres == 0) || throw(ArgumentError("m must be a multiple of mres"))
    (l ≥ m && l ≤ lmax) || throw(ArgumentError("require m ≤ l ≤ lmax"))
    im = m ÷ mres
    # ((((im)*(2*lmax + 2 - ((im)+1)*mres))>>1) + l)
    base = (im * (2*lmax + 2 - (im + 1)*mres)) >>> 1
    return base + l
end

"""
    LiM_index(lmax::Int, mres::Int, l::Int, im::Int) -> Int

Packed index `lm` for `(l, im)` where `m = im*mres`. Matches SHTns LiM.
"""
function LiM_index(lmax::Int, mres::Int, l::Int, im::Int)
    (im ≥ 0) || throw(ArgumentError("im must be ≥ 0"))
    m = im * mres
    (l ≥ m && l ≤ lmax) || throw(ArgumentError("require im*mres ≤ l ≤ lmax"))
    base = (im * (2*lmax + 2 - (im + 1)*mres)) >>> 1
    return base + l
end

"""
    build_li_mi(lmax::Int, mmax::Int, mres::Int)

Return `(li, mi)` arrays sized `nlm_calc(...)` with degrees and orders per packed index.
"""
function build_li_mi(lmax::Int, mmax::Int, mres::Int)
    n = nlm_calc(lmax, mmax, mres)
    li = Vector{Int}(undef, n)
    mi = Vector{Int}(undef, n)
    k = 1
    for m in 0:mres:mmax
        for l in m:lmax
            li[k] = l
            mi[k] = m
            k += 1
        end
    end
    return li, mi
end

"""
    im_from_lm(lm::Int, lmax::Int, mres::Int) -> Int

Compute `im = m/mres` from packed index `lm`.
"""
function im_from_lm(lm::Int, lmax::Int, mres::Int)
    lm ≥ 0 || throw(ArgumentError("lm must be ≥ 0"))
    im = 0
    base = 0
    while true
        block = lmax - im*mres + 1
        if lm < base + block
            return im
        end
        base += block
        im += 1
    end
end

