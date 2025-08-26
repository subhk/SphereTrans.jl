module SHTnsKitForwardDiffExt

using ForwardDiff
using PencilArrays
using ..SHTnsKit

"""
    fdgrad_scalar_energy(cfg, f) -> ∂E/∂f

ForwardDiff gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.
Returns an array with the same size as `f`.
"""
function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    nlat, nlon = size(f)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, reshape(x, nlat, nlon)))
    g = ForwardDiff.gradient(loss, vec(f))
    return reshape(g, nlat, nlon)
end

##########
# PencilArray distributed convenience wrappers
##########

function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::PencilArrays.PencilArray)
    # Flatten local block; gradient is computed locally and returned in same Pencil layout
    nlat = length(axes(fθφ, 1)); nlon = length(axes(fθφ, 2))
    function loss_flat(z)
        xloc = reshape(z, nlat, nlon)
        return SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, PencilArrays.PencilArray(xloc)))
    end
    g = ForwardDiff.gradient(loss_flat, vec(Array(fθφ)))
    gl = reshape(g, nlat, nlon)
    gout = similar(fθφ)
    copyto!(gout, gl)
    return gout
end

function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArrays.PencilArray, Vpθφ::PencilArrays.PencilArray)
    nlat = length(axes(Vtθφ, 1)); nlon = length(axes(Vtθφ, 2))
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, PencilArrays.PencilArray(Xt), PencilArrays.PencilArray(Xp))
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    z0 = vcat(vec(Array(Vtθφ)), vec(Array(Vpθφ)))
    g = ForwardDiff.gradient(loss_flat, z0)
    gVt = reshape(view(g, 1:nlat*nlon), nlat, nlon)
    gVp = reshape(view(g, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
    GVt = similar(Vtθφ); GVp = similar(Vpθφ)
    copyto!(GVt, gVt); copyto!(GVp, gVp)
    return GVt, GVp
end

"""
    fdgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = size(Vt)
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Xt, Xp)
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    z0 = vcat(vec(Vt), vec(Vp))
    g = ForwardDiff.gradient(loss_flat, z0)
    gVt = reshape(view(g, 1:nlat*nlon), nlat, nlon)
    gVp = reshape(view(g, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
    return gVt, gVp
end

end # module
