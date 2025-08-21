using Pkg
using Pkg.Artifacts

artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
mkpath(dirname(artifact_toml))

name = "SHTnsKit"

urls = String[]
sha256 = nothing

if Sys.isapple() && Sys.ARCH === :aarch64
    # Prefer correctly named macOS ARM tarball; fall back to older filename
    push!(urls, "https://subhk.github.io/release/download/SHTnsKit.v1.0.0.aarch64-apple-darwin.tar.gz")
    push!(urls, "https://subhk.github.io/release/download/SHTnsKit.v1.0.0.x86_64-linux-gnu.tar.gz")
    sha256 = "6d1e2155c4e96927de380e44a5a4b90c5d163e89737b79bd66b45daefa6acbd6"
elseif Sys.isapple() && Sys.ARCH === :x86_64
    # Placeholder for macOS Intel build; update after producing tarball via BinaryBuilder
    # Example name: SHTnsKit.v1.0.0.x86_64-apple-darwin.tar.gz
    push!(urls, "https://subhk.github.io/release/download/SHTnsKit.v1.0.0.x86_64-apple-darwin.tar.gz")
    sha256 = "<fill_sha256_here>"
elseif Sys.islinux() && Sys.ARCH === :x86_64
    # TODO: update once a real Linux build is uploaded
    @warn "No Linux x86_64 artifact URL configured yet; skipping bind."
else
    @warn "No artifact configured for this platform" arch=Sys.ARCH os=Sys.KERNEL
end

if !isnothing(sha256) && !isempty(urls)
    # Attempt to bind the artifact with the known URL(s) and sha256
    # This computes and writes git-tree-sha1 into Artifacts.toml
    try
        # Attempt each URL until one succeeds
        for u in urls
            try
                bind_artifact!(artifact_toml, name, ""; download_info=[(u, sha256)], lazy=true, force=true)
                @info "Bound artifact" name u
                break
            catch e
                @warn "Binding failed for URL; trying next" url=u err=e
            end
        end
    catch e
        @error "Failed to bind artifact" exception=e
    end
end
