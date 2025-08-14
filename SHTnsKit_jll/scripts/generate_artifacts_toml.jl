#!/usr/bin/env julia

"""
Generate complete Artifacts.toml entries (with git-tree-sha1) from local BinaryBuilder tarballs.

Usage:
  julia SHTnsKit_jll/scripts/generate_artifacts_toml.jl \
      --dir dist \
      --artifact SHTnsKit \
      --base-url https://github.com/<you>/<repo>/releases/download/v1.0.0

Notes:
  - Computes sha256 and git-tree-sha1 for each tarball.
  - Requires a `tar` executable available in PATH for extraction, or Julia's Tar stdlib with gzip support.
"""

using Printf
using SHA
using Pkg
import Pkg.GitTools

function parse_args!(args::Vector{String})
    dir = "dist"
    artifact = nothing
    base_url = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--dir"
            i += 1; dir = args[i]
        elseif arg == "--artifact"
            i += 1; artifact = args[i]
        elseif arg == "--base-url"
            i += 1; base_url = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    artifact === nothing && error("--artifact is required")
    base_url === nothing && error("--base-url is required")
    return (dir, artifact, base_url)
end

sha256hex(path::AbstractString) = open(path, "r") do io; bytes2hex(sha256(io)) end

function detect_platform_fields(filename::String)
    parts = split(filename, '.')
    if length(parts) < 3
        return nothing
    end
    trip = parts[end-2]  # e.g., x86_64-linux-gnu
    arch_os = split(trip, '-')
    if length(arch_os) < 2
        return nothing
    end
    arch = arch_os[1]
    os = arch_os[2]
    libc = nothing
    if os == "linux"
        libc = occursin("musl", trip) ? "musl" : "glibc"
    end
    return (arch=arch, os=os, libc=libc)
end

function try_tar_extract(tarball::String, dest::String)
    # Prefer Julia's Tar if available for gzip; otherwise shell out to tar
    try
        @eval begin
            import Tar
        end
        Tar.extract(tarball, dest)
        return
    catch err
        @warn "Tar.extract failed, falling back to system tar" err
    end
    run(`tar -xzf $(tarball) -C $(dest)`)  # Requires tar
end

function compute_tree_hash_from_tarball(tarpath::String)
    mktempdir() do tmp
        try_tar_extract(tarpath, tmp)
        # Compute git-tree-sha1 across extracted root
        h = GitTools.tree_hash(tmp)
        return string(h)
    end
end

function main()
    dir, artname, base_url = parse_args!(copy(ARGS))
    isdir(dir) || error("Directory not found: $dir")
    files = filter(f->endswith(f, ".tar.gz"), readdir(dir))
    isempty(files) && error("No .tar.gz files found in $dir")

    println("# Paste these blocks into SHTnsKit_jll/Artifacts.toml")
    for f in sort(files)
        tarpath = joinpath(dir, f)
        fields = detect_platform_fields(f)
        fields === nothing && (println("# Skipping unrecognized file name: $f"); continue)
        sha = sha256hex(tarpath)
        tree = compute_tree_hash_from_tarball(tarpath)
        println()
        println("[[", artname, "]]" )
        println("arch = \"", fields.arch, "\"")
        println("os = \"", fields.os, "\"")
        if fields.libc !== nothing
            println("libc = \"", fields.libc, "\"")
        end
        println("git-tree-sha1 = \"", tree, "\"")
        println("download = [")
        @printf("    { url = \"%s/%s\", sha256 = \"%s\" }\n", base_url, f, sha)
        println("]")
    end
end

isinteractive() || main()
