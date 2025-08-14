Publishing SHTnsKit_jll as a proper JLL

Overview
- Build platform-specific tarballs with BinaryBuilder (see build_tarballs.jl).
- Host the tarballs (e.g., GitHub Releases).
- Update Artifacts.toml with git-tree-sha1 and download entries per platform.
- Remove the path-based artifact entry when done.

1) Build tarballs with BinaryBuilder
- Locally or in a container:
  julia --project -e 'using Pkg; Pkg.add("BinaryBuilder"); using BinaryBuilder; include("SHTnsKit_jll/build_tarballs.jl");'
  # Or run via: julia SHTnsKit_jll/build_tarballs.jl

- Output: *.tar.gz files for each platform listed in build_tarballs.jl.

2) Host the tarballs
- Upload the *.tar.gz files to a stable URL (e.g., a GitHub Release in your repo).
- For each uploaded file, compute sha256:
  shasum -a 256 <yourfile.tar.gz>

3) Create Artifacts.toml entries
- Generate entries automatically with tree hashes:
  julia SHTnsKit_jll/scripts/generate_artifacts_toml.jl --dir <tarballs_dir> --artifact SHTnsKit --base-url <release_url>

- Replace the current path-based entry with the printed per-platform entries. Example:

  [[SHTnsKit]]
  arch = "x86_64"
  os = "linux"
  libc = "glibc"
  git-tree-sha1 = "<TREE_HASH_FROM_BINARYBUILDER>"
  download = [
      { url = "https://github.com/<you>/<repo>/releases/download/v1.0.0/SHTnsKit.v1.0.0.x86_64-linux-gnu.tar.gz", sha256 = "<SHA256>" }
  ]

  [[SHTnsKit]]
  arch = "x86_64"
  os = "macos"
  git-tree-sha1 = "<TREE_HASH>"
  download = [
      { url = "https://github.com/<you>/<repo>/releases/download/v1.0.0/SHTnsKit.v1.0.0.x86_64-apple-darwin.tar.gz", sha256 = "<SHA256>" }
  ]

  [[SHTnsKit]]
  arch = "x86_64"
  os = "windows"
  git-tree-sha1 = "<TREE_HASH>"
  download = [
      { url = "https://github.com/<you>/<repo>/releases/download/v1.0.0/SHTnsKit.v1.0.0.x86_64-w64-mingw32.tar.gz", sha256 = "<SHA256>" }
  ]

- Use one [[SHTnsKit]] table per platform/variant. You can add aarch64 for Linux/macOS as well.

Notes
- The wrapper in src/SHTnsKit_jll.jl loads both libshtns and libshtns_omp with the correct extension via Libdl.dlext. Ensure your build installs one or both names; simplest is to install libshtns, and optionally also libshtns_omp as a copy.
- Once Artifacts.toml is populated with download entries, remove the current path = "local_artifacts/current" stanza.
- Yggdrasil option: You can upstream SHTnsKit to Yggdrasil. If accepted, the registry bots will manage artifact hosting and auto-generate SHTnsKit_jll for you. Start by adapting build_tarballs.jl to the Yggdrasil style and opening a PR.
