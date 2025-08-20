SHTnsKit BinaryBuilder

This folder contains a BinaryBuilder `build_tarballs.jl` to produce platform-specific binary artifacts for SHTns, and to ultimately create a `SHTnsKit_jll` suitable for registration.

Steps

1) Prerequisites
- Install Julia and BinaryBuilder: `julia -q -e 'using Pkg; Pkg.add("BinaryBuilder")'`

2) Pick the upstream source
- Edit `BinaryBuilder/build_tarballs.jl`:
  - Set `version` to match your SHTns version.
  - Replace the placeholder in `sources` with the actual source tarball URL and sha256.
    - You can compute sha256 via: `shasum -a 256 <tarball>`.

3) Choose platforms
- The script includes `x86_64-linux-gnu` and `aarch64-apple-darwin` by default.
- Uncomment or add more platforms as needed.

4) Build locally
- From this repository root (or the BinaryBuilder folder):
  - `julia BinaryBuilder/build_tarballs.jl --verbose` 
  - Artifacts (tarballs) will be produced under a `products` directory.

5) Create a JLL (optional, recommended)
- Use BinaryBuilder's deploy options (e.g., to GitHub) or construct a JLL repo manually referencing the produced tarballs in `Artifacts.toml`.
- For testing without registry, you can also use the included `SHTnsKit_jll` skeleton in this repo and bind the produced tarballs.

Notes
- OpenMP: macOS builds require `LLVMOpenMP_jll`, which the script declares for Apple platforms.
- If your SHTns version uses Autotools (configure/make), the script will detect and use it; otherwise it tries CMake.
- If the build produces both `libshtns` and `libshtns_omp`, both will be packaged when present.

