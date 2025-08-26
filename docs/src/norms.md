## Normalization and Condon–Shortley Phase

SHTnsKit internally uses orthonormal spherical harmonics with the Condon–Shortley
phase (CS) included. At the API level you can select:

- `norm = :orthonormal` (default): same as internal.
- `norm = :fourpi`: scales basis by `sqrt(4π)` relative to orthonormal.
- `norm = :schmidt`: Schmidt semi-normalized, scales by `sqrt(4π/(2l+1))`.
- `cs_phase = true/false`: include or exclude the phase `(-1)^m`.

Transforms convert your coefficients to/from the internal basis automatically.
You can also use `convert_alm_norm!(dest, src, cfg; to_internal)` to map matrices
explicitly between cfg’s requested normalization/phase and the internal one.

For packed real layouts (m ≥ 0) and complex packed layouts (both signs of m),
point/latitude evaluation functions also honor `norm` and `cs_phase` per (l,m).

