# Performance Tips

This page summarizes practical tips to reduce allocations and improve locality and throughput in SHTnsKit.jl, especially for distributed (MPI + PencilArrays) use.

- Reuse plans: Construct `SHTPlan` (serial) and distributed plans (`DistAnalysisPlan`, `DistSphtorPlan`, `DistQstPlan`) once per size and reuse. Plans hold FFT plans and working buffers to avoid per-call allocations.

- use_rfft (distributed plans): When available in your `PencilFFTs`, set `use_rfft=true` in distributed plans to cut the (θ,k) spectral memory and accelerate real-output paths. The code falls back to complex FFTs when real transforms are not supported.

- with_spatial_scratch (vector/QST): Enable `with_spatial_scratch=true` to keep a single complex (θ,φ) scratch in the plan. This removes per-call iFFT allocations for real outputs. Default remains off to minimize footprint.

- Precomputed Legendre tables: On fixed grids, call `enable_plm_tables!(cfg)` to precompute `plm_tables` and `dplm_tables`. They provide identical results to on-the-fly recurrences and usually reduce CPU cost.

- Threading inside rank: For large lmax, enable Julia threads and (optionally) FFTW threads. Use `set_optimal_threads!()` or tune with `set_threading!()` and `set_fft_threads()` to match your core layout.

- LoopVectorization: If available, `analysis_turbo`/`synthesis_turbo` and related helpers can accelerate inner loops. Guard with `using LoopVectorization`.

- Data locality by m: Keep Alm distributed by m throughout your pipeline to avoid dense gathers. The distributed plans in this package consume and produce m-sliced data to preserve cache locality.

