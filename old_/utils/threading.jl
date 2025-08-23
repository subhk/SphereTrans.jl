"""
Threading utilities and configuration for SHTnsKit.
These provide OpenMP-like controls in pure Julia:
- Global on/off for parallel loops (uses `Threads.@threads`)
- Control FFTW's internal threading for azimuthal FFTs
"""

const _threading_enabled = Ref{Bool}(true)

"""
    set_threading!(flag::Bool) -> Bool

Enable or disable SHTnsKit parallel loops (uses `Threads.@threads`).
Returns the new state.
"""
function set_threading!(flag::Bool)
    _threading_enabled[] = flag
    return _threading_enabled[]
end

"""
    get_threading() -> Bool

Return whether SHTnsKit parallel loops are enabled.
"""
get_threading() = _threading_enabled[]

"""
    set_fft_threads(n::Integer)

Set the number of threads used internally by FFTW plans.
This controls threading within azimuthal FFTs.
"""
function set_fft_threads(n::Integer)
    FFTW.set_num_threads(Int(n))
    return Int(n)
end

"""
    get_fft_threads() -> Int

Return the number of threads used by FFTW.
"""
get_fft_threads() = FFTW.get_num_threads()

"""
    set_optimal_threads!() -> NamedTuple

Pick a reasonable threading setup: enable parallel loops, and set FFTW threads
to `min(Threads.nthreads(), Sys.CPU_THREADS)`.
Returns a summary NamedTuple.
"""
function set_optimal_threads!()::NamedTuple{(:threads, :fft_threads), Tuple{Int, Int}}
    set_threading!(true)
    n = min(Threads.nthreads(), Sys.CPU_THREADS)
    set_fft_threads(n)
    return (threads=n, fft_threads=n)
end

