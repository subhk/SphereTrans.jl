/* sht_config.h.  Generated from sht_config.h.in by configure.  */
/* sht_config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to 1 if you have the 'clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if you have the <complex.h> header file. */
#define HAVE_COMPLEX_H 1

/* Define to 1 if you have the <c_asm.h> header file. */
/* #undef HAVE_C_ASM_H */

/* Define to 1 if you have the 'fftw_cost' function. */
#define HAVE_FFTW_COST 1

/* Define to 1 if you have the 'gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if hrtime_t is defined in <sys/time.h> */
/* #undef HAVE_HRTIME_T */

/* Define to 1 if you have the <intrinsics.h> header file. */
/* #undef HAVE_INTRINSICS_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the 'amdhip64' library (-lamdhip64). */
/* #undef HAVE_LIBAMDHIP64 */

/* Define to 1 if you have the 'cuda' library (-lcuda). */
/* #undef HAVE_LIBCUDA */

/* Define to 1 if you have the 'cudart' library (-lcudart). */
/* #undef HAVE_LIBCUDART */

/* Define to 1 if you have the 'fftw3' library (-lfftw3). */
#define HAVE_LIBFFTW3 1

/* Define to 1 if you have the 'fftw3_omp' library (-lfftw3_omp). */
#define HAVE_LIBFFTW3_OMP 1

/* Define to 1 if you have the 'm' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the 'nvrtc' library (-lnvrtc). */
/* #undef HAVE_LIBNVRTC */

/* Define to 1 if you have the 'mach_absolute_time' function. */
#define HAVE_MACH_ABSOLUTE_TIME 1

/* Define to 1 if you have the <mach/mach_time.h> header file. */
#define HAVE_MACH_MACH_TIME_H 1

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Define to 1 if you have the 'read_real_time' function. */
/* #undef HAVE_READ_REAL_TIME */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the 'time_base_to_time' function. */
/* #undef HAVE_TIME_BASE_TO_TIME */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define if you have the UNICOS _rtc() intrinsic. */
/* #undef HAVE__RTC */

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "SHTns"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "SHTns 3.7.3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "shtns"

/* Define to the home page for this package. */
#define PACKAGE_URL "https://bitbucket.org/nschaeff/shtns"

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.7.3"

/* I need the transforms compatible with the MagIC code, to speed it up! */
/* #undef SHTNS4MAGIC */

/* Enable the new recurrence proposed by Ishioka (2018) see
   https://doi.org/10.2151/jmsj.2018-019 (faster) */
#define SHTNS_ISHIOKA 1

/* Compile the Fortran API */
#define SHT_F77_API 1

/* 0:no output, 1:output info to stdout, 2:more output (debug info), 3:also
   print fftw plans. */
#define SHT_VERBOSE 1

/* Define to 1 if all of the C89 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Use vkFFT for the FFT on the GPU when possible */
/* #undef VKFFT_BACKEND */

/* I compile with GCC 4 or ICC 14 or later, and I would like fast vectorized
   code (if SSE2, AVX or MIC is supported) ! */
#define _GCC_VEC_ 1

/* Define to '__inline__' or '__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define as 'unsigned int' if <stddef.h> doesn't define. */
/* #undef size_t */
