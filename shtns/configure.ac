#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([SHTns],[3.7.3],[],[shtns],[https://bitbucket.org/nschaeff/shtns])
AC_LANG([C])
AC_CONFIG_SRCDIR([sht_init.c])
AC_CONFIG_HEADERS([sht_config.h])
target="libshtns"		# by default, build the library.
objs="sht_fly.o"
install="install-lib"	# by default, install the library.

# optional features with --enable-XXX
AC_ARG_ENABLE([verbose],
	AS_HELP_STRING([--enable-verbose=0-3], [define verbosity level for the library: 0=silent, 1=default, 2=debug, 3=full]),
	[],[enable_verbose=1])
AC_ARG_ENABLE([openmp],
	AS_HELP_STRING([--enable-openmp], [Enable multi-threading with OpenMP]), [], [enable_openmp=default])
AC_ARG_ENABLE([mkl],
	AS_HELP_STRING([--enable-mkl], [Try to link with intel MKL instead of FFTW (can be slightly faster, but NOT THREAD SAFE)]), [], [enable_mkl=default])
AC_ARG_ENABLE([knl],
	AS_HELP_STRING([--enable-knl], [Enable compilation for Xeon Phi (KNL)]), [], [enable_knl=no])
AC_ARG_ENABLE([cuda],
	AS_HELP_STRING([--enable-cuda@<:@=kepler|pascal|volta|ampere@:>@], [Enable compilation for Nvidia gpu using cuda, optionally specifying architecture]), [], [enable_cuda=no])
AC_ARG_ENABLE([hip],
	AS_HELP_STRING([--enable-hip@<:@=mi100|mi200|mi250@:>@], [Enable compilation for AMD gpu using HIP, optionally specifying architecture]), [], [enable_hip=no])
AC_ARG_ENABLE([magic-layout],
	AS_HELP_STRING([--enable-magic-layout], [Compile specific version for the MagIC code]), [], [enable_magic_layout=no])
AC_ARG_ENABLE([python],
	AS_HELP_STRING([--enable-python], [Build Python interface, implies --enable-shared]), [], [enable_python=no])
AC_ARG_ENABLE([shared],
	AS_HELP_STRING([--enable-shared], [allows building of a shared library (instead of a static one)]), [], [enable_shared=no])
AC_ARG_ENABLE([f77],
	AS_HELP_STRING([--disable-f77], [Do not include F77 wrapper to call SHTns library from Fortran]))
AC_ARG_ENABLE([simd],
	AS_HELP_STRING([--disable-simd], [Do not use vector extensions (SSE2, AVX, AVX2, AVX-512, or VSX)]))
AC_ARG_ENABLE([ishioka],
	AS_HELP_STRING([--disable-ishioka], [Do not use the new and faster recursion]), [], [enable_ishioka=yes])
AC_ARG_ENABLE([kernel-compiler],
	AS_HELP_STRING([--enable-kernel-compiler=gcc], [Use a different compiler for CPU kernels (default=gcc)]), [], [enable_kernel_compiler=gcc])
AC_ARG_ENABLE([march],
	AS_HELP_STRING([--enable-march=native], [select target cpu architecture with a -march=XXX compiler option (for gcc, icc and clang) (default: XXX=native for best performance)]), [], [enable_march=native])


dnl Sanitize $prefix. Autoconf does this by itself, but so late in the
dnl generated configure script that the expansion does not occur until
dnl after our eval magic below.
AS_IF([test "$prefix" = "NONE"],[prefix=$ac_default_prefix])

# Checks for programs.
CFLAGS="-O2 $CFLAGS -I$prefix/include -L$prefix/lib"   # we need at least -O2 optimization level here, but it can be everriden by CFLAGS (gcc specifies that the last -O wins)
FFLAGS="$CFLAGS"
AC_PROG_CC
AC_PROG_SED
if test "$SED" = :; then
	AC_MSG_ERROR([sed program required.])
fi

# on MacOSX include the homebrew library search path:
AS_IF([uname | grep Darwin >/dev/null],[
    AS_IF([uname -m | grep arm64 >/dev/null], [LDFLAGS="$LDFLAGS -L/opt/homebrew/lib"], [LDFLAGS="$LDFLAGS -L/usr/local/lib"])
])

# define macro AX_CHECK_COMPILE_FLAG, and add it to CFLAGS
AC_DEFUN([AX_CHECK_COMPILE_FLAG],[
  AC_MSG_CHECKING([whether $CC accepts $1])
  ax_check_save_flags=$[]_AC_LANG_PREFIX[]FLAGS
  _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $1"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [_AC_LANG_PREFIX[]FLAGS=$ax_check_save_flags
     AC_MSG_RESULT([yes])
     m4_default([$2], [CFLAGS="$CFLAGS $1"])],
    [_AC_LANG_PREFIX[]FLAGS=$ax_check_save_flags
     AC_MSG_RESULT([no])
     m4_default([$3], :)])
])dnl AX_CHECK_COMPILE_FLAGS

# define macro AX_CHECK_CUDA_ARCH
AC_DEFUN([AX_CHECK_CUDA_ARCH],[
  AC_MSG_CHECKING([whether nvcc supports gpu with compute capability $1])
  AS_IF([nvcc sht_gpu.cu -c -arch=sm_$1 --dryrun > /dev/null 2>&1], [
	AC_MSG_RESULT(yes)
	nvcc_gencode_flags="$nvcc_gencode_flags -gencode=arch=compute_$1,code=compute_$1"
  ],[
	AC_MSG_RESULT(no)
  ])
])dnl AX_CHECK_CUDA_ARCH

############################################
### we can use two compilers. CC and CC2 ###
############################################
CC2=$enable_kernel_compiler
# backup current stuff
backup_cflags=$CFLAGS
backup_cc=$CC
CC=$CC2
#### setup the kernel compiler
AS_IF([test "x$enable_knl" != "xyes"], [
	# add gcc compile options if supported.
	AS_IF([test "x$enable_march" != "xno"], [AX_CHECK_COMPILE_FLAG([-march=$enable_march])])
	# tune for skylake, it gives significantly better performance on Haswell with gcc 9.2
	AX_CHECK_COMPILE_FLAG([-mtune=skylake])
	# this is an icc compile option, try if it is supported:
	AX_CHECK_COMPILE_FLAG([-qopt-zmm-usage=high])
  ],[
	# KNL native cross-compiling
	AX_CHECK_COMPILE_FLAG([-march=knl],[], [
	    CC=icc
	    AX_CHECK_COMPILE_FLAG([-xMIC-AVX512],[CC="$CC -xMIC-AVX512"], [AC_MSG_ERROR(["Xeon Phi (KNL) not supported by $CC. Use a recent gcc or icc compiler."],2)])
	])
])
CFLAGS="$CFLAGS -O3"
AX_CHECK_COMPILE_FLAG([-ffast-math])
AX_CHECK_COMPILE_FLAG([-std=gnu99],, [
    AX_CHECK_COMPILE_FLAG([-std=c99])
])
AX_CHECK_COMPILE_FLAG([-fopenmp-simd])
AX_CHECK_COMPILE_FLAG([-qopenmp-simd])
# the following prevents gcc from inserting memset in tight loops for zeroing stuff. Must appear after -O3
AX_CHECK_COMPILE_FLAG([-fno-tree-loop-distribute-patterns])
# avoid insering VZEROUPPER instructions everywhere.
AX_CHECK_COMPILE_FLAG([-mno-vzeroupper])
# avoid inserting stack protections in compute kernels:
AX_CHECK_COMPILE_FLAG([-fno-stack-protector])

## check for nasty bug in binutils, apply possible workaround. See https://bitbucket.org/nschaeff/shtns/issues/37/
bug_as=`echo 'asm("vmovaps 64(,%rax), %zmm0");' > x.c ; $CC -c x.c ; objdump -dw x.o | grep -c '1('`
rm -f x.c x.o
AS_IF([test "x$bug_as" == "x1"], [
    AX_CHECK_COMPILE_FLAG([-fno-tree-pre])
])

CC2=$CC
CFLAGS2=$CFLAGS
# restore the standard compiler
CC=$backup_cc
CFLAGS=$backup_cflags
################################
### kernel compiler is setup ###
################################

AS_IF([test "x$enable_knl" != "xyes"], [
	# add gcc compile options if supported.
	AS_IF([test "x$enable_march" != "xno"], [AX_CHECK_COMPILE_FLAG([-march=$enable_march])])
	# tune for skylake, it gives significantly better performance on Haswell with gcc 9.2
	AX_CHECK_COMPILE_FLAG([-mtune=skylake])
	# this is an icc compile option, try if it is supported:
	AX_CHECK_COMPILE_FLAG([-qopt-zmm-usage=high])
  ],[
	target="libshtns_mic"
	# KNL native cross-compiling
	AX_CHECK_COMPILE_FLAG([-march=knl],[], [
	    CC=icc
	    AX_CHECK_COMPILE_FLAG([-xMIC-AVX512],[CC="$CC -xMIC-AVX512"], [AC_MSG_ERROR(["Xeon Phi (KNL) not supported by $CC. Use a recent gcc or icc compiler."],2)])
	])
	# With KNL, enable mkl and openmp by default.
	AS_IF([test "x$enable_mkl" = "xdefault"], [enable_mkl=yes])
	AS_IF([test "x$enable_openmp" = "xdefault"], [enable_openmp=yes])
])
# add g++ compile options if supported. Do NOT set -ffast-math !!
# these may be acceptable: -fno-math-errno -freciprocal-math -fno-trapping-math -ffinite-math-only -fcx-limited-range
AX_CHECK_COMPILE_FLAG([-fno-math-errno])
AX_CHECK_COMPILE_FLAG([-fno-trapping-math])
AX_CHECK_COMPILE_FLAG([-ffinite-math-only])
AX_CHECK_COMPILE_FLAG([-fcx-limited-range])

# add icc compile options if supported.
AX_CHECK_COMPILE_FLAG([-fp-model=precise])
AX_CHECK_COMPILE_FLAG([-complex-limited-range])
AX_CHECK_COMPILE_FLAG([-prec-div -prec-sqrt])

AX_CHECK_COMPILE_FLAG([-std=gnu11],, [
    AX_CHECK_COMPILE_FLAG([-std=gnu99],, [
	AX_CHECK_COMPILE_FLAG([-std=c99])
    ])
])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h stdio.h string.h math.h complex.h])

# Checks for libraries.
AC_CHECK_LIB([m],[cos],,[AC_MSG_ERROR([math library not found.],2)])

# Checks related to Python:
AS_IF([test "x$enable_python" != "xno"], [
	AS_IF([test "x$enable_openmp" = "xdefault"], [enable_openmp=yes])   # enable openmp by default
	enable_shared=yes   # shared library required
	install="install-py"	# install python extension instead of C library.
])

AS_IF([test "x$enable_shared" != "xno"], [
	CFLAGS="$CFLAGS -fpic"	# required compile flag for python extensions.
	CFLAGS2="$CFLAGS2 -fpic"	# required compile flag for python extensions.
])

# for MagIC code, also disable matrix transforms.
AS_IF([test "x$enable_magic_layout" != "xno"], [
	AC_DEFINE([SHTNS4MAGIC],[1],[I need the transforms compatible with the MagIC code, to speed it up!])
])

AS_IF([test "x$enable_openmp" = "xyes"], [
	AC_OPENMP
	CFLAGS="$CFLAGS $OPENMP_CFLAGS"
	dnl AC_CHECK_HEADERS([omp.h])
	objs="$objs sht_omp.o"
	AS_IF([test "x$enable_knl" != "xno"], [
		target="libshtns_mic_omp"
	],[
		target="libshtns_omp"
	])
])

# Check for HIP
AS_IF([test "x$enable_hip" != "xno"], [
	CFLAGS="$CFLAGS -I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__"
	LDFLAGS="$LDFLAGS -L${ROCM_PATH}/lib"
	echo $LDFLAGS
	AC_CHECK_LIB([amdhip64],[hipMalloc],,[AC_MSG_ERROR([hip library not found. Environment variable ROCM_PATH is not correctly set: \$ROCM_PATH/lib should contain the rocm libraries.],5)])
	dnl AC_CHECK_LIB([hipfft],[hipfftCreate],,[AC_MSG_NOTICE([hipftt library not found.])])
	dnl AC_CHECK_LIB([rocfft],[rocfft_execute],,[AC_MSG_NOTICE([rocfft library not found.])])

	gpu_cflags="-DSHTNS_GPU=2"  # GPU transforms, 1 for CUDA, 2 for HIP
	gpu_compile="hipcc -std=c++11 -O3 --rocm-path=$ROCM_PATH -D__HIP_PLATFORM_AMD__ $gpu_cflags"
	#gpu_compile="amdclang -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -std=c++11 -O3 --rocm-path=$ROCM_PATH -D__HIP_PLATFORM_AMD__ -x hip"
	AS_IF([test "x$enable_hip" == "xyes"],[
	    gpu_compile="$gpu_compile --offload-arch=gfx908:xnack+ --offload-arch=gfx90a:xnack+ --offload-arch=gfx908:xnack- --offload-arch=gfx90a:xnack-"
	],[
	    AS_CASE([$enable_hip], "mi100", [hip_arch=gfx908], "mi200", [hip_arch=gfx90a], "mi250", [hip_arch=gfx90a], [hip_arch=$enable_hip])
	    gpu_compile="$gpu_compile --offload-arch=$hip_arch:xnack+ --offload-arch=$hip_arch:xnack-"
	])
	target="libshtns_cuda"
	AS_IF([test "x$enable_openmp" = "xyes"], [
	    gpu_compile="$gpu_compile -fopenmp"
	    target="libshtns_cuda_omp"
	])
	AC_CHECK_FILE([vkfft/vkFFT.h], AC_DEFINE([VKFFT_BACKEND],[2],[Use vkFFT for the FFT on the GPU when possible]))
])
# Check for CUDA
AS_IF([test "x$enable_cuda" != "xno"], [
	CFLAGS="$CFLAGS -I$CUDA_PATH/include"
	# assume 64 bit mode, add default directory to find CUDA
	LDFLAGS="$LDFLAGS -L$CUDA_PATH/lib64 -L$CUDA_PATH/lib64/stubs"
	echo $LDFLAGS
	AC_CHECK_LIB([cudart],[cudaMalloc],,[AC_MSG_ERROR([cudart library not found. Environment variable CUDA_PATH is not correctly set: \$CUDA_PATH/lib64 should contain the cuda libraries.],4)])
	dnl AC_CHECK_LIB([cufft],[cufftPlanMany],,[AC_MSG_NOTICE([cufft library not found.])])
	AC_CHECK_LIB([nvrtc],[nvrtcGetPTX],,[AC_MSG_ERROR([nvrtc library not found.],4)])
	AC_CHECK_LIB([cuda],[cuLaunchKernel],,[AC_MSG_ERROR([cuda library not found.],4)])

	nvcc_gencode_flags=""
	# find cuda arch:
	AS_IF([test "x$enable_cuda" == "xyes"],[
	    AX_CHECK_CUDA_ARCH([30])
	    AX_CHECK_CUDA_ARCH([60])
	    AX_CHECK_CUDA_ARCH([70])
	    AX_CHECK_CUDA_ARCH([80])
	],[
	    AS_CASE([$enable_cuda], "kepler", [cuda_cc=30], "pascal", [cuda_cc=60], "volta", [cuda_cc=70], 'ampere', [cuda_cc=80], 'hopper', [cuda_cc=90], [cuda_cc=$enable_cuda])
	    AX_CHECK_CUDA_ARCH([$cuda_cc])
	])
	AS_IF([test "x$nvcc_gencode_flags" == "x"],[
	    AC_MSG_ERROR([nvcc does not support any compatible gpu. Is nvcc available ?],4)
	])
	gpu_cflags="-DSHTNS_GPU=1"  # GPU transforms, 1 for CUDA, 2 for HIP
	# for nvcc, the underlying gcc is not always c++11 by default, so we need to explicitely ask for it.
	gpu_compile="nvcc -std=c++11 $nvcc_gencode_flags -lineinfo -Xptxas=-v $gpu_cflags"
	target="libshtns_cuda"
	AS_IF([test "x$enable_openmp" = "xyes"], [
	    gpu_compile="$gpu_compile -Xcompiler -fopenmp"
	    target="libshtns_cuda_omp"
	])
	AS_IF([test "x$enable_shared" != "xno"], [
	    gpu_compile="$gpu_compile -Xcompiler -fpic"
	])
	AC_CHECK_FILE([vkfft/vkFFT.h], AC_DEFINE([VKFFT_BACKEND],[1],[Use vkFFT for the FFT on the GPU when possible]))
])
AS_IF([test "x$gpu_compile" != "x" ], [
    objs="$objs sht_init_gpu.o sht_gpu.o"
    LIBS="$LIBS -lstdc++"	# cuda/hip uses the c++ standard library. This is needed to link with gcc
],[
    objs="$objs sht_init.o"
])

# Check for FFTW (MKL preferred, FFTW optional)
AS_IF([test "x$enable_mkl" == "xyes"], [
  mkl_found=no
  AX_CHECK_COMPILE_FLAG([-mkl], [
	LIBS_bak="$LIBS"
	LIBS="-mkl $LIBS"	# easy way, for intel compiler !
	AC_CHECK_FUNC(fftw_plan_many_dft, [
	    AC_MSG_NOTICE(["FFTW interface found in MKL, link with -mkl"])
	    mkl_found=yes
	],[
	    LIBS="$LIBS_bak"
	    AC_MSG_NOTICE(["FFTW interface not found in MKL libraries (-mkl)."])
	])
  ])
  AS_IF([test "x$mkl_found" == "xno"], [
	# assume 64 bit mode, add default directory to find the MKL
	LDFLAGS="$LDFLAGS -L$MKLROOT/lib/intel64"
	AS_IF([test "x$enable_openmp" = "xyes"], [
		# multi-threaded MKL libs
		#mkl_libs="-Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread"
		mkl_libs="-Wl,--start-group -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group"
	],[
		# sequential MKL libs
		mkl_libs="-Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group"
	])
	AC_CHECK_LIB(mkl_intel_lp64, fftw_plan_many_dft, [
	    LIBS="$mkl_libs $LIBS"
	    AC_MSG_NOTICE(["FFTW interface found in MKL, link with $mkl_libs"])
	    mkl_found=yes
	],[
	    AC_MSG_NOTICE(["FFTW interface not found in MKL libraries ($mkl_libs). Try to add 'LDFLAGS=-L/path/to/mkl' to ./configure command line."])
	],["$mkl_libs"])
  ])
  AS_IF([test "x$mkl_found" == "xno"], [
    AS_IF([test "x$enable_mkl" == "xyes"], [AC_MSG_ERROR(["FFTW interface not found in MKL."],3)], [enable_mkl=no])
  ],[
    AC_DEFINE(HAVE_LIBFFTW3_OMP, 1, [MKL has the multi-thread fftw interface.])
  ])
])

# Check for FFTW, if MKL was not required and not found.
AS_IF([test "x$enable_mkl" != "xyes"], [
	# fftw3, allows to specify library path with FFTW_ROOT environment variable
	AS_IF([test "x$FFTW_ROOT" != "x"], [LDFLAGS="$LDFLAGS -L$FFTW_ROOT/lib"])
	AC_CHECK_LIB([fftw3],[fftw_plan_many_dft],,[AC_MSG_ERROR(["FFTW3 library required. Try using the intel MKL library with --enable-mkl, or add 'LDFLAGS=-L/path/to/fftw3/lib' to ./configure command line."],3)])
	AS_IF([test "x$enable_openmp" = "xyes"], [
		AC_CHECK_LIB([fftw3_omp], [fftw_init_threads],,[AC_MSG_ERROR(["FFTW3 does not support OpenMP. Did you compile it with --enable-openmp ?."],3)])
	])
])

# Disable Fortran interface ?
AS_IF([test "x$enable_f77" != "xno"], [
	AC_DEFINE([SHT_F77_API],[1],[Compile the Fortran API])
])

# Disable SIMD ?
AS_IF([test "x$enable_simd" != "xno"], [
	AC_DEFINE([_GCC_VEC_],[1],[I compile with GCC 4 or ICC 14 or later, and I would like fast vectorized code (if SSE2, AVX or MIC is supported) !])
])


AS_IF([test "x$enable_ishioka" == "xyes"], [
       AC_DEFINE([SHTNS_ISHIOKA],[1],[Enable the new recurrence proposed by Ishioka (2018) see https://doi.org/10.2151/jmsj.2018-019 (faster)])
])

# Verbosity setting
AS_IF([test "x$enable_verbose" == "xno"], [enable_verbose=0],
	[test "x$enable_verbose" == "xyes"], [enable_verbose=1])
AC_DEFINE_UNQUOTED([SHT_VERBOSE],$enable_verbose,[0:no output, 1:output info to stdout, 2:more output (debug info), 3:also print fftw plans.])
echo "  verbose level = $enable_verbose"

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
#AC_FUNC_MALLOC
#AC_FUNC_REALLOC
AC_CHECK_FUNCS([fftw_cost])

### for cycle.h ###
   AC_C_INLINE
   AC_CHECK_HEADERS([sys/time.h c_asm.h intrinsics.h mach/mach_time.h])

   AC_CHECK_TYPE([hrtime_t],[AC_DEFINE(HAVE_HRTIME_T, 1, [Define to 1 if hrtime_t is defined in <sys/time.h>])],,[#if HAVE_SYS_TIME_H
#include <sys/time.h>
#endif])

   AC_CHECK_FUNCS([gethrtime read_real_time time_base_to_time clock_gettime mach_absolute_time])

   dnl Cray UNICOS _rtc() (real-time clock) intrinsic
   AC_MSG_CHECKING([for _rtc intrinsic])
   rtc_ok=yes
   AC_LINK_IFELSE([AC_LANG_PROGRAM([[#ifdef HAVE_INTRINSICS_H
#include <intrinsics.h>
#endif]], [[_rtc()]])],[AC_DEFINE(HAVE__RTC,1,[Define if you have the UNICOS _rtc() intrinsic.])],[rtc_ok=no])
   AC_MSG_RESULT($rtc_ok)
### end cycle.h ###

#### for the Fortran example ####
AC_PROG_FC
#### end Fortran #####

## add correct extension to target:
AS_IF([test "x$enable_shared" != "xno"], [
	target="${target}.so"
	target_cmd="$CC $CFLAGS -shared $LDFLAGS $LIBS -o"
],[
	target="${target}.a"
	target_cmd="ar rcs"
])

echo "prefix=$prefix"
AC_SUBST([target])
AC_SUBST([target_cmd])
AC_SUBST([install])
AC_SUBST([objs])
AC_SUBST([CC2])
AC_SUBST([CFLAGS2])
AC_SUBST([FFLAGS])
AC_SUBST([gpu_compile])
AC_SUBST([gpu_cflags])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

## if binutils bug: warn the user.
AS_IF([test "x$bug_as" == "x1"], [
    AC_MSG_NOTICE(["WARNING! your version of binutils is buggy. Please update the binutils package to v2.32 or more recent.
	Meanwhile, I try to apply a workaround (no guarantee, possible performance loss).
	See https://bitbucket.org/nschaeff/shtns/issues/37/"])
])

