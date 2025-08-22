## path prefix for make install (installs to $(PREFIX)/lib and $(PREFIX)/include)
PREFIX=/Users/subha/Downloads/shtns
SED=/usr/bin/sed
LIBS=-lfftw3_omp -lfftw3 -lm 
shtns_ver := $(shell grep -m 1 -e "^* v" CHANGELOG.md | sed -e "s/^* v\([^ ]*\).*/-DSHTNS_VER=\\\\\"\1\\\\\"/")
git_id := $(shell git describe --tags --always --dirty=*)

# compiler command with options for the sht codelets
shtcc=gcc -O2  -I/Users/subha/Downloads/shtns/include -L/Users/subha/Downloads/shtns/lib -march=native -O3 -ffast-math -std=gnu99 -fopenmp-simd -fno-stack-protector -fpic -D_GNU_SOURCE
# compiler command with options for other source (initialization, ...)
cc=gcc -O2  -I/Users/subha/Downloads/shtns/include -L/Users/subha/Downloads/shtns/lib -march=native -fno-math-errno -fno-trapping-math -ffinite-math-only -fcx-limited-range -std=gnu11 -fpic  -D_GNU_SOURCE -DSHTNS_GIT=\"$(git_id)\" $(shtns_ver) -L/Users/subha/Downloads/fftw-3.3.10/lib -L/opt/homebrew/lib
fc=gfortran -O2  -I/Users/subha/Downloads/shtns/include -L/Users/subha/Downloads/shtns/lib -L/Users/subha/Downloads/fftw-3.3.10/lib -L/opt/homebrew/lib

hfiles = sht_private.h sht_config.h shtns.h shtns_simd.h
objs = sht_kernels_a.o sht_kernels_s.o sht_odd_nlat.o sht_fly.o sht_omp.o sht_init.o
libname = libshtns_omp.so

default : libshtns_omp.so

install : install-lib

test : test_suite test_rot
	./test_suite
	./test_rot

libshtns_omp.so : Makefile $(objs)
	rm -f libshtns_omp.so
	gcc -O2  -I/Users/subha/Downloads/shtns/include -L/Users/subha/Downloads/shtns/lib -march=native -fno-math-errno -fno-trapping-math -ffinite-math-only -fcx-limited-range -std=gnu11 -fpic  -shared -L/Users/subha/Downloads/fftw-3.3.10/lib -L/opt/homebrew/lib -lfftw3_omp -lfftw3 -lm  -o libshtns_omp.so $(objs)
	@echo "**** link with : -L/Users/subha/Downloads/fftw-3.3.10/lib -L/opt/homebrew/lib -lfftw3_omp -lfftw3 -lm "

install-lib : $(libname)
	@mkdir -p $(PREFIX)/lib/
	@mkdir -p $(PREFIX)/include/
	cp $(libname) $(PREFIX)/lib/$(libname)
	cp shtns.h $(PREFIX)/include/
	cp shtns_cuda.h $(PREFIX)/include/
	cp shtns.f shtns.f03 shtns_cuda.f03 $(PREFIX)/include/
	@echo "**** link with : -L$(PREFIX)/lib -L/Users/subha/Downloads/fftw-3.3.10/lib -L/opt/homebrew/lib -lfftw3_omp -lfftw3 -lm "

# codelets :
SHT/SH_to_spat.c : SHT/hyb_SH_to_spat.gen.c
	$(MAKE) SH_to_spat.c -C SHT SED=$(SED)
SHT/spat_to_SH.c : SHT/hyb_spat_to_SH.gen.c
	$(MAKE) spat_to_SH.c -C SHT SED=$(SED)
SHT/SH_to_spat_fly.c : SHT/fly_SH_to_spat.gen.c
	$(MAKE) SH_to_spat_fly.c -C SHT SFX=fly SED=$(SED)
SHT/spat_to_SH_fly.c : SHT/fly_spat_to_SH.gen.c
	$(MAKE) spat_to_SH_fly.c -C SHT SFX=fly SED=$(SED)
SHT/SH_to_spat_omp.c : SHT/omp_SH_to_spat.gen.c
	$(MAKE) SH_to_spat_omp.c -C SHT SFX=omp SED=$(SED)
SHT/spat_to_SH_omp.c : SHT/omp_spat_to_SH.gen.c
	$(MAKE) spat_to_SH_omp.c -C SHT SFX=omp SED=$(SED)
SHT/SH_to_spat_kernel.c : SHT/kernel_SH_to_spat.gen.c
	$(MAKE) SH_to_spat_kernel.c -C SHT SFX=kernel SED=$(SED)
SHT/spat_to_SH_kernel.c : SHT/kernel_spat_to_SH.gen.c
	$(MAKE) spat_to_SH_kernel.c -C SHT SFX=kernel SED=$(SED)
SHT/cuda_legendre.inc : SHT/cuda_legendre.gen.cu
	$(MAKE) cuda_legendre.inc -C SHT SED=$(SED)

# objects :
sht_init.o : sht_init.c Makefile sht_legendre.c sht_func.c sht_com.c $(hfiles)
	$(cc) -c $< -o $@
sht_init_gpu.o : sht_init.c Makefile sht_legendre.c sht_func.c sht_com.c $(hfiles)
	$(cc)  -c $< -o $@

sht_odd_nlat.o : sht_odd_nlat.c Makefile sht_private.h
	$(cc) -c $< -o $@
sht_fly.o : sht_fly.c Makefile $(hfiles) SHT/SH_to_spat_fly.c SHT/spat_to_SH_fly.c
	$(shtcc) -c $< -o $@
sht_omp.o : sht_omp.c Makefile $(hfiles) SHT/SH_to_spat_omp.c SHT/spat_to_SH_omp.c
	$(cc) -c $< -o $@
sht_kernels_s.o : sht_kernels_s.c Makefile $(hfiles) SHT/SH_to_spat_kernel.c
	$(shtcc) -c $< -o $@
sht_kernels_a.o : sht_kernels_a.c Makefile $(hfiles) SHT/spat_to_SH_kernel.c
	$(shtcc) -c $< -o $@
sht_gpu.o : sht_gpu.cu sht_gpu_kernels.cu Makefile $(hfiles) SHT/cuda_legendre.inc
	 -c $< -o $@
# for more verbose cuda compilation, add to nvcc: -gencode=arch=compute_XX,code=sm_XX -Xptxas=-v

# programs & Examples
time_SHT : time_SHT.c $(libname) shtns.h
	$(cc) time_SHT.c -I. ./$(libname) $(LIBS) -o time_SHT

test_suite : examples/test_suite.c $(libname) shtns.h
	$(cc) examples/test_suite.c -I. ./$(libname) $(LIBS) -o test_suite
test_rot : examples/test_rot.c $(libname) shtns.h
	$(cc) examples/test_rot.c -I. ./$(libname) $(LIBS) -o test_rot

SHT_example : examples/SHT_example.c $(libname) shtns.h
	$(cc) examples/SHT_example.c -I. ./$(libname) $(LIBS) -o SHT_example

SHT_fort_ex : examples/SHT_example.f $(libname) shtns.f
	$(fc) -L$(PREFIX)/lib -I$(PREFIX)/include -I. -fdefault-real-8 examples/SHT_example.f ./$(libname) $(LIBS) -lc -o SHT_fort_ex
SHT_f90_ex : examples/SHT_example.f90 $(libname) shtns.f03
	$(fc) -L$(PREFIX)/lib -I$(PREFIX)/include -I. -fdefault-real-8 examples/SHT_example.f90 ./$(libname) $(LIBS) -lc -o SHT_f90_ex

wigner : examples/wigner.c $(libname) shtns.h
	$(cc) examples/wigner.c -I. ./$(libname) $(LIBS) -o wigner

#documentation :
docs :
	( cat doc/doxygen.conf ; grep -m 1 -e "^* v" CHANGELOG.md | sed -e "s/^* v\([^ ]*\).*/PROJECT_NUMBER=\1/" ) | doxygen -

clean :
	$(MAKE) clean -C SHT
	rm -f $(objs) $(libname) sht_init.o sht_init_gpu.o
	rm -rf doc/html
	rm -rf doc/latex
	rm -rf build


# generate python and c glue code with SWIG.
shtns_numpy_wrap.c : shtns_numpy.i sht_private.h shtns.h
	-swig -python -module shtns_cuda -o shtns_cuda_wrap.c -DSHTNS_GPU shtns_numpy.i
	-swig -python shtns_numpy.i
	-rm -f shtns_cuda.py

install-py :
	pip install shtns

.PHONY : install install-py install-lib clean docs


#fftw compiling options :
#-O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math -fno-schedule-insns -fno-web -fno-loop-optimize --param inline-unit-growth=1000 --param large-function-growth=1000
