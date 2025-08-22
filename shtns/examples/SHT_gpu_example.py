#
#  Copyright (c) 2010-2023 Centre National de la Recherche Scientifique.
#  written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
#
#  nathanael.schaeffer@univ-grenoble-alpes.fr
#
#  This software is governed by the CeCILL license under French law and
#  abiding by the rules of distribution of free software. You can use,
#  modify and/or redistribute the software under the terms of the CeCILL
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL license and that you accept its terms.
#

###################################
# SHTns Python interface example  #
###################################

import numpy        # numpy for arrays
import shtns        # shtns module compiled and installed using
                    #   pip install shtns
import time         # to measure elapsed time

lmax = 1023         # maximum degree of spherical harmonic representation. GPU acceleration works better on large lmax.

sh_cpu = shtns.sht(lmax)  # create sht object with given lmax and mmax (orthonormalized)
sh_gpu = shtns.sht(lmax)  # another one, which will run on GPU

# For now, the GPU only supports theta-contiguous grid. We need to tell shtns we want that, as well as GPU offload
sh_cpu.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)   # build grid to run on CPU (gauss grid, theta-contiguous)
sh_gpu.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)  # build grid (gauss grid, theta-contiguous, allowed to run on GPU)

# Note that even if specifying SHT_ALLOW_GPU, it may silently default to cpu for
# various reasons (shtns not compiled with cuda support, grid not supported on gpu, too small sizes, ...)
sh_cpu.print_info()
sh_gpu.print_info()  # if the printed output says many times 'gpu', then we can be sure it will run on gpu. 

print('cpu grid size:', sh_cpu.nlat, sh_cpu.nphi)     # displays the latitudinal and longitudinal grid sizes...
print('gpu grid size:', sh_gpu.nlat, sh_gpu.nphi)     # ... not guaranteed to be the same, unless we ask for specific sizes.

cost = sh_gpu.cos_theta         # latitudinal coordinates of the grid as cos(theta)
el = sh_gpu.l                   # array of size sh.nlm giving the spherical harmonic degree l for any sh coefficient
l2 = el*(el+1)              # array l(l+1) that is useful for computing laplacian

alm = sh_gpu.spec_array()       # a spherical harmonic spectral array, same as numpy.zeros(sh_gpu.nlm, dtype=complex)
alm[sh_gpu.idx(1, 0)] = 1.0     # set sh coefficient l=1, m=0 to value 1

t0 = time.perf_counter()
x_cpu = sh_cpu.synth(alm)           # transform sh description alm into spatial representation x (scalar transform)
t1 = time.perf_counter()
x_gpu = sh_gpu.synth(alm)
t2 = time.perf_counter()

print('transform on CPU took %.3g seconds' % (t1-t0))
print('transform on GPU took %.3g seconds (including copies between cpu and gpu -- this is %.3g times faster than CPU)' % (t2-t1, ((t1-t0)/(t2-t1))))
max_diff=numpy.max(abs(x_gpu-x_cpu))
print('maximum difference between gpu and cpu transformed data is', max_diff)

