/*
 * Copyright (c) 2010-2021 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 * 
 * nathanael.schaeffer@univ-grenoble-alpes.fr
 * 
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 * 
 */

/** \file shtns_cuda.h
 \brief shtns_cuda.h declares transforms and initialization functions for cuda-enabled GPU.
**/

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
typedef hipStream_t shtns_gpu_stream_t;
#else
#include <cuda_runtime.h>
typedef cudaStream_t shtns_gpu_stream_t;
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**\addtogroup cuda GPU transforms.
 * CUDA transforms working on GPU memory, without transfers, but **NOT thread-safe**.
 * \warning These transforms are **NOT thread-safe**. Use one distinct shtns_cfg per thread. Use \ref cushtns_clone to clone them.
 * The transforms are **Non-blocking**, working with their own streams. Each clone has distinct streams. \see \ref gpu
**/
///@{

///\name Transforms
///@{

/// Same as \ref spat_to_SH, but working on data residing on the GPU.
void cu_spat_to_SH(shtns_cfg shtns, double *Vr, cplx *Qlm, int ltr);
/// Same as \ref cu_spat_to_SH, but working on single precision data.
void cu_spat_to_SH_float(shtns_cfg shtns, float *Vr, cplx_f *Qlm, int ltr);
/// Same as \ref SH_to_spat, but working on data residing on the GPU.
void cu_SH_to_spat(shtns_cfg shtns, cplx *Qlm, double *Vr, int ltr);
/// Same as \ref cu_SH_to_spat, but working on single precision data.
void cu_SH_to_spat_float(shtns_cfg shtns, cplx_f *Qlm, float *Vr, int ltr);
/// Same as \ref spat_to_SHsphtor, but working on data residing on the GPU.
void cu_spat_to_SHsphtor(shtns_cfg, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, int ltr);
void cu_spat_to_SHsphtor_float(shtns_cfg, float *Vt, float *Vp, cplx_f *Slm, cplx_f *Tlm, int ltr);
/// Same as \ref SHsphtor_to_spat, but working on data residing on the GPU.
void cu_SHsphtor_to_spat(shtns_cfg, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, int ltr);
void cu_SHsphtor_to_spat_float(shtns_cfg, cplx_f *Slm, cplx_f *Tlm, float *Vt, float *Vp, int ltr);
/// Same as \ref SHsph_to_spat, but working on data residing on the GPU.
void cu_SHsph_to_spat(shtns_cfg, cplx *Slm, double *Vt, double *Vp, int ltr);
void cu_SHsph_to_spat_float(shtns_cfg, cplx_f *Slm, float *Vt, float *Vp, int ltr);
/// Same as \ref SHtor_to_spat, but working on data residing on the GPU.
void cu_SHtor_to_spat(shtns_cfg, cplx *Tlm, double *Vt, double *Vp, int ltr);
void cu_SHtor_to_spat_float(shtns_cfg, cplx_f *Tlm, float *Vt, float *Vp, int ltr);
/// Same as \ref spat_to_SHqst, but working on data residing on the GPU.
void cu_spat_to_SHqst(shtns_cfg, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, int ltr);
void cu_spat_to_SHqst_float(shtns_cfg, float *Vr, float *Vt, float *Vp, cplx_f *Qlm, cplx_f *Slm, cplx_f *Tlm, int ltr);
/// Same as \ref SHqst_to_spat, but working on data residing on the GPU.
void cu_SHqst_to_spat(shtns_cfg, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, int ltr);
void cu_SHqst_to_spat_float(shtns_cfg, cplx_f *Qlm, cplx_f *Slm, cplx_f *Tlm, float *Vr, float *Vt, float *Vp, int ltr);
///@}

///\name Initialization
///@{
 
/// Initialize given config to work on the current (or default) GPU, allowing to call GPU transforms cu_* above, working on data residing in the memory of this GPU.
/// This does not enable auto-offload. Use cudaSetDevice() or hipSetDevice() to set the target GPU before calling this function.
/// Note that it is the user's responsibility to ensure the current device will be the same for subsequent calls to transform functions with this configuration.
/// \param[in] shtns is a valid shtns configuration created with \ref shtns_create and with an associated grid (see \ref shtns_set_grid_auto )
/// \returns device_id on success, or -1 on failure.
int cushtns_init_gpu(shtns_cfg shtns);

/// Clone a gpu-enabled shtns config, and assign it to different streams (to allow compute overlap and/or usage from multiple threads).
/// This implies allocation of memory on the GPU and other limited resources, and thus may fail.
/// \param[in] shtns is a valid shtns configuration created with \ref shtns_create and with an associated grid (see \ref shtns_set_grid_auto )
/// \param[in] compute_stream is a cuda Stream that will be used for transforms. If 0, the default (0) stream will be used.
/// \param[in] transfer_stream is a cuda Stream that will be used for data transfers between host and device for auto-offload mode. If 0, a new stream will be created and used.
/// \returns a new \ref shtns_cfg that can safely be used concurrently with the original one.
shtns_cfg cushtns_clone(shtns_cfg shtns, shtns_gpu_stream_t compute_stream, shtns_gpu_stream_t transfer_stream);

/// Set user-specified streams for compute (including fft) and transfer.
void cushtns_set_streams(shtns_cfg shtns, shtns_gpu_stream_t compute_stream, shtns_gpu_stream_t transfer_stream);

/// Release resources needed for GPU transforms, which won't work after this call.
void cushtns_release_gpu(shtns_cfg);
///@}

void cushtns_profiling(shtns_cfg, int on);
double cushtns_profiling_read_time(shtns_cfg, double* time_1, double* time_2);

const char* cushtns_get_cfg_info(shtns_cfg);
///@}

#ifdef __cplusplus
}
#endif /* __cplusplus */
