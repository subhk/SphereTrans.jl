/*
 * Copyright (c) 2010-2023 Centre National de la Recherche Scientifique.
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

/* NOTES:
 * - the cuda transforms are NOT thread-safe. Use cushtns_clone() to clone transforms for each thread.
*/

/* TODO
 * 1) use static polar optimization (from constant memory ?)
 * 2) implement transposed layout, with appropriate transpose + fft
 */

/* Session with S. Chauveau from nvidia:
 * useful metrics = achieved_occupancy, cache_hit
 * for leg_m_lowllim, the "while(l<llim)" loop:
 * 		0) full al, ql load before the while loop.	=> DONE
 * 		1) reduce pointer update by moving ql and al updates into the "if" statement.	=> DONE
 * 	    2a) try to use a double-buffer (indexed by b, switched by b=1-b)		=> only 1 __syncthread() instead of 2.
 * 	OR:	2b) preload al, ql into registers => may reduce the waiting at __syncthreads()
 * 		3) unroll by hand the while loop (with an inner-loop of fixed size)		=> DONE
 * 		4) introduce NWAY (1 to 4) to avoid the need of several blocks in theta => one block for all means al, ql are read only once!	=> DONE
 * 				=> increases register pressure, but may be OK !
 */

// NOTE variables gridDim.x, blockIdx.x, blockDim.x, threadIdx.x, and warpSize are defined in device functions

#include "sht_private.h"

#ifndef SHTNS_ISHIOKA
#error "GPU transform requires SHTNS_ISHIOKA"
#endif

enum cushtns_flags { CUSHT_OFF=0, CUSHT_ON=1, CUSHT_OWN_XFER_STREAM=4, CUSHT_NO_ISHIOKA=32, CUSHT_PROFILING=64};
#define FFT_PHI_CONTIG (FFT_PHI_CONTIG_SPLIT | FFT_PHI_CONTIG_ODD)

#include "sht_gpu_kernels.cu"

/// include a compilable version of cuda_legendre.gen.cu (zero-terminated) :
const char *src_leg =
	#include "SHT/cuda_legendre.inc"
;

int ncplx_align(int nphi) {	const int align=2;	return ((nphi/2+1 + (align-1))/align)*align;	}

/* TOOL FUNCTIONS */

extern "C"
void* shtns_malloc(size_t size) {
	void* ptr = 0;
	cudaError_t err = cudaMallocHost(&ptr, size);		// try to allocate pinned memory (for faster transfers !)
	if (err != cudaSuccess) {
		cudaGetLastError();		// clears the error status.
		#if SHT_VERBOSE > 1
		printf("!WARNING! [shtns_malloc] failed to alloc pinned memory. using regular memory instead.\n");
		#endif
		ptr = VMALLOC(size);		// return regular memory instead...
	}
	return ptr;
}

extern "C"
void shtns_free(void* p) {
	if (p) {
		cudaError_t err = cudaFreeHost(p);
		if (err != cudaSuccess) {
			cudaGetLastError();		// clears the error status.
			#if SHT_VERBOSE > 1
			printf("!WARNING! [sntns_free] not page locked memory. trying regular free...\n");
			#endif
			VFREE(p);
		}
	}
}

// PROFILING TOOLS

inline void profiling_record_time(shtns_cfg shtns, int idx, cudaStream_t strm) {
	if (shtns->cu_flags & CUSHT_PROFILING) cudaEventRecord(shtns->gpu_timer[idx], strm);
}

extern "C"
void cushtns_profiling(shtns_cfg shtns, int on) {
	if (shtns->d_clm == 0) return;		// not a GPU transform!
	const bool time_kernels = shtns->cu_flags & CUSHT_PROFILING;
	if (on && !time_kernels) {
		for (int i=0; i<3; i++) cudaEventCreate(&shtns->gpu_timer[i]);
		shtns->cu_flags |= CUSHT_PROFILING;
	}
	if (!on && time_kernels) {
		for (int i=2; i>=0; i--) cudaEventDestroy(shtns->gpu_timer[i]);
		shtns->cu_flags &= ~((int)CUSHT_PROFILING);
	}
}

extern "C"
double cushtns_profiling_read_time(shtns_cfg shtns, double* time_1, double* time_2)
{
	float t1_ms = 0;	float t2_ms = 0;	// times in milliseconds
	if (shtns->cu_flags & CUSHT_PROFILING) {
		cudaEventSynchronize(shtns->gpu_timer[2]);		// wait for the events to happen
		cudaEventElapsedTime(&t1_ms, shtns->gpu_timer[0], shtns->gpu_timer[1]);
		cudaEventElapsedTime(&t2_ms, shtns->gpu_timer[1], shtns->gpu_timer[2]);
		cudaGetLastError();		// clears error which may arise if the event has not been recorded first.
	}
	*time_1 = t1_ms*1e-3;	*time_2 = t2_ms*1e-3;
	return (t1_ms+t2_ms)*1e-3;
}

static void destroy_cuda_buffer_fft(shtns_cfg shtns)
{
	#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
	if (shtns->nphi > 1) cufftDestroy(shtns->cufft_plan);
	#endif
	#ifdef VKFFT_BACKEND
	deleteVkFFT(&shtns->vkfft_plan);
	#endif
	if (shtns->gpu_buf_in) cudaFree(shtns->gpu_buf_in);
}

#define CACHE_LINE_GPU 128
inline void align_ptr(void** p, size_t ofs, uintptr_t align) {
	*p = (void*) ((((uintptr_t) (*p)) + (ofs+align-1)) &~ (align-1));
}

// WARNING! streams should be set BEFORE this routine is called!!
static int init_cuda_buffer_fft(shtns_cfg shtns, int cuda_gpu_id, int sizeof_real)
{
	cudaError_t err = cudaSuccess;
	int err_count = 0;

	/* GPU FFT init */
	int nfft = shtns->nphi;
	//int nreal = 2*(nfft/2+1);
	if (nfft > 1) {
		#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
			cufftResult res = CUFFT_SUCCESS;
			if (shtns->fft_mode & FFT_THETA_CONTIG) {
				printf("!!! Use theta-contiguous FFT on GPU !!!\n");
				long howmany = shtns->nlat_2 * shtns->howmany;		// support batched transforms
				long dist = shtns->nlat_padded / 2;
				res = cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, dist, 1, &nfft, dist, 1, (sizeof_real==4) ? CUFFT_C2C : CUFFT_Z2Z, howmany);
			} else {
				printf("WARNING: layout not available on GPU with cuFFT/rocFFT. Try to compile with VkFFT instead.\n");
				err_count ++;
				return 1;
			}
			if (res != CUFFT_SUCCESS) {
				printf("cufft init FAILED with error code %d\n", res);
				err_count ++;
			}
			res = cufftSetStream(shtns->cufft_plan, shtns->comp_stream);	// select stream for cufft
			size_t worksize = 0;
			cufftGetSize(shtns->cufft_plan, &worksize);
			#if SHT_VERBOSE > 1
				printf("cufft work-area size: %ld \t nlat*nphi = %d\n", worksize/sizeof_real, shtns->nlat * shtns->nphi);
			#endif
		#endif

		#ifdef VKFFT_BACKEND
			CUdevice vkfft_device_struct;
			VkFFTConfiguration config = {};		//zero-initialize configuration
			if (shtns->fft_mode & FFT_THETA_CONTIG) {
				printf("!!! Use theta-contiguous FFT on GPU !!!\n");
				const long dist = shtns->nlat_padded / 2;
				config.FFTdim = 2; //FFT dimension: 1D, but we use a second dimension to get non-unit strides.
				config.size[0] = shtns->nlat_2;
				config.size[1] = nfft;
				config.bufferStride[0] = dist;
				config.bufferStride[1] = dist * nfft;
				config.omitDimension[0] = 1;		// no FFT on the first dimension.
				if (2*(shtns->mmax+1) <= nfft) {	// let vkFFT perform the zero-padding (saves memory bandwidth)
					config.performZeropadding[1] = 1;
					config.frequencyZeroPadding = 1;
					config.fft_zeropad_left[1] = shtns->mmax + 1;			// first zero element
					config.fft_zeropad_right[1] = nfft - shtns->mmax;		// first non-zero element
				}
				config.numberBatches = shtns->howmany;
				#if SHTNS_GPU == 2
				if (dist*2*sizeof_real*(2*MMAX+1) > 8192*1024*1.2)   // when FFT larger than L2 cache (with a safety factor), always ensure coalescedMemory=64 on AMD GPU
					config.coalescedMemory = 64;    // important for AMD MI100, MI200+, up to 30% better, eg lmax=1023, nlorder=1
				#endif
			} else if (shtns->fft_mode & FFT_PHI_CONTIG) {
				printf("!!! Use phi-contiguous FFT on GPU (with transpose step) !!!\n");
				long howmany = shtns->nlat * shtns->howmany;		// support batched transforms
				config.FFTdim = 1; // 1D FFT
				config.size[0] = nfft;
				config.isInputFormatted = 1;		// out-of-place: separate buffer for input and output
				config.inverseReturnToInputBuffer = 1;
				config.inputBufferStride[0] = nfft;		// spatial data
				config.bufferStride[0] = ncplx_align(nfft);	// spectral data
				if (0) {	// disable zero-padding for now, as it is broken for large nfft
					config.performZeropadding[0] = 1;
					config.frequencyZeroPadding = 1;
					config.fft_zeropad_left[0] = shtns->mmax + 1;			// first zero element
					config.fft_zeropad_right[0] = nfft - shtns->mmax;		// first non-zero element
				}
				config.numberBatches = howmany;
				config.performR2C = 1;
				//config.disableMergeSequencesR2C = 1;		// reduces performance (don't use)
				//config.disableReorderFourStep = 1;		// avoids the use of temp buffer for large transforms at the cost of a mangled output.
			} else {
				printf("WARNING: layout not available on GPU.\n");
				err_count ++;
				return 1;
			}
			config.doublePrecision = sizeof_real / 8;
			cuDeviceGet(&vkfft_device_struct, cuda_gpu_id);
			config.device = &vkfft_device_struct;
			config.stream = &shtns->comp_stream;
			config.num_streams = 1;
			VkFFTResult vk_res = initializeVkFFT(&shtns->vkfft_plan, config);

			const int ver = VkFFTGetVersion();
			printf("=> Using VkFFT v%d.%d.%d\n",ver/10000,(ver%10000)/100,ver%100);
			if (vk_res != VKFFT_SUCCESS) {
				printf("vkfft init FAILED with error code %d\n", vk_res);
				err_count ++;
			}
		#endif
	}

	// Allocate working arrays for SHT on GPU:
	const long howmany = shtns->howmany;		// batch size
	const long nlm2 = shtns->nlm + (shtns->mmax+1);		// one more data per m
	const long nphi = shtns->nphi;
	const size_t nlm_stride = ((2*nlm2+WARPSZE-1)/WARPSZE) * WARPSZE;
	const size_t spat_stride = ((shtns->nlat_padded*(nphi + (nphi/2==shtns->mmax))+WARPSZE-1)/WARPSZE) * WARPSZE * howmany;		// for odd nphi, reserve more space to store the full Fourier data
	const size_t dual_stride = (spat_stride < nlm_stride*howmany) ? nlm_stride*howmany : spat_stride;		// we need two spatial buffers to also hold spectral data.

	size_t sze = nlm_stride;		// 1 spectral buffer for scalar only ...
	if (shtns->mx_stdt) sze *= 2;	// ... 2 spectral buffer for vector transforms.
	if (shtns->fft_mode & FFT_PHI_CONTIG) {
		int ncplx = ncplx_align(nphi);
		size_t fft_sze = ((shtns->nlat_padded*2*ncplx+WARPSZE-1)/WARPSZE) * WARPSZE;	// Fourier data in R2C format takes up a little more space
		if (shtns->mx_stdt) {	// for vector transform, we need to keep 2 spectral buffers together with a Fourier buffer:
			sze += fft_sze;
		} else if (fft_sze > sze) sze = fft_sze;		// one spatial buffer for FFT -OR- 2 spectral buffers should fit in.
	}
	err = cudaMalloc( (void **)&shtns->gpu_buf_in,  sze * sizeof_real * howmany );
	if (err != cudaSuccess)	{	err_count++;	CUDA_ERROR_CHECK;  }

	shtns->nlm_stride = nlm_stride;
	shtns->spat_stride = dual_stride;

	return err_count;
}

/// allocate buffers on the GPU for copying data to and from CPU memory 
extern "C"
int init_gpu_staging_buffer(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	if (shtns->xfer_stream == 0) {
		err = cudaStreamCreateWithFlags(&shtns->xfer_stream, cudaStreamNonBlocking);		// stream for async data transfer.
		if (err != cudaSuccess)	{	CUDA_ERROR_CHECK;  return 1; }
		shtns->cu_flags |= CUSHT_OWN_XFER_STREAM;		// mark the transfer stream as managed by shtns.
	}

	double* gpu_mem = 0;
	size_t sze = shtns->spat_stride;		// 1 spatial or spectral buffer
	if (shtns->mx_stdt) sze *= 3;		// for vector transform: 3 spatial or spectral buffers.
	err = cudaMalloc( (void **)&gpu_mem, sze * shtns->sizeof_real );		// maximum GPU memory required for SHT auto-offloading
	if (err != cudaSuccess)	{	CUDA_ERROR_CHECK;  return 1; }
	shtns->gpu_staging_mem = gpu_mem;
	return 0;
}

void read_line_int(FILE* fp, int* val)
{
	char s[32];
	char* x = fgets(s, 30, fp);		// read line
	if (x) sscanf(x, "%d", val);		// convert to int
}

extern "C"
const char* cushtns_get_cfg_info(shtns_cfg shtns)
{
	static char s[160];
	if (shtns->d_clm == 0) return 0;
	sprintf(s,"blocks=(%d,%d,%d) ", shtns->gridDim_x[0], shtns->gridDim_x[1], shtns->gridDim_x[2]);
	sprintf(s,"blksze=(%d,%d,%d) ", shtns->nwarp[0]*WARPSZE, shtns->nwarp[1]*WARPSZE, shtns->nwarp[2]*WARPSZE);
	sprintf(s,"nf=(%d,%d) ", shtns->howmany/shtns->gridDim_y[0], shtns->howmany/shtns->gridDim_y[1]);
	sprintf(s,"nw_s=%d ", (shtns->nlat_2 + shtns->gridDim_x[0]*shtns->nwarp[0]*WARPSZE-1) / (shtns->gridDim_x[0]*shtns->nwarp[0]*WARPSZE));
	sprintf(s,"lspan_a=%d", shtns->lspan_a);
	sprintf(s,"fp%d/fp%d", shtns->sizeof_real*8, shtns->sizeof_real_g*8);
	return s;
}

/// use some apriori metric to choose a good blocksize. An optimal one would require to measure.
static int optimize_nwarp(int* nwarp, int n_target, int nw, float loss_max, const bool div_by_2=false)
{
	float loss;
	int n = (div_by_2) ? *nwarp*2 : *nwarp+1;
	int nb = 0;
	do {
		n = (div_by_2) ? n/2 : n-1;
		nb = (n_target + n*nw-1)/(n*nw);	// number of block (should be minimum)
		loss = nb*n*nw / (float) n_target;
		if (SHT_VERBOSE > 1) printf("%d %d %f\n", n, nb, loss);
	} while (n>1 && loss>loss_max);		// either we found a good value, with less than 15% overhead due to large block size, or we reach n=1
	*nwarp = n;
	return nb;
}

/* When should we fuse pre/post-processing of harmonic coefficients with (i)legendre kernels ?
 * A separate pre/post-processing kernel reads and write all fields, that is NF_A * 2 fields access on memory.
 * Doing the pre/post-processing in the transform kernel (sh2ish_fuse) means additional read 
 * of coefficients NLAT_2/BLOCKSIZE times, which means
 * 		NLAT_2/BLOCKSIZE*X fields (where X is 0.75 for ishioka and 0.5 otherwise)
 * It is therefore better in terms of bandwidth when
 * 		NLAT_2/BLOCKSIZE*X <= 2*NF_A
 * 	or	NLAT_2 <= 8*NF_A*BLOCKSIZE/3  for Ishioka's reccurence
 * and  NLAT_2 <= 4*NF_A*BLOCKSIZE    for Standard recurrence
 */

int init_cuda_program(shtns_cfg shtns, const char* gpu_arch_target)
{
	const int nwarp_target = (shtns->nlat_2 + WARPSZE-1)/WARPSZE;		// number of 'warps' needed for nlat_2 points
	const bool hi_llim = (shtns->mmax > 0  &&  
		shtns->lmax > ((shtns->sizeof_real_g == 4) ? SHT_L_RESCALE_FLY_FLOAT : SHT_L_RESCALE_FLY));	// special rescaling needed
	bool sh2ish_fuse = (SHT_ALLOW_SH2ISH_FUSE  &&  shtns->lmax < 1024);		// don't fuse when polar optimization is profitable
	int nwarp_s=4;		// 1 to 4 warps is a good choice on V100 for vector or when sh2ish is disabled. Usually, 4 is a bit better.
	int nwarp_a=1;		// 1 WARP is by far the best choice here, at least on V100
	const int nw_a=1;	// only one point per thread possible for analysis
	int nw_s=2;		int nf_s=1;			int nf_a=1;
	int lspan_a = 16;	// V100 and MI100: 16/nf_a works best (mmax>0)
#if WARPSZE == 32
	if (nwarp_target % 3 == 0) nw_s=3;	// if we need a multiple of 3, nw_s=3 is likely a bit better
	// adjust values (heuristics)
	if (shtns->howmany % 4 == 0) 	  {	nf_s=4;	nw_s=1;		nf_a=4;	}
	else if (shtns->howmany % 2 == 0) {	nf_s=2;	nw_s=2; 	nf_a=2;	}
	else if (shtns->howmany % 3 == 0) { nf_s=3; nw_s=1; 	nf_a=1;	}
#else
	const bool gfx90a = (strcmp(gpu_arch_target,"gfx90a") >= 0);	// MI200 series
	const bool gfx94x = (strcmp(gpu_arch_target,"gfx94") >= 0);		// MI300 series
	if (shtns->howmany % 2 == 0) {	nf_a=2;		nf_s=2; }
	if (shtns->sizeof_real == 4  &&  nf_a==2) lspan_a = 32;
	if (gfx90a || gfx94x) {	// MI200+ or MI300+
		nw_s=4;
		if (nf_s==1  &&  shtns->howmany % 3 == 0) {	nf_s=3;	nw_s=3; }
		if (hi_llim  &&  shtns->howmany % 4 == 0) { nf_s=4;	nw_s=2; }	// nw_s=2 also allows fusion with sh2ish
		if (shtns->sizeof_real == 4) {	// maximize nf_s
			nw_s = 3;		// nw_s=4 should be avoided in fp32 mode
			if (nwarp_target % 3  &&  (nwarp_target % 2 == 0  ||  (nwarp_target+1) % 3))  nw_s = 2;
			if (shtns->howmany % 4 == 0) 	  {	nf_a=4;		nf_s=4;	}
			else if (shtns->howmany % 3 == 0) {	nf_s=3;	}
		}
		if (shtns->howmany % 4 == 0)	{ nf_a=4;	lspan_a=32; }	// good for fp32 & fp64 with new internal layout
	} else {	// assume MI100
		if (nwarp_target > 2  &&  !hi_llim)	nf_s=1;
	}
	if (shtns->howmany % 4 == 0  &&  nwarp_target == 1)	nf_s=4;
	if (hi_llim)	nwarp_s=1;
#endif
	//if (nf_s==4 && shtns->howmany / nf_s * nwarp_target / nw_s < 25) nf_s=2;		// ensure enough parallelism is exposed?
	if (shtns->mmax == 0) {
		lspan_a *= 2;
		//sh2ish_fuse = false;	// don't fuse mmax=0
	}
	if (hi_llim  &&  nw_s > 2) nw_s=2;	// nw_s = 1 or 2 only with hi_llim
	lspan_a /= nf_a;

	if (getenv("SHTNS_GPU_CONF"))
	{	// override from sht_gpu.conf file
		FILE *fp = fopen("sht_gpu.conf", "r");
		if (fp) {
			printf("WARNING! defaults override from sht_gpu.conf\n");
			read_line_int(fp, &nwarp_s);	read_line_int(fp, &nf_s);	read_line_int(fp, &nw_s);
			read_line_int(fp, &nwarp_a);	read_line_int(fp, &nf_a);	read_line_int(fp, &lspan_a);
			fclose(fp);
		}
	}

	// for analysis, simple:
	if (SHT_VERBOSE > 1) printf("optimize analysis:\n");
	optimize_nwarp(&nwarp_a, nwarp_target, nw_a, 1.14f);
	// for regular scalar synthesis (not fused) and vector synthesis
	if (SHT_VERBOSE > 1) printf("optimize vector synthesis:\n");
	optimize_nwarp(&nwarp_s, nwarp_target, nw_s, 1.14f);
	if (nw_s > 1  &&  nwarp_s == 1)	{
		if (SHT_VERBOSE > 1) printf("optimize NW synthesis:\n");
		optimize_nwarp(&nw_s, nwarp_target, nwarp_s, 1.3f);		// maybe we should reduce nw_s ?
	}
	#if WARPSZE == 64
		// maximize nf_s when nw_s is small
		if (nw_s <= 2 && nf_s < 4) 	for (int k=((shtns->sizeof_real==4) ? 8 : 4); k>0; k--) if (shtns->howmany % k == 0) { nf_s=k; break; }
	#endif

	int nwarp_s0=0;		int nblocks_s0=0;
	if (sh2ish_fuse) {
		// for scalar synthesis we should try to fuse sh2ish and leg_m_kernel for better performance.
		// this requires a larger blocksize (nwarp_s), up to MAX_THREADS_PER_BLOCK.
		//if (nw_s == 4 && nwarp_s == 1) {  nw_s=2; nwarp_s=2; }	// MI250: nw_s=4 does not work well with fuse
		nwarp_s0 = MAX_THREADS_PER_BLOCK/WARPSZE;		// start with maximum number of warps per block
		if (SHT_VERBOSE > 1) printf("optimize scalar synthesis:\n");
		nblocks_s0 = optimize_nwarp(&nwarp_s0, nwarp_target, nw_s, 1.14f);
		if (nwarp_s0==1  && nblocks_s0<=MAX_THREADS_PER_BLOCK/WARPSZE) { nwarp_s0=nblocks_s0;  nblocks_s0=1; }	// if one warp and several blocks, do one block and several warps!
		if (nblocks_s0 > 1) sh2ish_fuse = false;	// disable sh2ish_fuse, very likely slower or only marginally faster
		//if (nw_s == 4 && shtns->sizeof_real==8) sh2ish_fuse = false;			// MI250
		if (hi_llim && shtns->sizeof_real == 8) sh2ish_fuse = false;	// don't fuse hi_llim double-precision.
	}
	//if (nf_a * shtns->nlat_2 <= 512   &&   !ISHIOKA)  ==> we can include ish2sh into the ilegendre kernel.

	// also store into plan the kernel launch parameters:
	shtns->nwarp[0] = nwarp_s;		shtns->nwarp[1] = nwarp_a;		shtns->nwarp[2] = sh2ish_fuse ? nwarp_s0 : 0;
	shtns->gridDim_x[0] = (shtns->nlat_2 + nw_s*nwarp_s*WARPSZE-1)/(nw_s*nwarp_s*WARPSZE);
	shtns->gridDim_x[1] = (shtns->nlat_2 + nw_a*nwarp_a*WARPSZE-1)/(nw_a*nwarp_a*WARPSZE);
	shtns->gridDim_x[2] = sh2ish_fuse ? nblocks_s0 : 0;
	shtns->gridDim_y[0] = shtns->howmany / nf_s;
	shtns->gridDim_y[1] = shtns->howmany / nf_a;
	shtns->lspan_a = lspan_a;
	#if SHT_VERBOSE > 1
		printf("launch params: nblocks=(%d, %d, %d)\n", shtns->gridDim_x[0], shtns->gridDim_x[1], shtns->gridDim_x[2]);
	#endif

	const int sze_src = 100*1024;	// 100 KB
	char* const src = (char*) malloc(sze_src);
	// define what we need
	char* s = src;
	s += sprintf(s, "#define WARPSZE %d\n", WARPSZE);
	s += sprintf(s, "#define LMAX %d\n", shtns->lmax);
	s += sprintf(s, "#define MRES %d\n", shtns->mres);
	s += sprintf(s, "#define HI_LLIM %d\n", (hi_llim) ? 1 : 0);
	s += sprintf(s, "#define M0_ONLY %d\n", (shtns->mmax == 0) ? 1 : 0);
	s += sprintf(s, "#define ROBERT_FORM %d\n", shtns->robert_form);
	s += sprintf(s, "#define BLKSZE_S %d\n", nwarp_s*WARPSZE);
	s += sprintf(s, "#define BLKSZE_A %d\n", nwarp_a*WARPSZE);
	s += sprintf(s, "#define BLKSZE_SH2ISH %d\n", shtns->nwarp[2] * WARPSZE);	// 0 in case sh2ish is disabled
	s += sprintf(s, "#define NF_S %d\n", nf_s);
	s += sprintf(s, "#define NF_A %d\n", nf_a);
	s += sprintf(s, "#define LSPAN_A %d\n", lspan_a);
	s += sprintf(s, "#define NW_S %d\n", nw_s);
	s += sprintf(s, "#define MPOS_SCALE %g\n", shtns->mpos_scale_analys * ((shtns->fft_mode & FFT_PHI_CONTIG) ? 2 : 1));
	s += sprintf(s, "#define NLAT_2 %d\n", shtns->nlat_2);
	s += sprintf(s, (shtns->sizeof_real == 4) ? "typedef float real;\ntypedef float2 real2;\n" : "typedef double real;\ntypedef double2 real2;\n");	// single or double-precision data
	if (shtns->sizeof_real_g == 4) {
		 s += sprintf(s, "typedef float real_g;\n#define SHT_ACCURACY 1.0e-15f\n#define SHT_SCALE_FACTOR 7.2057594037927936e16f\n");	// for single-precision recurrence
	} else {
		//s += sprintf(s, "#define SHT_HI_PREC 2\n");		// improve numerical accuracy by using full FP64 for m=0 (not only recurrence, but also accumulators) ==> TODO: some work needed in kernels.
		s += sprintf(s, "typedef double real_g;\n#define SHT_ACCURACY 1.0e-33\n#define SHT_SCALE_FACTOR 2.0370359763344860863e90\n");	// for double-precision recurrence
	}
	s += sprintf(s, "#define SHT_HI_PREC %d\n", (shtns->sizeof_real == 4) ? 1 : 0);		// improve numerical accuracy by computing the mean separately for scalar (S=0) transforms.
	if ((shtns->kernel_flags & CUSHT_NO_ISHIOKA) == 0) 	s += sprintf(s, "#define LEG_ISHIOKA\n#define ILEG_ISHIOKA\n");		// use ishioka's recurrence in gpu kernels
	if (shtns->fft_mode & FFT_PHI_CONTIG) s += sprintf(s, "#define LAYOUT_REAL_FFT\n");
	#if SHT_VERBOSE > 1
		printf("%s", src);		// displays the defines for debug purposes
	#endif

	// first look for file to read (allows quick changes without recompiling), otherwise use embedded kernel source.
	FILE *fp = fopen("SHT/cuda_legendre.gen.cu", "r");
	if (fp) {
		int k = fread(s, 1, sze_src-10-(s-src), fp);
		s[k]=0;	// zero-terminated
		fclose(fp);
	} else 	snprintf(s, sze_src-10-(s-src), "%s", src_leg);		// copy embedded kernel source
	if (getenv("SHTNS_PRINT_SRC")) {		// allows to dump the whole kernel source to a file
		FILE *fp = fopen("_shtns_gen_tmp_.cu", "w");
		if (fp) {	 fprintf(fp, "%s", src);	fclose(fp);		}
	}

	nvrtcProgram prog;
	nvrtcResult rtc_res = nvrtcCreateProgram(&prog, src, "shtns.cu", 0, NULL, NULL);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcCreateProgram failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;	// fail
	}
	const char *ker_inst[] = {"leg_m_kernel<0>", "leg_m_kernel<1>", "ileg_m_kernel<0>", "ileg_m_kernel<1>"};
	for (int k=0; k<4; k++) {
		rtc_res = nvrtcAddNameExpression(prog,  ker_inst[k]);
		if (rtc_res != NVRTC_SUCCESS) {
			printf("ERROR nvrtcAddNameExpression(\"%s\") failed with error '%s'\n", ker_inst[k], nvrtcGetErrorString(rtc_res));
			return 1;	// fail
		}
	}

	// Compile
	char arch[64];
#if SHTNS_GPU == 1
	#if CUDA_VERSION >= 11030
		snprintf(arch, 64, "-arch=sm%s", gpu_arch_target);		// compile for the current gpu
		const char *opts[] = {"-std=c++11", "-ftz=true", "-lineinfo", "--ptxas-options","-v", arch};
	#else
		snprintf(arch, 64, "-arch=compute%s", gpu_arch_target);		// compile for the current gpu
		const char *opts[] = {"-std=c++11", "-ftz=true", "-lineinfo", arch};
	#endif
#else
	snprintf(arch, 64, "--offload-arch=%s", gpu_arch_target);             // compile for the current gpu
	const char *opts[] = {"-std=c++11", "-O3", arch};
#endif
	#if SHT_VERBOSE > 1
		printf("compiling cuda kernels (lmax=%d, nlat=%d, nbatch=%d) for %s\n", shtns->lmax, shtns->nlat, shtns->howmany, gpu_arch_target);
	#endif
	rtc_res = nvrtcCompileProgram(prog, sizeof(opts)/sizeof(const char*), opts);
	if ((rtc_res != NVRTC_SUCCESS) || (SHT_VERBOSE > 1)) {		// show compile log in case of failure, or if verbose (debug) output required
		size_t sze = 0;
		nvrtcGetProgramLogSize (prog, &sze);
		char* log = (char*) malloc(sze);
		nvrtcGetProgramLog (prog, log);
		if (sze > 0) printf("%s", log);
		free(log);
	}
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcCompileProgram failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;	// fail
	}

	// Obtain PTX of the program.
	size_t sze;
	rtc_res = nvrtcGetPTXSize(prog, &sze);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcGetPTXSize failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;
	}
	char *ptx = (char*) malloc(sze);
	rtc_res = nvrtcGetPTX(prog, ptx);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcGetPTX failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;
	}

	// Load the generated PTX module
	CUmodule module;
	CUresult cu_res = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	if (cu_res != CUDA_SUCCESS) {
		printf("\nERROR cuModuleLoadDataEx failed with error %d\n", cu_res);
		return 1;
	}

	// get the kernel pointers
	for (int k=0; k<4; k++) {
		const char *name;
		rtc_res = nvrtcGetLoweredName(prog, ker_inst[k], &name);
		if (rtc_res != NVRTC_SUCCESS) {
			printf("\nERROR nvrtcGetLoweredName(%s) failed with error '%s'\n", ker_inst[k], nvrtcGetErrorString(rtc_res));
			return 1;
		}
		CUfunction kernel;
		cu_res = cuModuleGetFunction(&kernel, module, name);
		if (cu_res != CUDA_SUCCESS) {
			printf("\nERROR cuModuleGetFunction(%s -> %s) failed with error %d\n", ker_inst[k], name, cu_res);
			return 1;
		}
		shtns->gpu_kernels[k] = kernel;
	}
	shtns->gpu_module = module;

	nvrtcDestroyProgram(&prog);		// no longer needed.
	free(src);
	return 0;	// success
}


extern "C"
void cushtns_release_gpu(shtns_cfg shtns)
{
	if (shtns->gpu_staging_mem) cudaFree(shtns->gpu_staging_mem);
	if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
	destroy_cuda_buffer_fft(shtns);
	cushtns_profiling(shtns, 0);		// frees resources allocated for profiling
	// TODO: arrays possibly shared between different shtns_cfg should be deallocated ONLY if not used by other shtns_cfg.
	if (shtns->d_clm) cudaFree(shtns->d_clm);
	shtns->d_clm = 0;		// disable gpu.
	shtns->cu_flags = 0;
}

int gpu_upload_convert(void* dst, double* src, size_t n_real, int dst_sizeof_real)
{
	float* tmp_f = 0;
	if (dst_sizeof_real == 4) {		// convert from double to float
		tmp_f = (float*) malloc(n_real * sizeof(float));
		for (int i=0; i<n_real; i++) tmp_f[i] = src[i];
		src = (double*) &tmp_f[0];
	}
	cudaError_t err = cudaMemcpy(dst, src, n_real * dst_sizeof_real, cudaMemcpyHostToDevice);
	if (tmp_f) free(tmp_f);
	return (err != cudaSuccess);
}

extern "C"
int cushtns_init_gpu(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	const long nlm = shtns->nlm;
	const long nlat_2 = shtns->nlat_2;
	const int sizeof_real = shtns->sizeof_real;

	void *buf = 0;
	double *d_ct  = 0;
	double *d_mx_stdt = 0;
	double *d_mx_van = 0;
	double *d_xlm = 0;
	double *d_x2lm = 0;
	double *d_clm = 0;
	int err_count = 0;
	int device_id = -1;
	bool fast_fp64 = true;	// assume GPU has good fp64 performance

	cudaDeviceProp prop;
	cudaGetDevice(&device_id);
	err = cudaGetDeviceProperties(&prop, device_id);
	if (err != cudaSuccess) return -1;
	#if SHT_VERBOSE > 0
	#if SHTNS_GPU == 1
	printf("  cuda GPU #%d \"%s\" found (warp size = %d, compute capabilities = %d.%d", device_id, prop.name, prop.warpSize, prop.major, prop.minor);
	char gpu_arch_target[16];
	sprintf(gpu_arch_target, "_%d", prop.major*10 + prop.minor);		// the gpu_arch we will compile for!
	if (prop.major < 6 || prop.minor != 0) fast_fp64 = false;			// devices known with poor fp64 performance
	#elif SHTNS_GPU == 2
	printf("  hip GPU #%d \"%s\" found (warp size = %d", device_id, prop.gcnArchName, prop.warpSize);
	const char* gpu_arch_target = prop.gcnArchName;
	#endif
	printf(", CU=%d, max_threads=%d, %.3g GB, L2 cache=%.3gMB).\n", prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount, prop.totalGlobalMem/(1024.*1024.*1024.), prop.l2CacheSize/(1024.*1024.));
	#endif
	if (prop.warpSize != WARPSZE) return -1;		// failure, warpsize must be known at compile time (does it?).
	if (prop.major < 3) return -1;			// failure, SHTns requires compute cap. >= 3 (warp shuffle instructions)
	if (shtns->nlat % 4  &&  shtns->fft_mode == FFT_THETA_CONTIG) return -1;			// failure, nlat must be a multiple of 4 if theta-contiguous layout is requested

	int sizeof_real_g = (shtns->lmax > SHT_L_RESCALE_FLY_FLOAT && fast_fp64) ? sizeof(double) : sizeof_real;		// decide if recurrence happens in double precision or not.
	if (sizeof_real==4) {
		const char* vv = getenv("SHTNS_GPU_REC_PREC");		// 1 for float, everything else for double
		if (vv) 	sizeof_real_g = (atoi(vv)==1) ? 4 : 8;
	}
	shtns->sizeof_real_g = sizeof_real_g;
	shtns->kernel_flags = (sizeof_real_g == 4) ? CUSHT_NO_ISHIOKA : 0;		// ishioka disabled for fp32 recurrence (accuracy issues)
	{	// allow to override the default recurrence with environment variable
		const char* rec = getenv("SHTNS_GPU_REC");		// I for Ishioka, everything else for standard
		if (rec) 	shtns->kernel_flags = (rec[0] == 'I') ? 0 : CUSHT_NO_ISHIOKA;
	}

	const long nlm0 = nlm_calc(LMAX+4, MMAX, MRES);	// for ishioka
	const long nlm1 = nlm_calc(LMAX+2, MMAX, MRES);	// for non-ishioka
	// Allocate the coefficients vectors alm, ...
	size_t sze = 4*nlat_2*sizeof_real_g/sizeof_real;	// for cos(theta), sin(theta), weights, ...
	if (shtns->kernel_flags & CUSHT_NO_ISHIOKA) {
		sze += nlm1*sizeof_real_g/sizeof_real + (CACHE_LINE_GPU/sizeof_real-1);		// alm2
		sze += nlm1 + (CACHE_LINE_GPU/sizeof_real-1);		// glm
		if (shtns->glm != shtns->glm_analys)  sze += nlm1 + (CACHE_LINE_GPU/sizeof_real-1);		// reserve space for glm_analys if needed
	} else { 
		sze += nlm0 * sizeof_real_g/sizeof_real + 3*nlm0/2 + (CACHE_LINE_GPU/sizeof_real-1)*2;
		if (shtns->x2lm != shtns->xlm)  sze += 3*nlm0/2 +  (CACHE_LINE_GPU/sizeof_real-1);		// reserve space for x2lm
	}
	if (shtns->mx_stdt) sze += ( 2*nlm + (CACHE_LINE_GPU/sizeof_real-1) ) * ((shtns->mx_van == shtns->mx_stdt) ? 1 : 2);
	err = cudaMalloc(&buf, (sze + MAX_THREADS_PER_BLOCK-1)*sizeof_real);	// allow some overflow.
	if (err != cudaSuccess) err_count ++;
	if (err_count == 0) {
		const bool ish = (shtns->kernel_flags & CUSHT_NO_ISHIOKA) ? false : true;	// false for educed recurrence, usful for single precision, for which ishioka's recurrence loses too much accuracy.
		const long n_clm = (ish) ? nlm0 : nlm1;
		const long n_xlm = (ish) ? 3*nlm0/2 : nlm1;
		d_clm = (double*) buf;		align_ptr(&buf, n_clm * sizeof_real_g, CACHE_LINE_GPU);
		d_xlm = (double*) buf;		align_ptr(&buf, n_xlm * sizeof_real,   CACHE_LINE_GPU);
		err_count += gpu_upload_convert(d_clm, (ish) ? shtns->clm : shtns->alm2, n_clm, sizeof_real_g);
		err_count += gpu_upload_convert(d_xlm, (ish) ? shtns->xlm : shtns->glm,  n_xlm, sizeof_real);
		if ((ish && shtns->x2lm != shtns->xlm) || ((!ish) && shtns->glm_analys != shtns->glm)) {		// different arrays for Schmidt normalization
			d_x2lm = (double*) buf;		align_ptr(&buf, n_xlm * sizeof_real, CACHE_LINE_GPU);
			err_count += gpu_upload_convert(d_x2lm, (ish) ? shtns->x2lm : shtns->glm_analys, n_xlm, sizeof_real);
		} else d_x2lm = d_xlm;

		if (shtns->mx_stdt) {
			d_mx_van = d_mx_stdt = (double*) buf;	align_ptr(&buf, 2*nlm*sizeof_real, CACHE_LINE_GPU);	// Allocate the device matrix for d(sin(t))/dt
			err_count += gpu_upload_convert(d_mx_stdt, shtns->mx_stdt, 2*nlm, sizeof_real);
			if (shtns->mx_stdt != shtns->mx_van) {		// may be the same array
				d_mx_van = (double*) buf;	align_ptr(&buf, 2*nlm*sizeof_real, CACHE_LINE_GPU);  // Same thing for analysis
				err_count += gpu_upload_convert(d_mx_van, shtns->mx_van, 2*nlm, sizeof_real);
			}
		}
		// Allocate the device input vector cos(theta) and gauss weights, sin(theta) and 1/sin(theta)
		d_ct = (double*) buf;			align_ptr(&buf, 4*nlat_2*sizeof_real_g, CACHE_LINE_GPU);

		err_count += gpu_upload_convert(d_ct, shtns->ct, nlat_2, sizeof_real_g);
		err_count += gpu_upload_convert(((char*)d_ct) +   nlat_2*sizeof_real_g, shtns->wg, nlat_2, sizeof_real_g);
		err_count += gpu_upload_convert(((char*)d_ct) + 2*nlat_2*sizeof_real_g, shtns->st, nlat_2, sizeof_real_g);
		err_count += gpu_upload_convert(((char*)d_ct) + 3*nlat_2*sizeof_real_g, shtns->st_1, nlat_2, sizeof_real_g);
	}

	shtns->d_xlm = d_xlm;
	shtns->d_x2lm = d_x2lm;
	shtns->d_clm = d_clm;
	shtns->d_ct  = d_ct;
	shtns->d_mx_stdt = d_mx_stdt;
	shtns->d_mx_van = d_mx_van;

	err_count += init_cuda_buffer_fft(shtns, device_id, sizeof_real);
	err_count += init_cuda_program(shtns, gpu_arch_target);

	if (err_count != 0) {
		cushtns_release_gpu(shtns);
		return -1;	// fail
	}

	return device_id;		// success, return device_id
}

/// WARNING: cushtns_set_streams must be called BEFORE shtns_set_grid
extern "C"
void cushtns_set_streams(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	if (shtns->gpu_buf_in) {
		printf("[cushtns_set_streams] must be called before initializing shtns on GPU");
		exit(1);
	}
	shtns->comp_stream = compute_stream;
	if (transfer_stream != 0) {
		if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
		shtns->xfer_stream = transfer_stream;
		shtns->cu_flags &= ~((int)CUSHT_OWN_XFER_STREAM);		// we don't manage this stream
	}
}

extern "C"
shtns_cfg cushtns_clone(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	shtns_cfg sht_clone = shtns_create_with_grid(shtns, shtns->mmax, 0);		// copy the shtns_cfg, sharing all data.
	if (sht_clone == 0) return 0;

	int err_count = 0;
	sht_clone->cu_flags = 0;	// reset
	sht_clone->gpu_buf_in = 0;
	sht_clone->gpu_staging_mem = 0;
	sht_clone->xfer_stream = 0;
	cushtns_set_streams(sht_clone, compute_stream, transfer_stream);
	cushtns_init_gpu(sht_clone);	// for now, we should copy everything again
	//err_count += init_cuda_buffer_fft(sht_clone);
	if (shtns->gpu_staging_mem) {
		err_count += init_gpu_staging_buffer(sht_clone);
	}
	if (err_count == 0) {
		return sht_clone;
	} else {
		shtns_destroy(sht_clone);
		return 0;		// fail
	}
}

void fourier_to_spat_gpu(shtns_cfg shtns, void* q, const int mmax, const long sizeof_real = 8)
{
	const int nphi = shtns->nphi;
	if (nphi > 1) {
	#ifndef VKFFT_BACKEND
		cufftResult res = CUFFT_SUCCESS;
		void* xfft = q;
		if (2*(mmax+1) <= nphi) {
			const long nlat = shtns->nlat_padded;
			cudaMemsetAsync( ((char*)q) + sizeof_real*(mmax+1)*nlat, 0, sizeof_real*(nphi-2*mmax-1)*nlat, shtns->comp_stream );		// zero out m>mmax before fft
		}
		res = (sizeof_real==8) ? cufftExecZ2Z(shtns->cufft_plan, (cufftDoubleComplex*) xfft, (cufftDoubleComplex*) q, CUFFT_INVERSE) :
								 cufftExecC2C(shtns->cufft_plan, (cufftComplex*)       xfft, (cufftComplex*)       q, CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) printf("[fourier_to_spat_gpu] cufft error %d\n", res);
	#else
		char* xfft;
		VkFFTLaunchParams launchParams = {};
		if (shtns->fft_mode & FFT_PHI_CONTIG) {
			xfft = (char*) shtns->gpu_buf_in;
			if (shtns->mx_stdt) xfft += shtns->nlm_stride * shtns->howmany * 2*sizeof_real;	// vector transforms: do not overwrite temporary spectral data stored in gpu_buf_in
			int ncplx = ncplx_align(nphi);
			transpose_cplx_zero_C2R(shtns->comp_stream, q, xfft, shtns->nlat, ncplx, nphi/2, mmax, sizeof_real, shtns->howmany, shtns->spat_dist, shtns->nlat*2*ncplx);		// zero out m>mmax during transpose
			launchParams.buffer = (void**) &xfft;
			launchParams.inputBuffer = (void**) &q;
		} else {
			// THETA_CONTIGUOUS
			// rely on vkfft to avoid reading the unused Fourier modes above shtns->mmax
			if (mmax < shtns->mmax) {	// some zero must be added, only if more than nominal
				const long nlat = shtns->nlat_padded;
				const long width = sizeof_real*nlat*(nphi-2*mmax-1);
				char *dst = ((char*)q) + sizeof_real*nlat*(mmax+1);
				if (shtns->howmany == 1) {
					cudaMemsetAsync( dst, 0, width, shtns->comp_stream );		// zero out m>mmax before fft
				} else {
					cudaMemset2DAsync( dst, sizeof_real*shtns->spat_dist, 0, width, shtns->howmany, shtns->comp_stream );		// zero out m>mmax before fft
				}
			}
			launchParams.buffer = (void**) &q;
		}
		VkFFTAppend(&shtns->vkfft_plan, 1, &launchParams);
	#endif
	}
}

void spat_to_fourier_gpu(shtns_cfg shtns, void* q, const int mmax, const long sizeof_real = 8)
{
	const int nphi = shtns->nphi;
	if (nphi > 1) {
	#ifndef VKFFT_BACKEND
		cufftResult res = CUFFT_SUCCESS;
		void* xfft = q;
		res = (sizeof_real==8) ? cufftExecZ2Z(shtns->cufft_plan, (cufftDoubleComplex*) q, (cufftDoubleComplex*) xfft, CUFFT_FORWARD) :
								 cufftExecC2C(shtns->cufft_plan, (cufftComplex*) q, (cufftComplex*) xfft, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) printf("[spat_to_fourier_gpu] cufft error %d\n", res);
	#else
		char* xfft;
		VkFFTLaunchParams launchParams = {};
		if (shtns->fft_mode & FFT_PHI_CONTIG) {
			xfft = (char*) shtns->gpu_buf_in;
			if (shtns->mx_stdt) xfft += shtns->nlm_stride * shtns->howmany * 2*sizeof_real;	// vector transforms: do not overwrite temporary spectral data stored in gpu_buf_in
			launchParams.buffer = (void**) &xfft;
			launchParams.inputBuffer = (void**) &q;			
		} else {
			// THETA_CONTIGUOUS
			launchParams.buffer = (void**) &q;
		}
		VkFFTAppend(&shtns->vkfft_plan, -1, &launchParams);
		if (shtns->fft_mode & FFT_PHI_CONTIG) {
			int ncplx = ncplx_align(nphi);
			transpose_cplx_skip_R2C(shtns->comp_stream, xfft, q, ncplx, shtns->nlat, mmax, sizeof_real, shtns->howmany, shtns->nlat*2*ncplx, shtns->spat_dist);		// ignore m > mmax during transpose
		}
	#endif
	}
}

/************************
 * TRANSFORMS ON DEVICE *
 ************************/

static void legendre(shtns_cfg shtns, const int S, const void *ql, void *q, const int llim, const int mmax)
{
	int nlat_2 = shtns->nlat_2;
	cudaStream_t stream = shtns->comp_stream;

	const bool sh2ish_fuse = (SHT_ALLOW_SH2ISH_FUSE==1 && S==0 && shtns->nwarp[2]>0);
	int nlm_stride = (sh2ish_fuse) ? shtns->spec_dist*2 : shtns->nlm_stride;
	int par_idx = (sh2ish_fuse) ? 2 : 0;

	int llim_ = llim;
	void* params[11] = {&shtns->d_clm, &shtns->d_ct, &ql, &q, &llim_, &nlat_2, &shtns->nphi, &shtns->nlat_padded, &nlm_stride, &shtns->spat_dist, &shtns->d_xlm};
	cuLaunchKernel(shtns->gpu_kernels[S], 
			shtns->gridDim_x[par_idx], shtns->gridDim_y[0], mmax+1,		// grid dim
			shtns->nwarp[par_idx]*WARPSZE, 1, 1,					// block dim
			0, stream,								 // shared memory, stream
			params, 0);		// kernel params
}


/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
static void ilegendre(shtns_cfg shtns, const int S, const void *q, void* ql, const int llim)
{
	int mmax = shtns->mmax;
	int mres = shtns->mres;
	int nlat_2 = shtns->nlat_2;
	cudaStream_t stream = shtns->comp_stream;
	const int blksze = shtns->nwarp[1]*WARPSZE;

	if (blksze < nlat_2)	// this condition saves 10-15% on small sizes where blksze >= nlat_2
		cudaMemsetAsync(ql, 0, shtns->sizeof_real * shtns->nlm_stride * shtns->howmany, stream);		// set to zero before we start.

	if (llim < mmax*mres) mmax = llim / mres;	// truncate mmax too !

	int llim_ = llim;
	float w_norm_1_f = shtns->wg[-1];	// convert to float
	void* params[11] = {&shtns->d_clm, &shtns->d_ct, &q, &ql, &llim_, &nlat_2, &shtns->nphi, &shtns->nlat_padded, &shtns->spat_dist, &shtns->nlm_stride, &(shtns->wg[-1])};
	if (shtns->sizeof_real == 4) params[10] = &w_norm_1_f;		// weight_norm_1 as a float
	cuLaunchKernel(shtns->gpu_kernels[2+S], 		// analysis kernels
			shtns->gridDim_x[1], shtns->gridDim_y[1], mmax+1,		// grid dim
			blksze, 1, 1,					// block dim
			0, stream,								 // shared memory, stream
			params, 0);		// kernel params
}


/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, typename real=double>
void cuda_SH_to_spat(shtns_cfg shtns, std::complex<real>* d_Qlm, real *d_Vr, const long int llim, const int mmax)
{
	std::complex<real>* d_qlm = d_Qlm;
	
	if (sizeof(real) != shtns->sizeof_real) { printf("ERROR: SHTns plan not prepared for fp%ld data\n", sizeof(real)*8);	exit(1); }

	if (S==0) profiling_record_time(shtns, 0, shtns->comp_stream);
	if (S==0  &&  (SHT_ALLOW_SH2ISH_FUSE==0 || shtns->nwarp[2]==0)) {
		d_qlm = (std::complex<real>*) shtns->gpu_buf_in;
		sh2ishioka_gpu(shtns, d_Qlm, d_qlm, llim, mmax, S);
	} else
		if (d_Vr == (real*) d_Qlm) { printf("ERROR: cuda_SH_to_spat must have distinct in and out fields");	exit(1); }
	legendre(shtns, S, d_qlm, d_Vr, llim, mmax);
	if (S==0) profiling_record_time(shtns, 1, shtns->comp_stream);
	fourier_to_spat_gpu(shtns, d_Vr, mmax, sizeof(real));	// in-place
	if (S==0) profiling_record_time(shtns, 2, shtns->comp_stream);
}

/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, typename real=double>
void cuda_spat_to_SH(shtns_cfg shtns, real *d_Vr, std::complex<real>* d_Qlm, const long int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;		// truncate mmax too !

	if (sizeof(real) != shtns->sizeof_real) { printf("ERROR: SHTns plan not prepared for fp%ld data\n", sizeof(real)*8);	exit(1); }

	if (S==0) profiling_record_time(shtns, 0, shtns->comp_stream);
	spat_to_fourier_gpu(shtns, d_Vr, mmax, sizeof(real));
	if (S==0) profiling_record_time(shtns, 1, shtns->comp_stream);
	if (S==0) {
		std::complex<real>* d_Qlm_ish = (std::complex<real>*) shtns->gpu_buf_in;
		ilegendre(shtns, S, d_Vr, d_Qlm_ish, llim);
		ishioka2sh_gpu(shtns, d_Qlm_ish, d_Qlm, llim, mmax, S);
	} else {
		if (d_Vr == (real*) d_Qlm) { printf("ERROR: cuda_spat_to_SH must have distinct in and out fields");	exit(1); }
		ilegendre(shtns, S, d_Vr, d_Qlm, llim);
	}
	if (S==0) profiling_record_time(shtns, 2, shtns->comp_stream);
}


extern "C"
void cu_SH_to_spat(shtns_cfg shtns, cplx* d_Qlm, double *d_Vr, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !
	cuda_SH_to_spat<0>(shtns, d_Qlm, d_Vr, llim, mmax);
}

extern "C"
void cu_SH_to_spat_float(shtns_cfg shtns, cplx_f* d_Qlm, float *d_Vr, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !
	cuda_SH_to_spat<0,float>(shtns, d_Qlm, d_Vr, llim, mmax);
}


extern "C"
void cu_SHsphtor_to_spat(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
	double* d_vwlm = shtns->gpu_buf_in;

	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	sphtor2scal_gpu(shtns, d_Slm, d_Tlm, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1>(shtns, (cplx*) d_vwlm, d_Vt, llim+1, mmax);
	cuda_SH_to_spat<1>(shtns, (cplx*) (d_vwlm + nlm_stride), d_Vp, llim+1, mmax);
}

extern "C"
void cu_SHsphtor_to_spat_float(shtns_cfg shtns, cplx_f* d_Slm, cplx_f* d_Tlm, float* d_Vt, float* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
	float* d_vwlm = (float*) shtns->gpu_buf_in;

	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	sphtor2scal_gpu<float>(shtns, d_Slm, d_Tlm, (cplx_f*) d_vwlm, (cplx_f*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,float>(shtns, (cplx_f*) d_vwlm, d_Vt, llim+1, mmax);
	cuda_SH_to_spat<1,float>(shtns, (cplx_f*) (d_vwlm + nlm_stride), d_Vp, llim+1, mmax);
}



extern "C"
void cu_SHqst_to_spat(shtns_cfg shtns, cplx* d_Qlm, cplx* d_Slm, cplx* d_Tlm, double* d_Vr, double* d_Vt, double* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	cuda_SH_to_spat<0>(shtns, d_Qlm, d_Vr, llim, mmax);
	cu_SHsphtor_to_spat(shtns, d_Slm, d_Tlm, d_Vt, d_Vp, llim);
}

extern "C"
void cu_SHqst_to_spat_float(shtns_cfg shtns, cplx_f* d_Qlm, cplx_f* d_Slm, cplx_f* d_Tlm, float* d_Vr, float* d_Vt, float* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	cuda_SH_to_spat<0,float>(shtns, d_Qlm, d_Vr, llim, mmax);
	cu_SHsphtor_to_spat_float(shtns, d_Slm, d_Tlm, d_Vt, d_Vp, llim);
}

extern "C"
void cu_SHsph_to_spat(shtns_cfg shtns, cplx* d_Slm, double* d_Vt, double* d_Vp, int llim)
{
	cu_SHsphtor_to_spat(shtns, d_Slm, 0, d_Vt, d_Vp, llim);
}

extern "C"
void cu_SHtor_to_spat(shtns_cfg shtns, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
{
	cu_SHsphtor_to_spat(shtns, 0, d_Tlm, d_Vt, d_Vp, llim);
}

extern "C"
void cu_spat_to_SH(shtns_cfg shtns, double *d_Vr, cplx* d_Qlm, int llim)
{
	cuda_spat_to_SH<0>(shtns, d_Vr, d_Qlm, llim);
}

extern "C"
void cu_spat_to_SH_float(shtns_cfg shtns, float *d_Vr, cplx_f* d_Qlm, int llim)
{
	if (shtns->sizeof_real != 4) { printf("ERROR: SHTns plan not prepared for float");	exit(1); }
	cuda_spat_to_SH<0,float>(shtns, d_Vr, d_Qlm, llim);
}

extern "C"
void cu_spat_to_SHsphtor(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, int llim)
{
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
	double* d_vwlm = shtns->gpu_buf_in;

	// SHT on the GPU
	cuda_spat_to_SH<1>(shtns, Vt, (cplx*) d_vwlm, llim+1);
	cuda_spat_to_SH<1>(shtns, Vp, (cplx*) (d_vwlm + nlm_stride), llim+1);
	if (CUDA_ERROR_CHECK) return;
	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), Slm, Tlm, llim);
	CUDA_ERROR_CHECK;
}

extern "C"
void cu_spat_to_SHsphtor_float(shtns_cfg shtns, float *Vt, float *Vp, cplx_f *Slm, cplx_f *Tlm, int llim)
{
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
	float* d_vwlm = (float*)shtns->gpu_buf_in;

	// SHT on the GPU
	cuda_spat_to_SH<1,float>(shtns, Vt, (cplx_f*) d_vwlm, llim+1);
	cuda_spat_to_SH<1,float>(shtns, Vp, (cplx_f*) (d_vwlm + nlm_stride), llim+1);
	if (CUDA_ERROR_CHECK) return;
	scal2sphtor_gpu<float>(shtns, (cplx_f*) d_vwlm, (cplx_f*) (d_vwlm+nlm_stride), Slm, Tlm, llim);
	CUDA_ERROR_CHECK;
}


extern "C"
void cu_spat_to_SHqst(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, int llim)
{
	cuda_spat_to_SH<0>(shtns, Vr, Qlm, llim);
	cu_spat_to_SHsphtor(shtns, Vt,Vp, Slm,Tlm, llim);
}

extern "C"
void cu_spat_to_SHqst_float(shtns_cfg shtns, float *Vr, float *Vt, float *Vp, cplx_f *Qlm, cplx_f *Slm, cplx_f *Tlm, int llim)
{
	cuda_spat_to_SH<0,float>(shtns, Vr, Qlm, llim);
	cu_spat_to_SHsphtor_float(shtns, Vt,Vp, Slm,Tlm, llim);
}


/*******************************************************
 * TRANSFORMS OF HOST DATA, INCLUDING TRANSFERS TO GPU *
 *******************************************************/ 

cudaError_t copy_convert_field_to_gpu(void* dst, void* src, long n, int sizeof_real, cudaStream_t strm)
{
	cudaError_t err = cudaSuccess;
	if (sizeof_real == 4) {
		// convert
		float* tmp_f = (float*) malloc(n*sizeof(float));
		for (int i=0; i<n; i++) tmp_f[i] = ((double*) src)[i];
		// copy spectral data to GPU
		err = cudaMemcpyAsync(dst, tmp_f, n*sizeof(float), cudaMemcpyHostToDevice, strm);
		free(tmp_f);
	} else {
		err = cudaMemcpyAsync(dst, src, n*sizeof(double), cudaMemcpyHostToDevice, strm);
	}
	return err;
}

extern "C"
void SH_to_spat_gpu(shtns_cfg shtns, cplx *Qlm, double *Vr, const long int llim)
{
	cudaError_t err = cudaSuccess;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	int mmax = shtns->mmax;
	long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;

	double *d_q   = shtns->gpu_staging_mem;		// buffer for transfer (safe)
	double *d_qlm = d_q;		// "in-place" operation possible with ishioka
	if (SHT_ALLOW_SH2ISH_FUSE == 1  &&  shtns->nwarp[2]>0) d_qlm = shtns->gpu_buf_in; // include sh2ishioka into legendre kernel

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (shtns->howmany == 1) nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}

	if (howmany > 1  &&  2*shtns->spec_dist > shtns->nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }

	// copy spectral data to GPU
	err = copy_convert_field_to_gpu(d_qlm, Qlm, 2*nlm_pad, shtns->sizeof_real, shtns->comp_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	if (shtns->sizeof_real == 4) {
		cuda_SH_to_spat<0,float>(shtns, (cplx_f*) d_qlm, (float*)d_q, llim, mmax);	// start with Legendre, d_qlm may be available for Fourier.
	} else
	cuda_SH_to_spat<0>(shtns, (cplx*) d_qlm, d_q, llim, mmax);	// start with Legendre, d_qlm may be available for Fourier.
	if (CUDA_ERROR_CHECK) return;

	// copy back spatial data
	err = cudaMemcpy(Vr, d_q, (long) shtns->nspat * shtns->sizeof_real, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	if (shtns->sizeof_real == 4) {	// convert float to double in-place
		for (long i=shtns->nspat-1; i>=0; i--) 	Vr[i] = ((float*)Vr)[i];
	}
}


extern "C"
void SHsphtor_to_spat_gpu(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride * howmany;
	const long spat_stride = shtns->spat_stride;
	long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;
	const long sizeof_real = shtns->sizeof_real;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (howmany == 1) nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	if (howmany > 1  &&  2*nlm_pad > nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }

	char* d_vwlm = (char*)shtns->gpu_buf_in;
	char* d_vtp = (char*)shtns->gpu_staging_mem;
	void* d_Slm = 0;
	void* d_Tlm = 0;
	// (convert and) transfer to gpu
	if (Slm) {
		d_Slm = d_vtp;		
		err = copy_convert_field_to_gpu(d_Slm, Slm, 2*nlm_pad, sizeof_real, shtns->comp_stream);
		if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	}
	if (Tlm) {
		d_Tlm = d_vtp + nlm_stride*sizeof_real;
		err = copy_convert_field_to_gpu(d_Tlm, Tlm, 2*nlm_pad, sizeof_real, shtns->comp_stream);
		if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	}

	if(sizeof_real != 4)		// fp64
		sphtor2scal_gpu(shtns, (cplx*) d_Slm, (cplx*) d_Tlm, (cplx*) d_vwlm, (cplx*) (d_vwlm+sizeof(double)*nlm_stride), llim, mmax);
	else
		sphtor2scal_gpu<float>(shtns, (cplx_f*) d_Slm, (cplx_f*) d_Tlm, (cplx_f*) d_vwlm, (cplx_f*) (d_vwlm+sizeof(float)*nlm_stride), llim, mmax);

	// SHT on the GPU
	if (Vt) {
		if (sizeof_real != 4)
			cuda_SH_to_spat<1>(shtns, (cplx*) d_vwlm, (double*)d_vtp, llim+1, mmax);
		else
			cuda_SH_to_spat<1,float>(shtns, (cplx_f*) d_vwlm, (float*)d_vtp, llim+1, mmax);
		if (Vp) {
			cudaEventCreateWithFlags(&ev_sht, cudaEventDisableTiming );
			cudaEventRecord(ev_sht, shtns->comp_stream);					// record the end of scalar SH (theta).
		}
	}
	if (Vp) {
		if (sizeof_real != 4)
			cuda_SH_to_spat<1>(shtns, (cplx*) (d_vwlm + sizeof(double)*nlm_stride), ((double*)d_vtp) + spat_stride, llim+1, mmax);
		else
			cuda_SH_to_spat<1,float>(shtns, (cplx_f*) (d_vwlm + sizeof(float)*nlm_stride), ((float*)d_vtp) + spat_stride, llim+1, mmax);
	}
	if (CUDA_ERROR_CHECK) return;

	if (Vt) {	// copy back spatial data (theta)
		if (Vp) {
			cudaStreamWaitEvent(shtns->xfer_stream, ev_sht, 0);					// xfer stream waits for end of scalar SH (theta).
			cudaMemcpyAsync(Vt, d_vtp, nspat*sizeof_real, cudaMemcpyDeviceToHost, shtns->xfer_stream);
			cudaEventDestroy(ev_sht);
		} else {
			err = cudaMemcpy(Vt, d_vtp, nspat*sizeof_real, cudaMemcpyDeviceToHost);
		}
	}
	if (Vp) {	// copy back spatial data (phi)
		err = cudaMemcpy(Vp, d_vtp + sizeof_real*spat_stride, nspat*sizeof_real, cudaMemcpyDeviceToHost);
	}
	if (err != cudaSuccess) CUDA_ERROR_CHECK;

	if (shtns->sizeof_real == 4) {	// convert float to double in-place
		for (int i=shtns->nspat-1; i>=0; i--) 	{	Vt[i] = ((float*)Vt)[i];	Vp[i] = ((float*)Vp)[i];  }
	}
}

extern "C"
void SHsph_to_spat_gpu(shtns_cfg shtns, cplx *Slm, double *Vt, double *Vp, const long int llim)
{
	SHsphtor_to_spat_gpu(shtns, Slm, 0, Vt,Vp, llim);
}

extern "C"
void SHtor_to_spat_gpu(shtns_cfg shtns, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	SHsphtor_to_spat_gpu(shtns, 0, Tlm, Vt,Vp, llim);
}

extern "C"
void SHqst_to_spat_gpu(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	if (shtns->sizeof_real == 4) {	// for testing purposes
		SH_to_spat_gpu(shtns, Qlm, Vr, llim);
		SHsphtor_to_spat_gpu(shtns, Slm,Tlm, Vt,Vp, llim);
		return;
	}

	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_sht1, ev_up;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride * howmany;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;
	long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;

	double* d_qvwlm = shtns->gpu_buf_in;
	double* d_vrtp = shtns->gpu_staging_mem;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (howmany == 1) nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	if (howmany > 1  &&  2*nlm_pad > nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }

	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm + nlm_stride, Qlm, 2*nlm_pad*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_SH_to_spat<0>(shtns, (cplx*) (d_qvwlm+nlm_stride), d_vrtp + 2*spat_stride, llim, mmax);		// may use gpu_buf_in = d_qvwlm internally

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm_pad*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride, Tlm, 2*nlm_pad*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1>(shtns, (cplx*) d_qvwlm, d_vrtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht1, cudaEventDisableTiming );
	cudaEventRecord(ev_sht1, comp_stream);					// record the end of scalar SH (theta).

	cuda_SH_to_spat<1>(shtns, (cplx*) (d_qvwlm + nlm_stride), d_vrtp + spat_stride, llim+1, mmax);

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht0, 0);					// xfer stream waits for end of scalar SH (radial).
	cudaMemcpyAsync(Vr, d_vrtp + 2*spat_stride, nspat * sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventDestroy(ev_sht0);

	cudaStreamWaitEvent(xfer_stream, ev_sht1, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(Vt, d_vrtp, nspat * sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventDestroy(ev_sht1);

	// copy back the last transform (compute stream).
	err = cudaMemcpy(Vp, d_vrtp + spat_stride, nspat * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
}

extern "C"
void spat_to_SH_gpu(shtns_cfg shtns, double *Vr, cplx *Qlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	double *d_q   = shtns->gpu_staging_mem;
	double *d_qlm = d_q;		// "in-place" operation possible

	if (shtns->howmany > 1  &&  2*shtns->spec_dist > shtns->nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }

	// copy spatial data to GPU
	if (shtns->sizeof_real == 4) {		// convert double to float
		float* tmp_f = (float*) malloc(sizeof(float)*shtns->nspat);
		for (int i=0; i<shtns->nspat; i++) tmp_f[i] = Vr[i];
		err = cudaMemcpyAsync(d_q, tmp_f, shtns->nspat * sizeof(float), cudaMemcpyHostToDevice, shtns->comp_stream);
		free(tmp_f);
	} else
	err = cudaMemcpyAsync(d_q, Vr, shtns->nspat * sizeof(double), cudaMemcpyHostToDevice, shtns->comp_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	if (shtns->sizeof_real == 4) {
		cu_spat_to_SH_float(shtns, (float*) d_q, (cplx_f*) d_qlm, llim);
	} else
	cu_spat_to_SH(shtns, d_q, (cplx*) d_qlm, llim);
	CUDA_ERROR_CHECK;

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	long nlm_pad = (shtns->howmany==1) ? shtns->nlm : shtns->spec_dist*shtns->howmany;
	if ((llim < mmax*mres) && (shtns->howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Qlm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}
	// copy back spectral data
	err = cudaMemcpy(Qlm, d_qlm, 2*nlm_pad * shtns->sizeof_real, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	if (shtns->sizeof_real == 4) {	// convert float to double in-place
		for (int i=2*nlm_pad-1; i>=0; i--)	((double*)Qlm)[i] = ((float*)Qlm)[i];
	}
}


extern "C"
void spat_to_SHsphtor_gpu(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up;
	const long nspat = shtns->nspat;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	const long nlm_stride = shtns->nlm_stride * howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	if (howmany > 1  &&  2*shtns->spec_dist > shtns->nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }
	
	if (shtns->sizeof_real == 4) {	// fp32, for testing purposes
		float* d_vtp = (float*) shtns->gpu_staging_mem;
		copy_convert_field_to_gpu(d_vtp, Vt, nspat, sizeof(float), shtns->comp_stream);
		copy_convert_field_to_gpu(d_vtp+spat_stride, Vp, nspat, sizeof(float), shtns->comp_stream);
		cu_spat_to_SHsphtor_float(shtns, d_vtp, d_vtp + spat_stride, (cplx_f*) d_vtp, (cplx_f*)(d_vtp+nlm_stride), llim);
		long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;
		err = cudaMemcpy(Slm, d_vtp, 2*nlm_pad*sizeof(float), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(Tlm, d_vtp+nlm_stride, 2*nlm_pad*sizeof(float), cudaMemcpyDeviceToHost);
		for (int i=2*nlm_pad-1; i>=0; i--) {	((double*)Slm)[i] = ((float*)Slm)[i];	((double*)Tlm)[i] = ((float*)Tlm)[i];	}
		return;
	}

	double* d_vwlm = shtns->gpu_buf_in;
	double* d_vtp = shtns->gpu_staging_mem;

	// copy spatial data to gpu
	err = cudaMemcpy(d_vtp, Vt, nspat*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_spat_to_SH<1>(shtns, d_vtp, (cplx*) d_vwlm, llim+1);

	err = cudaMemcpyAsync(d_vtp + spat_stride, Vp, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(shtns->comp_stream, ev_up, 0);					// compute stream waits for end of data transfer (phi).
	cuda_spat_to_SH<1>(shtns, d_vtp + spat_stride, (cplx*) (d_vwlm + nlm_stride), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), llim);

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;
	if ((llim < mmax*mres) && (howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Slm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Tlm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}

	err = cudaMemcpy(Slm, d_vtp, 2*nlm_pad*sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(Tlm, d_vtp+nlm_stride, 2*nlm_pad*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
}

extern "C"
void spat_to_SHqst_gpu(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, const long int llim)
{
	if (shtns->sizeof_real == 4) {	// fp32, for testing purposes
		spat_to_SH_gpu(shtns, Vr, Qlm, llim);
		spat_to_SHsphtor_gpu(shtns, Vt,Vp, Slm,Tlm, llim);
		return;
	}
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up, ev_up2, ev_sh2;
	const long nspat = shtns->nspat;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	const long nlm_stride = shtns->nlm_stride * howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm;
	double* d_vrtp;

	d_qvwlm = shtns->gpu_buf_in;
	d_vrtp = shtns->gpu_staging_mem;	//d_qvwlm + 2*nlm_stride;

	if (howmany > 1  &&  2*shtns->spec_dist > shtns->nlm_stride) { printf("ERROR: distance between field too large, unsupported\n."); return; }

	// copy spatial data to gpu
	err = cudaMemcpy(d_vrtp, Vt, nspat*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_spat_to_SH<1>(shtns, d_vrtp, (cplx*) d_qvwlm, llim+1);

	err = cudaMemcpyAsync(d_vrtp + spat_stride, Vp, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(comp_stream, ev_up, 0);			// compute stream waits for end of data transfer (phi).
	cuda_spat_to_SH<1>(shtns, d_vrtp + spat_stride, (cplx*) (d_qvwlm + nlm_stride), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), llim);
	cudaEventCreateWithFlags(&ev_sh2, cudaEventDisableTiming );
	cudaEventRecord(ev_sh2, comp_stream);				// record the end of vector transform.

	err = cudaMemcpyAsync(d_vrtp + 2*spat_stride, Vr, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up2, cudaEventDisableTiming );
	cudaEventRecord(ev_up2, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(comp_stream, ev_up2, 0);		// compute stream waits for end of data transfer (phi).
	// scalar SHT on the GPU
	cuda_spat_to_SH<0>(shtns, d_vrtp + 2*spat_stride, (cplx*) (d_qvwlm+nlm_stride), llim);	// uses gpu_buf_in == d_qvwlm internally

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	long nlm_pad = (howmany==1) ? shtns->nlm : shtns->spec_dist*howmany;
	if ((llim < mmax*mres) && (howmany ==1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm_pad = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Slm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Tlm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Qlm+nlm_pad, 0, 2*(shtns->nlm - nlm_pad)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}

	cudaStreamWaitEvent(xfer_stream, ev_sh2, 0);					// xfer stream waits for end of vector sht.
	err = cudaMemcpyAsync(Slm, d_vrtp, 2*nlm_pad*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	err = cudaMemcpyAsync(Tlm, d_vrtp+nlm_stride, 2*nlm_pad*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);

	err = cudaMemcpy(Qlm, d_qvwlm+nlm_stride, 2*nlm_pad*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);	cudaEventDestroy(ev_up2);	cudaEventDestroy(ev_sh2);
}

void* fgpu[4][SHT_NTYP] = {
	{ (void*) SH_to_spat_gpu, (void*) spat_to_SH_gpu, (void*) SHsphtor_to_spat_gpu, (void*) spat_to_SHsphtor_gpu, (void*) SHsph_to_spat_gpu, (void*) SHtor_to_spat_gpu, (void*) SHqst_to_spat_gpu, (void*) spat_to_SHqst_gpu },
	{0}, //{ 0, 0, (void*) SHsphtor_to_spat_gpu2, (void*) spat_to_SHsphtor_gpu2, 0, 0, (void*) SHqst_to_spat_gpu2, (void*) spat_to_SHqst_gpu2 },
	{0}, // former "hostfft" transforms
	{0}, // former "hostfft" transforms
};
