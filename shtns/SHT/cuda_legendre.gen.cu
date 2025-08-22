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

// CUDA kernels for SHTns that require run-time compiling using nvrtc
// at compilation, the following must be defined
// WARPSZE : the warpsize (32 for nvidia, 64 for amd)
// LMAX : max degree of spherical harmonics. used for address calculation
// MRES : order multiplicity (often 1). used for address calculation.
// HI_LLIM : 0 for small values of lmax, 1 for large (rescaling needed)
// M0_ONLY : 0 for all m's, 1 for axisymmetric
// ROBERT_FORM : 0 for regular transform, 1 for the Robert form transform (applies to S=1 only)
// BLKSZE_S : blocksize for synthesis (leg_m_kernel)
// BLKSZE_A : blocksize for analysis (ileg_m_kernel)
// BLKSZE_SH2ISH : 0 for separate ishioka pre-computation, or blocksize for in-kernel pre-computation -- for scalar (S=0) synthesis only.
// NF_S : number of fields treated together for synthesis
// NF_A : number of fields treated together for analysis
// LSPAN_A : number of SH degrees treated together (analysis)
// NW_S : number of spatial points per thread (synthesis)
// MPOS_SCALE : scale factor for analysis (used once)
// NLAT_2 : half the number of latidunal points, should be equal to the nlat_2 kernel parameter

// TODO: some parameters can be made compile-time constants!
// 		nlat_2, nphi, m_inc, mpos_scale

// define our own suffle macros, to accomodate cuda<9 and cuda>=9
#if __CUDACC_VER_MAJOR__ < 9
	#define shfl_xor(...) __shfl_xor(__VA_ARGS__)
	#define shfl_down(...) __shfl_down(__VA_ARGS__)
	#define shfl(...) __shfl(__VA_ARGS__)
	#define _any(p) __any(p)
	#define _all(p) __all(p)
	#define _ballot(p) __ballot(p)
	#define _syncwarp 0
	#define _syncwarp_fence __threadfence_block()
#else
	#define shfl_xor(...) __shfl_xor_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl_down(...) __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl(...) __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
	#define _any(p) __any_sync(0xFFFFFFFF, p)
	#define _all(p) __all_sync(0xFFFFFFFF, p)
	#define _ballot(p) __ballot_sync(0xFFFFFFFF, p)
	#define _syncwarp __syncwarp()
	#define _syncwarp_fence __syncwarp()
#endif

#if defined(__gfx908__) || defined(__gfx90a__)
// better shfl_xor operating on 32bit registers only
template <unsigned XOR_MASK>
inline __device__ int shfl_xor_b32(int v)
{
	if (XOR_MASK==0) return v;
	else if (XOR_MASK<4) {
		return __builtin_amdgcn_mov_dpp(v, (0^XOR_MASK) | ((1^XOR_MASK)<<2) | ((2^XOR_MASK)<<4) | ((3^XOR_MASK)<<6),
			0xF, 0xF, 1);
	} else if (XOR_MASK==0x8) {
		return __builtin_amdgcn_mov_dpp(v, 0x128, 0xF, 0xF, 1);		// row rotate right by 8 threads within row (group of 16)
	} else if (XOR_MASK==0xF) {
		return __builtin_amdgcn_mov_dpp(v, 0x140, 0xF, 0xF, 1);		// reverse within row (group of 16)
	} else if (XOR_MASK==0x7) {
		return __builtin_amdgcn_mov_dpp(v, 0x141, 0xF, 0xF, 1);		// reverse within half-row (group of 8)
	} else if (XOR_MASK<32) {
		// ds_swizzle_b32: xor_mask is encoded into instruction, saves instructions compared to next case
		return __builtin_amdgcn_ds_swizzle(v, (XOR_MASK << 10) | 31);
	} else
		return __builtin_amdgcn_ds_bpermute((threadIdx.x ^ XOR_MASK)*4, v);
	//	return __shfl_xor(v,XOR_MASK);		// emit ds_bpermute_b32, with lots of instructions to compute lanes.
}

// better broadcast operating on 32bit registers only. NGROUP must be a power of 2.
template <unsigned LANE_ID, unsigned NGROUP=64>
inline __device__ int broadcast_b32(int v)
{
	static_assert(LANE_ID < NGROUP, "LANE_ID must be less than NGROUP.");
	if (NGROUP==1) return v;
	else if (NGROUP<=4) {		// NGROUP==2 or 4
		return __builtin_amdgcn_mov_dpp(v, (LANE_ID) | ((LANE_ID)<<2) | ((LANE_ID+4-NGROUP)<<4) | ((LANE_ID+4-NGROUP)<<6),
			0xF, 0xF, 1);
#ifdef __gfx90a__
	} else if (NGROUP==16) {
		return __builtin_amdgcn_mov_dpp(v, 0x150 + LANE_ID, 0xF, 0xF, 1);		// broadcast within row (group of 16), only for MI200
#endif
	} else if (NGROUP<=32) {
		// ds_swizzle_b32: broadcast lane encoded into instruction, saves instructions compared to next case
		return __builtin_amdgcn_ds_swizzle(v, (LANE_ID << 5) | (32-NGROUP));
	} else if (NGROUP==64) {
		//return __builtin_amdgcn_readlane(v, LANE_ID);
		return __builtin_amdgcn_ds_bpermute(LANE_ID*4, v);
	} else
		return __shfl(v,LANE_ID, NGROUP);		// emit ds_bpermute_b32, good for broadcast
}

// better shfl_down operating on 32bit registers only. NGROUP must be a power of 2.
// WARNING: threads that read out of bounds (group) are undefined (NOT like nvidia cuda __shfl_down which specifies that those lanes are unchanged)
template <unsigned NSHIFT, unsigned NGROUP=64>
inline __device__ int shfl_down_b32(int v)
{
	static_assert(NSHIFT < NGROUP, "NSHIFT must be less than NGROUP.");
	if ((NGROUP==1) || (NSHIFT==0)) return v;
	else if (NGROUP<=4) {
		return __builtin_amdgcn_mov_dpp(v, NSHIFT | (((NSHIFT<3) ? 1+NSHIFT : 1) <<2) | (3<<4) | (3<<6),
			0xF, 0xF, 1);
	} else if (NGROUP<=16) {	// shift crosses group boundary for NGROUP==8
		return __builtin_amdgcn_mov_dpp(v, 0x100 | NSHIFT, 0xF, 0xF, 0);
	} else if ((NGROUP<=64) && (NSHIFT==1)) {
		return __builtin_amdgcn_mov_dpp(v, 0x130, 0xF, 0xF, 0);		// shift crosses group boundary for NGROUP<64
	} else if (NGROUP==32) {
		// ds_swizzle_b32 in rotate mode: upper lanes are filled with lower lanes
		return __builtin_amdgcn_ds_swizzle(v, 0xC000 | (NSHIFT << 5));
	} else
		return __builtin_amdgcn_ds_bpermute((threadIdx.x + NSHIFT)*4, v);	// rotate: fill upper lanes with lower ones
	//return __shfl_down(v,NSHIFT,NGROUP);		// emit ds_bpermute_b32, with lots of instructions to compute lanes exactly as cuda __shfl_down() does.
}

template <unsigned XOR_MASK>
inline __device__ double shfl_xor_(double v) {
	union {double d; int i[2];};		// allow access to the 2 words forming the double separately
	d = v;
	i[0] = shfl_xor_b32<XOR_MASK>(i[0]);		// shuffle
	i[1] = shfl_xor_b32<XOR_MASK>(i[1]);		// shuffle
	return d;
}
template <unsigned XOR_MASK>
inline __device__ float shfl_xor_(float v) {
	return __int_as_float( shfl_xor_b32<XOR_MASK>( __float_as_int(v) ) );
}
template <unsigned XOR_MASK>
inline __device__ int shfl_xor_(int v) {
	return shfl_xor_b32<XOR_MASK>(v);
}

template <unsigned NSHIFT, unsigned NGROUP=64>
inline __device__ float shfl_down_(float v) {
	return __int_as_float( shfl_down_b32<NSHIFT, NGROUP>( __float_as_int(v) ) );
}
template <unsigned NSHIFT, unsigned NGROUP=64>
inline __device__ int shfl_down_(int v) {
	return shfl_down_b32<NSHIFT, NGROUP>( v );
}
template <unsigned NSHIFT, unsigned NGROUP=64>
inline __device__ double shfl_down_(double v)
{
	union {double d; int i[2];};		// allow access to the 2 words forming the double separately
	d = v;
	i[0] = shfl_down_b32<NSHIFT, NGROUP>(i[0]);		// shuflle first half
	i[1] = shfl_down_b32<NSHIFT, NGROUP>(i[1]);		// and second half
	return d;
}

template <unsigned LANE_ID, unsigned NGROUP=64>
inline __device__ int bcast_(int v) {
	return broadcast_b32<LANE_ID, NGROUP>(v);
}
template <unsigned LANE_ID, unsigned NGROUP=64, class T>
inline __device__ T bcast_(T v) {
	const int NINT = (sizeof(T)+3)/4;
	union {T d; int i[NINT];};
	d = v;
	for (int k=0; k<NINT; k++)
		i[k] = broadcast_b32<LANE_ID, NGROUP>(i[k]);		// shuflle
	return d;
}

#undef shfl_xor
#define shfl_xor(v,xor_mask) shfl_xor_<xor_mask>(v)
#undef shfl
#define shfl(v,lane,group) bcast_<lane,group>(v)
#undef shfl_down
#define shfl_down(v,shift,group) shfl_down_<shift,group>(v)
#endif

#ifdef __gfx90a__
	#define atomicAdd_sht unsafeAtomicAdd
#else	/* NOT __gfx90a__ */
#if WARPSZE == 64
	// AMD HIP
	#define __forceinline__ inline
#endif

#if (__CUDACC_VER_MAJOR__ < 8) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
__device__ __forceinline__ void atomicAdd_sht(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}
__device__ __forceinline__ void atomicAdd_sht(float* address, float val) {
	atomicAdd(address,val);
}
#else
	#define atomicAdd_sht atomicAdd
#endif
#endif

__device__ __forceinline__  double2 make_real2(double a, double b) {	return make_double2(a,b);	}
__device__ __forceinline__  float2  make_real2(float a,  float b)  {	return make_float2 (a,b);	}

__device__ __forceinline__ bool polar_skip_sint(double sint, int llim, int m)
{
	// polar optimization (see Reinecke 2013, section 3.3)
	int x = sint * llim;
	#if LMAX > 10350
		return (m - max(80, llim>>7) > x);
	#else
		return (m - 80 > x);
	#endif
}
__device__ __forceinline__ bool polar_skip_sint(float sint, int llim, int m) {
	return false;
}

__device__ __forceinline__ bool polar_skip_cost(double cost, int llim, int m)
{
	// polar optimization (see Reinecke 2013, section 3.3) -- squared
	int mm = m - ((LMAX > 10350) ? max(80, llim>>7) : 80);
	return (mm>0) && (mm*mm > (int) ((1.-cost*cost)*(llim*llim)));
}
__device__ __forceinline__ bool polar_skip_cost(float cost, int llim, int m) {
	return false;
}

/// requirements : blockSize must be 1 in the y- and z-direction and BLKSZE_S in the x-direction.
/// llim MUST BE <= 1800, unless HI_LLIM=1
template<int S> __global__
#if WARPSZE == 64
__launch_bounds__((BLKSZE_S<BLKSZE_SH2ISH) ? BLKSZE_SH2ISH : BLKSZE_S, 3)	// leads to better performance for small transforms on AMD MI250
#endif
void leg_m_kernel(
	const real_g* __restrict__ al, const real_g* __restrict__ ct, const real* __restrict__ ql, real *q,
	const int llim, const int nlat_2, const int nphi, const int m_inc,
	const int ql_dist, const int q_dist
#if BLKSZE_SH2ISH > 0
	,const real* __restrict__ xlm
#endif
)

{
	const int BLOCKSIZE = (BLKSZE_SH2ISH>0 && S==0) ? BLKSZE_SH2ISH : BLKSZE_S;
	const int NW=NW_S;
	const int NFIELDS=NF_S;
	const int im = (M0_ONLY) ? 0 : blockIdx.z;
	const int j = threadIdx.x;
	const int b = blockIdx.y;		// position in batch
	//const int m_inc = 2*nlat_2;
  #ifndef LAYOUT_REAL_FFT
	const int k_inc = 1;
  #else
	const int k_inc = 2;
  #endif

	const int LSPAN = (WARPSZE==32 && BLOCKSIZE >= 2*WARPSZE) ? BLOCKSIZE/2 : WARPSZE;		// always WARPSZE for amd
	static_assert(LSPAN <= BLOCKSIZE, "LSPAN must not exceed BLOCKSIZE");
	static_assert(LSPAN % 4 == 0, "LSPAN must be a multiple of 4");
	__shared__ real_g ak[LSPAN];
	__shared__ real qk[NFIELDS][(M0_ONLY) ? LSPAN : LSPAN*2];

	static_assert( (!HI_LLIM) || ( NW==1 || (NW&1)==0 ), "high llim works with NW=1 or NW even" );

	#define COST_CACHE		// optional: store cos(theta) into shared memory to reduce register pressure
	real_g y0[NW];
	real_g y1[NW];
	real_g ct2[NW];
#ifdef LEG_ISHIOKA
	#ifndef COST_CACHE
	real cost_[NW];
	#define COST(i,j) cost_[i]
	#else
	__shared__ real cost_[NW][BLOCKSIZE];
	#define COST(i,j) cost_[i][j]
	#endif
#endif
	#pragma unroll
	for (int i=0; i<NW; i++) {
		const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
		ct2[i] = (it < nlat_2) ? ct[it] : 0;
	}

	if (im==0) {
		if ((LSPAN==BLOCKSIZE || j<LSPAN) && (j<=llim)) {
			ak[j] = al[j+2];
			#if BLKSZE_SH2ISH > 0
			if (S==0) {
			  #ifdef LEG_ISHIOKA
				int xofs = 3*(j>>1);	// load xlm coeffs once for all fields
				real x0 = xlm[xofs+2*(j&1)];
				real x1 = xlm[xofs+1];		// only used for j&2==0
				bool use_x1 = (((j&1)==0) & (j+1 <llim));	// l-m even
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					int qofs = 2*j + (b*NFIELDS+f)*ql_dist;
					real ql_ = ql[qofs] * x0;
					if (use_x1)	ql_ += x1 * ql[qofs + 4];
					qk[f][j] = ql_;
				}
			  #else
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	qk[f][j] = ql[2*j + (b*NFIELDS+f)*ql_dist] * xlm[j];
			  #endif
			} else
			#endif
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	qk[f][j] = ql[j + (b*NFIELDS+f)*ql_dist];		// keep only real part
		}
	#ifdef LEG_ISHIOKA
		#pragma unroll
		for (int i=0; i<NW; i++) {	COST(i,j) = ct2[i];		ct2[i] *= ct2[i];	}	// cos(theta)^2
	#endif

	#if SHT_HI_PREC & 2
		real_g re[NFIELDS][NW], ro[NFIELDS][NW];
	#else
		real re[NFIELDS][NW], ro[NFIELDS][NW];
	#endif
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++) {
				re[f][i] = 0;
				ro[f][i] = 0;
			}
		}

		if (S==1 && !ROBERT_FORM) {		// for vectors, divide by sin(theta) -- except in Robert form
			#pragma unroll
			for (int i=0; i<NW; i++) {
				const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
				y0[i] = (it < nlat_2) ? ct[it+3*nlat_2] : 0;		// 1/sin(theta)
			}
		}
	  #ifdef LEG_ISHIOKA
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = al[1]*ct2[i] + al[0];
	  #else
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = al[1]*ct2[i];
	 #endif
		if (S==1 && !ROBERT_FORM) {
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] *= y0[i];
		} else {
			#pragma unroll
			for (int i=0; i<NW; i++) y0[i] = 1;
		}

		al+=2;
		if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }

	#if SHT_HI_PREC & 1
		__shared__ real mean[NFIELDS];
		if (S==0) {				// TODO: this hack could be made cleaner
			if (j<NFIELDS) {
				mean[j] = y0[0] * qk[j][0];		// mean value may be much larger: we keep it separated for better accuracy (especially in fp32)
				qk[j][0] = 0;						// we are done with the mean ==> set to zero
			}
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
	#endif

		int l = 0;
		while (l<=llim - LSPAN) {	// compute even and odd parts
		  #ifdef LEG_ISHIOKA
			for (int k = 0; k<LSPAN; k+=4) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					real y0g = y0[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						re[f][i] += y0g * qk[f][k];		// real
						ro[f][i] += y0g * qk[f][k+1];		// real
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y0[i] += (ak[k+1]*ct2[i] + ak[k]) * y1[i];
				#pragma unroll
				for (int i=0; i<NW; i++) {
					real y1g = y1[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						re[f][i] += y1g * qk[f][k+2];		// real
						ro[f][i] += y1g * qk[f][k+3];		// real
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y1[i] += (ak[k+3]*ct2[i] + ak[k+2]) * y0[i];
			}
			al += LSPAN;
			l  += LSPAN;
		  #else
			for (int k = 0; k<LSPAN; k+=2) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					real y0g = y0[i];
					real y1g = y1[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						re[f][i] += y0g * qk[f][k];		// real
						ro[f][i] += y1g * qk[f][k+1];	// real
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y0[i] += (ak[k]*ct2[i]) * y1[i];
				#pragma unroll
				for (int i=0; i<NW; i++) y1[i] += (ak[k+1]*ct2[i]) * y0[i];
			}
			al += LSPAN;
			l  += LSPAN;
		  #endif
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if ((l+j <= llim) && (BLOCKSIZE==LSPAN || j<LSPAN)) {
				#if BLKSZE_SH2ISH > 0
				if (S==0) {
				  #ifdef LEG_ISHIOKA
					int xofs = 3*((l+j)>>1);	// load xlm coeffs once for all fields
					real x0 = xlm[xofs+2*((l+j)&1)];
					real x1 = xlm[xofs+1];		// only used for j&2==0
					bool use_x1 = ((((l+j)&1)==0) & (l+j+1 <llim));	// l-m even
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						int qofs = 2*(j+l) + (b*NFIELDS+f)*ql_dist;
						real ql_ = ql[qofs] * x0;
						if (use_x1)	ql_ += x1 * ql[qofs + 4];
						qk[f][j] = ql_;
					}
				  #else
					#pragma unroll
					for (int f=0; f<NFIELDS; f++)	qk[f][j] = ql[2*(j+l) + (b*NFIELDS+f)*ql_dist] * xlm[l+j];
				  #endif
				} else
				#endif
					#pragma unroll
					for (int f=0; f<NFIELDS; f++)	qk[f][j] = ql[l+j + (b*NFIELDS+f)*ql_dist];		// keep only real part

				ak[j] = al[j];
			}
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
		int k=0;
		while (l<llim) {	// compute even and odd parts
		  #ifdef LEG_ISHIOKA
			#pragma unroll
			for (int i=0; i<NW; i++) {
				real y0g = y0[i];
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					re[f][i] += y0g * qk[f][k];	// real
					ro[f][i] += y0g * qk[f][k+1];	// real
				}
			}
			#pragma unroll
			for (int i=0; i<NW; i++) {
				real_g tmp = (ak[k+1]*ct2[i] + ak[k]) * y1[i] + y0[i];
				y0[i] = y1[i];
				y1[i] = tmp;
			}
		  #else
			#pragma unroll
			for (int i=0; i<NW; i++) {
				real y0g = y0[i];
				real y1g = y1[i];
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					re[f][i] += y0g * qk[f][k];	// real
					ro[f][i] += y1g * qk[f][k+1];	// real
				}
			}
			#pragma unroll
			for (int i=0; i<NW; i++) y0[i] += (ak[k]*ct2[i]) * y1[i];
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] += (ak[k+1]*ct2[i]) * y0[i];
		  #endif
			l+=2;	k+=2;
		}
		if (l==llim) {
			#pragma unroll
			for (int i=0; i<NW; i++) {
				real y0g = y0[i];
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					re[f][i] += y0g * qk[f][k];		// real
				}
			}
		}

		#pragma unroll
		for (int i=0; i<NW; i++) {
			const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
			if (it < nlat_2) {
				// store mangled for complex fft
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
				  #ifdef LEG_ISHIOKA
					real north = re[f][i]+ro[f][i]*COST(i,j);
					real south = re[f][i]-ro[f][i]*COST(i,j);
				  #else
					real north = re[f][i]+ro[f][i];
					real south = re[f][i]-ro[f][i];
				  #endif
				  #if SHT_HI_PREC & 1
					if (S==0)	{	north += mean[f];	south += mean[f];	}		// mean added at the very end for improved accuracy when mean >> std
				  #endif
				  #ifndef LAYOUT_REAL_FFT
					q[it*k_inc              + (b*NFIELDS+f)*q_dist] = north;
					q[(nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = south;
				  #else
					*((real2*)(q+it*k_inc           + (b*NFIELDS+f)*q_dist)) = make_real2(north, (real)0);
					*((real2*)(q+(m_inc-1-it)*k_inc + (b*NFIELDS+f)*q_dist)) = make_real2(south, (real)0);
				  #endif
				}
			}
		}
	}
#if M0_ONLY==0
	else { 	// m>0
		real rer[NFIELDS][NW], ror[NFIELDS][NW], rei[NFIELDS][NW], roi[NFIELDS][NW];
		const int m = im*MRES;
		int l = (im*(2*(LMAX+1)-MRES-m))>>1;
		#if BLKSZE_SH2ISH > 0
			#ifdef LEG_ISHIOKA
			if (S==0)	xlm += 3*im*(2*(LMAX+4)+MRES-m)/4;
			#else
			if (S==0)	xlm += im*(LMAX+3) - (m*(im-1))/2;
			#endif
		#endif
		#ifdef LEG_ISHIOKA
		al += l+m;
		#else
		al += im*(LMAX+3) - (m*(im-1))/2;
		#endif
		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1

		if ((LSPAN==BLOCKSIZE || j<LSPAN) && (m+j<=llim)) 	ak[j] = al[j+2];
			if ((m+j/2 <= llim) && (2*LSPAN>=BLOCKSIZE || j<2*LSPAN)) {
				#if BLKSZE_SH2ISH > 0
				real x0, x1;
				bool use_x1;
				if (S==0) {
				  #ifdef LEG_ISHIOKA
					int xofs = 3*(j>>2);	// load xlm coeffs once for all fields
					x0 = xlm[xofs+(j&2)];
					x1 = xlm[xofs+1];		// only used for j&2==0
					use_x1 = (((j&2)==0) & (j+2 <2*(llim-m)));	// l-m even
				  #else
					x0 = xlm[j>>1];
				  #endif
				}
				#endif
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					int qofs = 2*m+j + (b*NFIELDS+f)*ql_dist;
					real ql_ = ql[qofs];
					#if BLKSZE_SH2ISH > 0
					if (S==0) {		ql_ *= x0;
					  #ifdef LEG_ISHIOKA
						if (use_x1)	ql_ += x1 * ql[qofs + 4];
					  #endif
					}
					#endif
					qk[f][j] = ql_;
				}
			}
			if ((BLOCKSIZE < 2*LSPAN) && (m+j/2+BLOCKSIZE/2 <= llim) && (2*BLOCKSIZE<=2*LSPAN || j+BLOCKSIZE < 2*LSPAN)) {
				#if BLKSZE_SH2ISH > 0
				real x0, x1;
				bool use_x1;
				if (S==0) {
				  #ifdef LEG_ISHIOKA
					int xofs = 3*(j>>2) + 3*BLOCKSIZE/4;	// load xlm coeffs once for all fields
					x0 = xlm[xofs+(j&2)];
					x1 = xlm[xofs+1];		// only used for j&2==0
					use_x1 = (((j&2)==0) & (j+BLOCKSIZE+2 <2*(llim-m)));	// l-m even
				  #else
					x0 = xlm[(j>>1) + BLOCKSIZE/2];
				  #endif
				}
				#endif
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					int qofs = 2*m+j+BLOCKSIZE + (b*NFIELDS+f)*ql_dist;
					real ql_ = ql[qofs];
					#if BLKSZE_SH2ISH > 0
					if (S==0) {		ql_ *= x0;
					  #ifdef LEG_ISHIOKA
						if (use_x1)	ql_ += x1 * ql[qofs + 4];
					  #endif
					}
					#endif
					qk[f][j+BLOCKSIZE] = ql_;
				}
			}

		#pragma unroll
		for (int i=0; i<NW; i++) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				ror[f][i] = 0;		roi[f][i] = 0;
				rer[f][i] = 0;		rei[f][i] = 0;
			}
		}

		bool skip_block = false;
		if (NLAT_2 > BLOCKSIZE*NW) {	// polar optimization
			if (j == BLOCKSIZE-1)	skip_block = polar_skip_cost(ct2[NW-1], llim, m);
			#if WARPSZE==32
			if (BLOCKSIZE == WARPSZE) skip_block = _any(skip_block);	// get largest value in block/warp
			#else
			if (BLOCKSIZE == WARPSZE) skip_block = shfl(skip_block,WARPSZE-1,WARPSZE);	// get largest value in block/warp
			#endif
			else {
				__shared__ volatile int xx;
				if (j == BLOCKSIZE-1) xx = skip_block;	// one thread writes its value (the largest one)
				__syncthreads();
				skip_block = xx;	// everyone reads that value
			}
		} else if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		// at this point, block is in sync (consistent view of shared memory).
	if (!skip_block) {

		#ifdef LEG_ISHIOKA
			#pragma unroll
			for (int i=0; i<NW; i++) {	COST(i,j) = ct2[i];		ct2[i] *= ct2[i];	}	// cos(theta)^2
		#endif
		#pragma unroll
		for (int i=0; i<NW; i++) 	y0[i] = 1;

		#if HI_LLIM==1
		int ny = 0;		// only used for HI_LLIM
		#else
		constexpr int ny = 0;
		#endif
		{	// compute sin(theta)^(m-S)
			l = (S==1 && ROBERT_FORM) ? m : m-S;		// multiply vectors by sin(theta) with robert_form
			if (S==0 || l!=0) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
					y1[i] = (it < nlat_2) ? ct[it + 2*nlat_2] : 0;		// sin(theta)
				}
				if (l&1) {
					#pragma unroll
					for (int i=0; i<NW; i++) y0[i] = y1[i];
				}
				#if HI_LLIM==1
				int nsint = 0;
				#endif
				while( l >>= 1 ) {
					#pragma unroll
					for (int i=0; i<NW; i++) y1[i] *= y1[i];
					#if HI_LLIM==1
						nsint += nsint;
						if (y1[NW-1] < 1/SHT_SCALE_FACTOR) {
							nsint--;
							#pragma unroll
							for (int i=0; i<NW; i++) y1[i] *= SHT_SCALE_FACTOR;
						}
					#endif
					if (l&1) {
						#pragma unroll
						for (int i=0; i<NW; i++) y0[i] *= y1[i];
						#if HI_LLIM==1
							ny += nsint;
							if (y0[NW-1] < (SHT_ACCURACY+1/SHT_SCALE_FACTOR)) {
								#pragma unroll
								for (int i=0; i<NW; i++) y0[i] *= SHT_SCALE_FACTOR;
								ny--;
							}
						#endif
					}
				}
			}
		}

	#ifdef LEG_ISHIOKA
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i] + al[0])*y0[i];
	#else
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i])*y0[i];
	#endif

		l=m;		al+=2;
		while (l<=llim - LSPAN) {	// compute even and odd parts
		  #ifdef LEG_ISHIOKA
			for (int k = 0; k<LSPAN; k+=4) {
				real_g tmp[NW];
				#pragma unroll
				for (int i=0; i<NW; i++)	tmp[i] = ak[k+1]*ct2[i] + ak[k];
				if ((!HI_LLIM) || (ny==0)) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						real y0g = y0[i];
						#pragma unroll
						for (int f=0; f<NFIELDS; f++) {
							rer[f][i] += y0g * qk[f][2*k];	// real
							rei[f][i] += y0g * qk[f][2*k+1];	// imag
							ror[f][i] += y0g * qk[f][2*k+2];	// real
							roi[f][i] += y0g * qk[f][2*k+3];	// imag
						}
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++)	y0[i] += tmp[i] * y1[i];
				#pragma unroll
				for (int i=0; i<NW; i++)	tmp[i] = ak[k+3]*ct2[i] + ak[k+2];
				if ((!HI_LLIM) || (ny==0)) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						real y1g = y1[i];
						#pragma unroll
						for (int f=0; f<NFIELDS; f++) {
							rer[f][i] += y1g * qk[f][2*k+4];	// real
							rei[f][i] += y1g * qk[f][2*k+5];	// imag
							ror[f][i] += y1g * qk[f][2*k+6];	// real
							roi[f][i] += y1g * qk[f][2*k+7];	// imag
						}
					}
				}
				#if HI_LLIM==1
				else if (fabs(y0[NW-1]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1)
				{	// rescale when value is significant
					++ny;
					#pragma unroll
					for (int i=0; i<NW; i++) {
						y0[i] *= 1/SHT_SCALE_FACTOR;
						y1[i] *= 1/SHT_SCALE_FACTOR;
					}
				}
				#endif
				#pragma unroll
				for (int i=0; i<NW; i++)	y1[i] += tmp[i] * y0[i];
			}
		  #else
			for (int k = 0; k<LSPAN; k+=2) {
				if ((!HI_LLIM) || (ny==0)) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						real y0g = y0[i];
						#pragma unroll
						for (int f=0; f<NFIELDS; f++) {
							rer[f][i] += y0g * qk[f][2*k];	// real
							rei[f][i] += y0g * qk[f][2*k+1];	// imag
						}
						y0g = y1[i];
						#pragma unroll
						for (int f=0; f<NFIELDS; f++) {
							ror[f][i] += y0g * qk[f][2*k+2];	// real
							roi[f][i] += y0g * qk[f][2*k+3];	// imag
						}
					}
				}
				#if HI_LLIM==1
				else if (fabs(y0[NW-1]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1)
				{	// rescale when value is significant
					++ny;
					#pragma unroll
					for (int i=0; i<NW; i++) {
						y0[i] *= 1/SHT_SCALE_FACTOR;
						y1[i] *= 1/SHT_SCALE_FACTOR;
					}
				}
				#endif
				#pragma unroll
				for (int i=0; i<NW; i++) y0[i] += (ak[k]*ct2[i]) * y1[i];
				#pragma unroll
				for (int i=0; i<NW; i++) y1[i] += (ak[k+1]*ct2[i]) * y0[i];
			}
		  #endif
			al += LSPAN;
			l  += LSPAN;
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if ((l+j/2 <= llim) && (BLOCKSIZE<=2*LSPAN || j<2*LSPAN)) {
				#if BLKSZE_SH2ISH > 0
				real x0, x1;
				bool use_x1;
				if (S==0) {
				  #ifdef LEG_ISHIOKA
					int ll = 2*(l-m)+j;
					int xofs = 3*(ll>>2);	// load xlm coeffs once for all fields
					x0 = xlm[xofs+(ll&2)];
					x1 = xlm[xofs+1];		// only used for j&2==0
					use_x1 = (((ll&2)==0) & (ll+2 <2*(llim-m)));	// l-m even
				  #else
					x0 = xlm[(l-m)+(j>>1)];
				  #endif
				}
				#endif
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					int qofs = 2*l+j + (b*NFIELDS+f)*ql_dist;
					real ql_ = ql[qofs];
					#if BLKSZE_SH2ISH > 0
					if (S==0) {		ql_ *= x0;
					  #ifdef LEG_ISHIOKA
						if (use_x1)	ql_ += x1 * ql[qofs + 4];
					  #endif
					}
					#endif
					qk[f][j] = ql_;
				}
			}
			if ((BLOCKSIZE < 2*LSPAN) && (l+j/2+BLOCKSIZE/2 <= llim) && (2*BLOCKSIZE<=2*LSPAN || j+BLOCKSIZE < 2*LSPAN)) {
				#if BLKSZE_SH2ISH > 0
				real x0, x1;
				bool use_x1;
				if (S==0) {
				  #ifdef LEG_ISHIOKA
					int ll = 2*(l-m)+j+BLOCKSIZE;
					int xofs = 3*(ll>>2);	// load xlm coeffs once for all fields
					x0 = xlm[xofs+(ll&2)];
					x1 = xlm[xofs+1];		// only used for j&2==0
					use_x1 = (((ll&2)==0) & (ll+2 <2*(llim-m)));	// l-m even
				  #else
					x0 = xlm[(l-m)+(j>>1)+BLOCKSIZE/2];
				  #endif
				}
				#endif
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					int qofs = 2*l+j+BLOCKSIZE + (b*NFIELDS+f)*ql_dist;
					real ql_ = ql[qofs];
					#if BLKSZE_SH2ISH > 0
					if (S==0) {		ql_ *= x0;
					  #ifdef LEG_ISHIOKA
						if (use_x1)	ql_ += x1 * ql[qofs + 4];
					  #endif
					}
					#endif
					qk[f][BLOCKSIZE+j] = ql_;
				}
			}
			if ((l+j <= llim) && (LSPAN==BLOCKSIZE || j<LSPAN))	 ak[j] = al[j];
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
		int k=0;
		while (l<llim) {	// compute even and odd parts
		  #ifdef LEG_ISHIOKA
			real_g tmp[NW];
			#pragma unroll
			for (int i=0; i<NW; i++)	tmp[i] = ak[k+1]*ct2[i] + ak[k];
		  #endif
			if ((!HI_LLIM) || (ny==0)) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
				  #ifdef LEG_ISHIOKA
					real y0g = y0[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						rer[f][i] += y0g * qk[f][2*k];	// real
						rei[f][i] += y0g * qk[f][2*k+1];	// imag
						ror[f][i] += y0g * qk[f][2*k+2];	// real
						roi[f][i] += y0g * qk[f][2*k+3];	// imag
					}
				  #else
					real y0g = y0[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						rer[f][i] += y0g * qk[f][2*k];	// real
						rei[f][i] += y0g * qk[f][2*k+1];	// imag
					}
					y0g = y1[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						ror[f][i] += y0g * qk[f][2*k+2];	// real
						roi[f][i] += y0g * qk[f][2*k+3];	// imag
					}
				  #endif
				}
			}
			#if HI_LLIM==1
			else if (fabs(y1[NW-1]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1)
			{	// rescale when value is significant
				++ny;
				#pragma unroll
				for (int i=0; i<NW; i++) {
					y0[i] *= 1/SHT_SCALE_FACTOR;
					y1[i] *= 1/SHT_SCALE_FACTOR;
				}
			}
			#endif
		  #ifdef LEG_ISHIOKA
			#pragma unroll
			for (int i=0; i<NW; i++) tmp[i] = tmp[i] * y1[i] + y0[i];
			#pragma unroll
			for (int i=0; i<NW; i++) y0[i] = y1[i];
			l+=2;	k+=2;
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] = tmp[i];
		  #else
			#pragma unroll
			for (int i=0; i<NW; i++) y0[i] += (ak[k]*ct2[i]) * y1[i];
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] += (ak[k+1]*ct2[i]) * y0[i];
			l+=2;	k+=2;
		  #endif
		}
		if (l==llim) {
			if ((!HI_LLIM) || (ny==0)) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					real y0g = y0[i];
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						rer[f][i] += y0g * qk[f][2*k];	// real
						rei[f][i] += y0g * qk[f][2*k+1];	// imag
					}
				}
			}
		}

		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++) {
			  #ifdef LEG_ISHIOKA
				real t    = rer[f][i]+ror[f][i]*COST(i,j);
				rer[f][i] = rer[f][i]-ror[f][i]*COST(i,j);
				ror[f][i] = rei[f][i]-roi[f][i]*COST(i,j);
				rei[f][i] = rei[f][i]+roi[f][i]*COST(i,j);
			  #else
				real t    = rer[f][i]+ror[f][i];
				rer[f][i] = rer[f][i]-ror[f][i];	// south, real
				ror[f][i] = rei[f][i]-roi[f][i];	// south, imag
				rei[f][i] = rei[f][i]+roi[f][i];	// north, imag
			  #endif
				roi[f][i] = t;			// north, real
			}
		}

	  #ifndef LAYOUT_REAL_FFT
		/// store mangled for complex fft
		if ((!HI_LLIM) || (NW==1)) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					ror[f][i] = shfl_xor(ror[f][i], 1);
					rei[f][i] = shfl_xor(rei[f][i], 1);
				}
			}
		}
	  #endif
	}

	#ifndef LAYOUT_REAL_FFT
	  #if HI_LLIM==1 && NW_S==2
		const int ofs_m1 = im*m_inc;
		const int ofs_m2 = (nphi-im)*m_inc;
		#pragma unroll
		for (int i=0; i<NW; i+=2) {
			const int it = BLOCKSIZE*NW * blockIdx.x + NW*j+i;
			if (it < nlat_2) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					*((real2*)(q+ofs_m1 + it*k_inc              + (b*NFIELDS+f)*q_dist)) = make_real2(roi[f][i]-rei[f][i+1], roi[f][i+1]+rei[f][i]);
					*((real2*)(q+ofs_m2 + it*k_inc              + (b*NFIELDS+f)*q_dist)) = make_real2(roi[f][i]+rei[f][i+1], roi[f][i+1]-rei[f][i]);
					*((real2*)(q+ofs_m1 + (nlat_2*2-2-it)*k_inc + (b*NFIELDS+f)*q_dist)) = make_real2(rer[f][i+1]-ror[f][i], rer[f][i]+ror[f][i+1]);
					*((real2*)(q+ofs_m2 + (nlat_2*2-2-it)*k_inc + (b*NFIELDS+f)*q_dist)) = make_real2(rer[f][i+1]+ror[f][i], rer[f][i]-ror[f][i+1]);
				}
			}
		}
	  #else
		#pragma unroll
		for (int i=0; i<NW; i++) {
			const real sgn = (HI_LLIM && NW>1) ? (i^1)-i : (j^1)-j; 	//(it^1) - it;	// 1 - 2*(j&1);		// 1 for even j, -1 for odd j.
			const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
			const int i2 = (HI_LLIM && NW>1) ? i^1 : i;
			if (it < nlat_2) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					q[im*m_inc        + it*k_inc              + (b*NFIELDS+f)*q_dist] = roi[f][i] - rei[f][i2]*sgn;
					q[(nphi-im)*m_inc + it*k_inc              + (b*NFIELDS+f)*q_dist] = roi[f][i] + rei[f][i2]*sgn;
					q[im*m_inc        + (nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = rer[f][i] + ror[f][i2]*sgn;
					q[(nphi-im)*m_inc + (nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = rer[f][i] - ror[f][i2]*sgn;
				}
			}
		}
	  #endif
	#else
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			long ofs = (b*NFIELDS+f)*q_dist + im*2*m_inc;
			#pragma unroll
			for (int i=0; i<NW; i++) {
				const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
				if (it < nlat_2) {
					*((real2*)(q+ofs + it*k_inc)) = 			make_real2(roi[f][i], rei[f][i]);	// north, real+imag
					*((real2*)(q+ofs + (m_inc-it-1)*k_inc)) =	make_real2(rer[f][i], ror[f][i]);	// south, real+imag
				}
			}
		}
	#endif
	}
#endif
}


#if LMAX >= 1000
	#undef HI_LLIM
	#define HI_LLIM 1
#endif

template<int S> __global__
__launch_bounds__(BLKSZE_A,1)
void ileg_m_kernel(const real_g* __restrict__ al, const real_g* __restrict__ ct, const real* __restrict__ q, real *ql, const int llim, 
	const int nlat_2, const int nphi, const int m_inc, const int q_dist, const int ql_dist, const real w_norm
#if BLKSZE_SH2ISH > 0
	//, const real* __restrict__ xlm
#endif
)
{
	const int BLOCKSIZE=BLKSZE_A;
	const int NFIELDS=NF_A;
	const int LSPAN=LSPAN_A;
	const int it = BLOCKSIZE * blockIdx.x + threadIdx.x;
	const int j = threadIdx.x;
	const int im = (M0_ONLY) ? 0 : blockIdx.z;
	const int b = blockIdx.y;
	//const int m_inc = 2*nlat_2;
	const int f0 = (NFIELDS==1) ? 0 : j / (BLOCKSIZE/NFIELDS);			// assign each thread a field f0

	static_assert((BLOCKSIZE % (((M0_ONLY)?1:2)*LSPAN*NFIELDS)) == 0, "BLOCKSIZE must be a multiple of 2*LSPAN*NFIELDS");
	static_assert( ((WARPSZE >= BLOCKSIZE/LSPAN) ? (WARPSZE % (BLOCKSIZE/LSPAN)) : ((BLOCKSIZE/LSPAN) % WARPSZE)) == 0, "WARPSZE and BLOCKSIZE/LSPAN must be multiples");
	static_assert((LSPAN % 4) == 0, "LSPAN must be a multiple of 4");

	__shared__ real_g ak[WARPSZE];	// cache
  #ifdef ILEG_ISHIOKA
	const int padding = WARPSZE/16;		// padding = 0 is very bad for performance (shared-memory bank conflicts).
	const int NROWS = M0_ONLY ? ( (LSPAN>4*NFIELDS) ? LSPAN/2 : 2*NFIELDS ) : ( (LSPAN>8*NFIELDS) ? LSPAN/2 : 4*NFIELDS );
  #else
	const int padding = 2;		// padding = 0 is very bad for performance (shared-memory bank conflicts).
	const int NROWS = M0_ONLY ? ( (LSPAN>2*NFIELDS) ? LSPAN : 2*NFIELDS ) : ( (LSPAN>4*NFIELDS) ? LSPAN : 4*NFIELDS );
  #endif
	const int l_inc = BLOCKSIZE+padding;
	__shared__ real yl[NROWS*l_inc - padding];		// yl is also used for even/odd computation.

	real_g cost = (it < nlat_2) ? ct[it] : 0;
	real_g y0, y1;

	if (im == 0) {
		const int NW = NFIELDS*LSPAN;
		real my_reo[NW];			// in registers

		q += b*NFIELDS*q_dist;

	#if SHT_HI_PREC & 1
		// HANDLE THE MEAN SEPARATELY. THIS IS ESPECIALLY IMPORTANT IN FP32 TO AVOID ACCURACY ISSUES
		if (S==0) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) my_reo[f] = 0;	// first, we use my_reo to store the mean of each field, as NW >= NFIELDS
			for (int k=j; k<nlat_2; k+=BLOCKSIZE) {
				real w = ct[nlat_2 +k];
			  #ifndef LAYOUT_REAL_FFT
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	my_reo[f] += w * (q[k + f*q_dist]  +  q[nlat_2*2-1 - k + f*q_dist]);
			  #else
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	my_reo[f] += w * (q[k*2 + f*q_dist]  +  q[(m_inc-1-k)*2 + f*q_dist]);
			  #endif
			}
				// reduction of my_reo[f] : sum accross all threads
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)  yl[f*l_inc + j] = my_reo[f];	// store to shared mem
				#pragma unroll
				for (int ofs=BLOCKSIZE/2; ofs>=1; ofs/=2) {	// /!\ BLOCKSIZE must be power of 2 here. TODO: remove this limitation
					if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
					if (j<ofs) {
						#pragma unroll
						for (int f=0; f<NFIELDS; f++) yl[f*l_inc + j] += yl[f*l_inc + j + ofs];
					}
				}
				// TODO: could be optimized once everything is in a warp, can be distributed among NFIELDS.
				if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
				if (it<NFIELDS)  ql[llim+1  + (b*NFIELDS+it)*ql_dist] = yl[it*l_inc];		// store the mean for future assembly, in ishioka2sh_kernel()
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) my_reo[f] = yl[f*l_inc] * w_norm;
		}
	#endif

		if (j < LSPAN+2) ak[j] = al[j];

		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
		  #ifndef LAYOUT_REAL_FFT
			real x0 = (it < nlat_2) ? q[it              + f*q_dist] : 0;	// north
			real x1 = (it < nlat_2) ? q[nlat_2*2-1 - it + f*q_dist] : 0;	// south
		  #else
			real x0 = (it < nlat_2) ? q[it*2             + f*q_dist] : 0;	// north
			real x1 = (it < nlat_2) ? q[(m_inc-1 - it)*2 + f*q_dist] : 0;	// south
		  #endif
		  #if SHT_HI_PREC & 1
			yl[f*2*l_inc +j]     = (x0+x1) - ((S==0) ? my_reo[f] : 0);	// even, subtract mean
		  #else
			yl[f*2*l_inc +j]     = x0+x1;		// even
		  #endif
		  #ifdef ILEG_ISHIOKA
			yl[(f*2+1)*l_inc +j] = (x0-x1)*((real)cost);	// odd
		  #else
			yl[(f*2+1)*l_inc +j] = x0-x1;	// odd
		  #endif
		}
		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }

		y0 = (it < nlat_2) ? ct[it + nlat_2] : 0;		// weights are stored just after ct.
		if (S==1)  y1 = (it < nlat_2) ? ct[it + 3*nlat_2] : 0;		// 1/sin(theta)
	#ifdef ILEG_ISHIOKA
		cost *= cost;	// ct2
	#endif

	#if (WARPSZE == 32)  ||  !defined( ILEG_ISHIOKA )
		const int NACC = 4;		// number of independent accumulators per NFIELD. 4 is good for V100
		// re-assign each thread an l (transposed view)
		const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);
	#else	/* WARPSZE == 64 */
		const int NACC = 2;
		const int ll = j % LSPAN;
		const bool write = (BLOCKSIZE == LSPAN*NFIELDS) ? true : ((j % (BLOCKSIZE/NFIELDS)) < LSPAN);
	#endif

		// transpose reo to my_reo
		#pragma unroll
		for (int k=0; k<NW; k++) {
		  #if (WARPSZE == 32)  ||  !defined( ILEG_ISHIOKA )
			int it = j % (BLOCKSIZE/NW) + k*(BLOCKSIZE/NW);
			my_reo[k] = yl[(2*f0  + (ll&1))*l_inc + it];
		  #else /* WAPRSZE == 64 */
			const int ofs = (2*f0+((ll&1)^(k%NACC)))*l_inc + (((j/(LSPAN))*NW) % BLOCKSIZE);
			my_reo[k] = yl[ofs + (k/NACC)*NACC + j%NACC];		// all lanes have different ordering. Less bank conflicts?
		  #endif
		}


		if (S==1) y0 *= (ROBERT_FORM) ? y1*y1 : y1;
	  #ifdef ILEG_ISHIOKA
		y1 = (ak[1]*cost + ak[0]) * y0;
	  #else
		y1 = (ak[1]*cost) * y0;
	  #endif
		if (WARPSZE < LSPAN+2  &&  j<LSPAN+2-WARPSZE)	ak[WARPSZE+j] = al[WARPSZE+j];		// sometimes a bit more than a warp is needed

		al+=2;
		int l = 0;
		do {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			#ifdef ILEG_ISHIOKA
				#pragma unroll
				for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
					real_g c0 = ak[2*k+3]*cost + ak[2*k+2];
					real_g c1 = ak[2*k+5]*cost + ak[2*k+4];
					yl[k*l_inc +j]     = y0;		// l and l+1
					yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
					y0 += c0 * y1;
					y1 += c1 * y0;
				}
				// re-assign each thread an l (transpose)
			  #if (WARPSZE == 32)  ||  !defined( ILEG_ISHIOKA )
				const int itl = (ll >> 1)*l_inc + j % (BLOCKSIZE/NW);
			  #else
				const int itl = (ll>>1)*l_inc + (((j/LSPAN)*NW) % BLOCKSIZE) + j%NACC;
			  #endif
			#else
				#pragma unroll
				for (int k=0; k<LSPAN; k+=2) {		// compute a block of the matrix, write it in shared mem.
					real_g c0 = ak[k+2]*cost;
					real_g c1 = ak[k+3]*cost;
					yl[k*l_inc +j]     = y0;		// l
					yl[(k+1)*l_inc +j] = y1;		// l+1
					y0 += c0 * y1;
					y1 += c1 * y0;
				}
				// re-assign each thread an l (transpose)
				const int itl = ll*l_inc + j % (BLOCKSIZE/NW);
			#endif

			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }

			real qll[NACC];		// accumulators (2 or 4)

		#if (WARPSZE == 32)  ||  !defined( ILEG_ISHIOKA )

			#pragma unroll
			for (int a=0; a<NACC; a++) 	qll[a] = my_reo[a] * yl[itl + a*(BLOCKSIZE/NW)];	// first element of sum
			const int ql_ofs = (l+ll) + (b*NFIELDS+f0)*ql_dist;		// compute destination ofset in parallel with reduce!
			#pragma unroll
			for (int k=NACC; k<NW; k+=NACC) {
				#pragma unroll
				for (int a=0; a<NACC; a++) 	qll[a] += my_reo[a+k] * yl[itl + (k+a)*(BLOCKSIZE/NW)];
			}

			al += LSPAN;
			if (j<LSPAN) ak[j+2] = al[j];

			if (NACC > 1) {		// reduce the NACC independent accumulators
				#pragma unroll
				for (int a=0; a<NACC; a+=2) {
					qll[a] += qll[a+1];
				}
				for (int a=2; a<NACC; a+=2) {
					qll[0] += qll[a];
				}
			}

			static_assert(BLOCKSIZE/NW <= 16, "Block size must not exceed 16*NW");
			// reduce_add within same l is in same warp too:
				if (BLOCKSIZE/NW > 8) qll[0] += shfl_down(qll[0], 8, 16);
				if (BLOCKSIZE/NW > 4) qll[0] += shfl_down(qll[0], 4, 8);
				if (BLOCKSIZE/NW > 2) qll[0] += shfl_down(qll[0], 2, 4);
				if (BLOCKSIZE/NW > 1) qll[0] += shfl_down(qll[0], 1, 2);

			const bool write = ((j % (BLOCKSIZE/NW)) == 0);

		#else	/* WARPSZE == 64 */

				{	real y = yl[itl];
					#pragma unroll
					for (int a=0; a<NACC; a++)  qll[a] = my_reo[a] * y;
				}
				const int ql_ofs = (l+ll) + (b*NFIELDS+f0)*ql_dist;		// compute destination ofset in parallel with reduce!
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					real y = yl[itl + k];
					#pragma unroll
					for (int a=0; a<NACC; a++) qll[a] += my_reo[k+a] * y;
				}

				al += LSPAN;
				if (j<LSPAN) ak[j+2] = al[j];	// loading after accumulation is more efficient

				// reduce the NACC independent accumulators, which are shuffled accross lanes so that they share the same y above
				if (NACC>1) qll[0] += shfl_xor(qll[1],1);
				if (NACC==4) {
					qll[NACC-2] += shfl_xor(qll[NACC-1],1);
					qll[0]      += shfl_xor(qll[NACC-2],2);
				}

				static_assert(BLOCKSIZE/NW <= 8, "Blocksize must not exceed 8*NW");
					// reduce_add within same l is in same warp too:
					if (BLOCKSIZE/NW > 4) qll[0] += shfl_down(qll[0], 4*LSPAN, 8*LSPAN);
					if (BLOCKSIZE/NW > 2) qll[0] += shfl_down(qll[0], 2*LSPAN, 4*LSPAN);
					if (BLOCKSIZE/NW > 1) qll[0] += shfl_down(qll[0],   LSPAN, 2*LSPAN);

		#endif

				if ( write && ((l+ll)<=llim) ) {	// write result
					#if NLAT_2 <= BLKSZE_A
						// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
						#ifndef ILEG_ISHIOKA
							//if (S==0)	ql[ql_ofs + (l+ll)] = qll[0] * xlm[l+ll];	// this can be done here without the need for another kernel... maybe ?
							//else
						#endif
						ql[ql_ofs] = qll[0];
					#else
						atomicAdd_sht(ql+ql_ofs, qll[0]);		// VERY slow atomic add on Kepler.
					#endif
				}

			l+=LSPAN;
		} while (l <= llim);
	}
#if M0_ONLY==0
	else {	// im > 0
		const int NW = NFIELDS*LSPAN*2;
		real my_reo[NW];			// in registers
		const int m = im*MRES;
		int l = (im*(2*(LMAX+1)-MRES-m))>>1;
	  #ifdef ILEG_ISHIOKA
		al += l+m;
	  #else
		al += im*(LMAX+3) - (m*(im-1))/2;
	  #endif
		if (j < 2) ak[j] = al[j];
		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1

		#if NLAT_2 > BLKSZE_A
		{	// polar optimization
			bool skip_block = (j == BLOCKSIZE-1) ? polar_skip_cost(cost, llim, m) : false;
			#if WARPSZE == 32
			if (BLOCKSIZE == WARPSZE) skip_block = _any(skip_block);	// get largest value in block/warp
			#else
			if (BLOCKSIZE == WARPSZE) skip_block = shfl(skip_block,WARPSZE-1,WARPSZE);	// get largest value in block/warp
			#endif
			else {
				__shared__ volatile int xx;
				if (j == BLOCKSIZE-1) xx = skip_block;	// one thread writes its value (the largest one)
				__syncthreads();
				skip_block = xx;	// everyone reads that value
			}
			// at this point, block is in sync (consistent view of shared memory)
			if (skip_block)  return;
		}
		#endif

		q += b*NFIELDS*q_dist;
	  #ifdef ILEG_ISHIOKA
		const real cost_ = cost;
		const real sgn = (j^1)-j;	//	1-2*(j&1);	// +/-
		const real costx = shfl_xor(cost_, 1)*sgn;		// neighboor cost for "reverse" exchange
	  #else
		const real cost_ = 1;
		const real sgn = (j^1)-j;	//	1-2*(j&1);	// +/-
		const real costx = sgn;		// neighboor cost for "reverse" exchange
	  #endif
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
		  #ifndef LAYOUT_REAL_FFT
			real qer = (it < nlat_2) ? q[im*m_inc        + it            + f*q_dist] : 0;	// north imag (ani)
			real t0  = (it < nlat_2) ? q[(nphi-im)*m_inc + it            + f*q_dist] : 0;	// north real (an)
			real qor = (it < nlat_2) ? q[im*m_inc        + nlat_2*2-1-it + f*q_dist] : 0;	// south imag (asi)
			real t1  = (it < nlat_2) ? q[(nphi-im)*m_inc + nlat_2*2-1-it + f*q_dist] : 0;	// south real (as)
			real qei = t0-qer;		qer += t0;		// ani = -qei[lane+1],   bni = qei[lane-1]
			real qoi = t1-qor;		qor += t1;		// bsi = -qoi[lane-1],   asi = qoi[lane+1];

			yl[(f*4+3)*l_inc +(j^1)] = (qei + qoi)*costx;	// roi, exchange even and odd lanes
			yl[(f*4+2)*l_inc + j]    = (qer - qor)*cost_;	// ror
			yl[(f*4+1)*l_inc +(j^1)] = (qei - qoi)*sgn;		// rei, exchange even and odd lanes
			yl[f*4*l_inc     + j]    =  qer + qor;			// rer
		  #else
			long ofs = f*q_dist + im*2*m_inc;
			real qer = (it < nlat_2) ? q[ofs + it*2] : 0;	// north, real
			real qei  = (it < nlat_2) ? q[ofs + it*2+1] : 0;	// north, imag
			real qor = (it < nlat_2) ? q[ofs + 2*(m_inc - it-1)] : 0;	// south, real
			real qoi  = (it < nlat_2) ? q[ofs + 2*(m_inc - it-1)+1] : 0;	// south, imag

			yl[(f*4+3)*l_inc + j] = (qei - qoi)*cost_;	// roi
			yl[(f*4+2)*l_inc + j] = (qer - qor)*cost_;	// ror
			yl[(f*4+1)*l_inc + j] =  qei + qoi;			// rei
			yl[f*4*l_inc     + j] =  qer + qor;			// rer
		  #endif
		}

		//const int NW = NFIELDS*LSPAN*2;
		#if WARPSZE == 32
			const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);		// actualy ll = 2*l + (imag ? 1 : 0)
			const int ofs = (4*f0+(ll&3))*l_inc + j % (BLOCKSIZE/NW);
			#ifndef __gfx90a__
				const int NACC = 2;		// number of independent accumulators (2 is the sweetspot for V100).
			#else
				const int NACC = 4;		// number of independent accumulators (4 is the sweetspot for MI200 / CDNA2).
			#endif
		#else
			const int ll = (j % (2*LSPAN));		// actualy ll = 2*l + (imag ? 1 : 0)
			const bool write = (BLOCKSIZE == 2*LSPAN*NFIELDS) ? true : ((j % (BLOCKSIZE/NFIELDS)) < 2*LSPAN);
			#ifdef ILEG_ISHIOKA
				const int NACC = 4;		// influences the register layout in order to minimize shared memory loads				
			#else
				const int NACC = 2;		// Ishioka recurrence required for NACC=4
			#endif
		#endif

		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }

			// transpose yl to my_reo (registers)

			#pragma unroll
			for (int k=0; k<NW; k++) {
				#if WARPSZE == 32
					my_reo[k] = yl[ofs + k*(BLOCKSIZE/NW)];
				#else
					const int ofs = (4*f0+((ll&3)^(k%NACC)))*l_inc + (((j/(2*LSPAN))*NW) % BLOCKSIZE);
					my_reo[k] = yl[ofs + (k/NACC)*NACC + j%NACC];		// all lanes have different ordering. Less bank conflicts?
				#endif
			}

		y1 = (it < nlat_2) ? ct[it + 2*nlat_2] : 0;		// sin(theta)
	  #ifdef ILEG_ISHIOKA
		cost *= cost;		// cos(theta)^2
	  #endif
		#if HI_LLIM==1
		int ny = 0;
		#endif
		{	// compute sin(theta)^(m-S)
			y0 = MPOS_SCALE;	// y0
			l = m - S;		// exponent of sin(theta)
			if (ROBERT_FORM && S==1) {
				if (MRES==1 && l==0) {		// division by sin(theta) only for m=1 in Robert form (incorrect at the poles)
					if (it < nlat_2) y0 *= ct[it + 3*nlat_2];		// 1/sin(theta)
				} else --l;		// otherwise we just reduce the exponent of sin(theta)^l
			}
			#if HI_LLIM==1
			int nsint = 0;
			#endif
			do {		// sin(theta)^(m-S)
				if (l&1) {
					y0 *= y1;
					#if HI_LLIM==1
						ny += nsint;
						if (y0 < (SHT_ACCURACY+1/SHT_SCALE_FACTOR)) {
							ny--;
							y0 *= SHT_SCALE_FACTOR;
						}
					#endif
				}
				y1 *= y1;
				#if HI_LLIM==1
					nsint += nsint;
					if (y1 < 1/SHT_SCALE_FACTOR) {
						nsint--;
						y1 *= SHT_SCALE_FACTOR;
					}
				#endif
			} while(l >>= 1);
		}

		if (it < nlat_2)     y0 *= ct[it + nlat_2];		// include quadrature weights.
	  #ifdef ILEG_ISHIOKA
		y1 = (ak[1]*cost + ak[0]) * y0;
	  #else
		y1 = ak[1]*cost * y0;
	  #endif

		l=m;		al+=2;		int k0 = 0;
		if ((BLOCKSIZE==WARPSZE  ||  j<WARPSZE) && (l+j<=llim))  ak[j] = al[j];
		al += WARPSZE;
	  #ifdef ILEG_ISHIOKA
	    #if WARPSZE == 32
			const int itl = (ll>>2)*l_inc + (j % (BLOCKSIZE/NW));		// transposed work (at given l)
		#else
			const int itl = (ll>>2)*l_inc + (((j/(2*LSPAN))*NW) % BLOCKSIZE) + j%NACC;		// transposed work (at given l)
		#endif
	  #else
	    #if WARPSZE == 32
			const int itl = (ll>>1)*l_inc + (j % (BLOCKSIZE/NW));		// transposed work (at given l)
		#else
			const int itl = (ll>>1)*l_inc + (((j/(2*LSPAN))*NW) % BLOCKSIZE) + j%NACC;		// transposed work (at given l)
		#endif
	  #endif
	#if HI_LLIM==1
		static_assert(BLOCKSIZE == WARPSZE, "with HI_LLIM, block size must equal warp size");
		#if WARPSZE == 32
		unsigned int y_zero = _ballot(ny);
		#else
		unsigned long long y_zero = _ballot(ny);
		#endif
		#ifdef ILEG_ISHIOKA
		if (ny) for (int k=0; k<LSPAN/2; k++)  yl[k*l_inc +j] = 0;
		#else
		if (ny) for (int k=0; k<LSPAN; k++)  yl[k*l_inc +j] = 0;
		#endif
		while (y_zero && l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
		  #ifdef ILEG_ISHIOKA
			#pragma unroll 4
			for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real_g c0 = ak[k0 + 2*k+1]*cost + ak[k0 + 2*k];
				real_g c1 = ak[k0 + 2*k+3]*cost + ak[k0 + 2*k+2];
					if (fabs(y0) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1)
					{	// rescale when value is significant
						++ny;
						y0 *= 1/SHT_SCALE_FACTOR;
						y1 *= 1/SHT_SCALE_FACTOR;
					}
				if (ny==0) yl[k*l_inc +j]     = y0;		// l and l+1
				if (ny==0) yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}
		  #else
			#pragma unroll 4
			for (int k=0; k<LSPAN; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real_g c0 = ak[k0 + k]*cost;
				real_g c1 = ak[k0 + k+1]*cost;
					if (fabs(y0) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1)
					{	// rescale when value is significant
						++ny;
						y0 *= 1/SHT_SCALE_FACTOR;
						y1 *= 1/SHT_SCALE_FACTOR;
					}
				if (ny==0) yl[k*l_inc +j]     = y0;		// l and l+1
				if (ny==0) yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}
		  #endif
			k0 += LSPAN;

			y_zero = _ballot(ny);	// at this point block is in sync (consistent view of shared memory).

			if (k0==WARPSZE) {
				if ((BLOCKSIZE==WARPSZE  ||  j<WARPSZE) && (l+j+LSPAN<=llim))  ak[j] = al[j];
				al+=WARPSZE;	k0=0;
			}

			if (y_zero + 1 != 0) {		// when all y are zero (all bits set -- independent of size), we can skip this.

				real qlri[NACC];		// accumulators (NACC can be 1, 2 or 4)

			#if WARPSZE == 32

				#pragma unroll
				for (int a=0; a<NACC; a++) {	// NACC independent accumulators
					qlri[a]   = my_reo[a]   * yl[itl + a*(BLOCKSIZE/NW)];
				}
				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination offset in parallel with reduce!
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					#pragma unroll
					for (int a=0; a<NACC; a++) {	// NACC independent accumulators
						qlri[a]   += my_reo[k+a]   * yl[itl + (k+a)*(BLOCKSIZE/NW)];
					}
				}
				if (NACC>1) {	// reduce the NACC independent accumulators
					#pragma unroll
					for (int a=0; a<NACC; a+=2) 	qlri[a] += qlri[a+1];
					#pragma unroll
					for (int a=2; a<NACC; a+=2) 	qlri[0] += qlri[a];
				}

				static_assert(BLOCKSIZE/NW <= 8, "Blocksize must not exceed 8*NW");
					// reduce_add within same l is in same warp too:
					if (BLOCKSIZE/NW > 4) qlri[0] += shfl_down(qlri[0], 4, 8);
					if (BLOCKSIZE/NW > 2) qlri[0] += shfl_down(qlri[0], 2, 4);
					if (BLOCKSIZE/NW > 1) qlri[0] += shfl_down(qlri[0], 1, 2);
					
				const bool write = ((j % (BLOCKSIZE/NW)) == 0);

			#else  /* WARPSZE == 64 : avoids shared memory bank conflicts on AMD */

				{	real y = yl[itl];
					#pragma unroll
					for (int a=0; a<NACC; a++)  qlri[a] = my_reo[a] * y;
				}
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					real y = yl[itl + k];
					#pragma unroll
					for (int a=0; a<NACC; a++) qlri[a] += my_reo[k+a] * y;
				}

				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination offset in parallel with reduce!
				// reduce the NACC independent accumulators, which are shuffled accross lanes so that they share the same y above
				if (NACC>1) qlri[0] += shfl_xor(qlri[1],1);
				if (NACC==4) {
					qlri[NACC-2] += shfl_xor(qlri[NACC-1],1);
					qlri[0]      += shfl_xor(qlri[NACC-2],2);
				}

				static_assert(BLOCKSIZE/NW <= 4, "Blocksize must not exceed 4*NW");
					// reduce_add within same l is in same warp too:
					if (BLOCKSIZE/NW > 2) qlri[0] += shfl_down(qlri[0], 4*LSPAN, 8*LSPAN);
					if (BLOCKSIZE/NW > 1) qlri[0] += shfl_down(qlri[0], 2*LSPAN, 4*LSPAN);

			#endif

					if ( write && ((l+(ll>>1))<=llim) ) {	// write result
						#if NLAT_2 <= BLKSZE_A
							// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
							ql[ql_ofs]   = qlri[0];
						#else
							atomicAdd_sht(ql+ql_ofs, qlri[0]);
						#endif
					}
			}

			l+=LSPAN;
		}
	#endif

		while (l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
		  #ifdef ILEG_ISHIOKA
			#pragma unroll 4
			for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real_g c0 = ak[k0 + 2*k+1]*cost + ak[k0 + 2*k];
				real_g c1 = ak[k0 + 2*k+3]*cost + ak[k0 + 2*k+2];
				yl[k*l_inc +j]     = y0;		// l and l+1
				yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}
		  #else
			#pragma unroll 4
			for (int k=0; k<LSPAN; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real_g c0 = ak[k0 + k]*cost;
				real_g c1 = ak[k0 + k+1]*cost;
				yl[k*l_inc +j]     = y0;		// l and l+1
				yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}
		  #endif
			k0 += LSPAN;

			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			// at this point block is in sync (consistent view of shared memory).

				real qlri[NACC];		// accumulators (NACC can be 1, 2 or 4)

			#if WARPSZE == 32

				#pragma unroll
				for (int a=0; a<NACC; a++) {	// NACC independent accumulators
					qlri[a]   = my_reo[a]   * yl[itl + a*(BLOCKSIZE/NW)];
				}
				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination offset in parallel with reduce!
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					#pragma unroll
					for (int a=0; a<NACC; a++) {	// NACC independent accumulators
						qlri[a]   += my_reo[k+a]   * yl[itl + (k+a)*(BLOCKSIZE/NW)];
					}
				}

				if (NACC>1) {	// reduce the NACC independent accumulators
					#pragma unroll
					for (int a=0; a<NACC; a+=2) 	qlri[a] += qlri[a+1];
					#pragma unroll
					for (int a=2; a<NACC; a+=2) 	qlri[0] += qlri[a];
				}

				static_assert(BLOCKSIZE/NW <= 8, "Blocksize must not exceed 8*NW");
					// reduce_add within same l is in same warp too:
					if (BLOCKSIZE/NW > 4) qlri[0] += shfl_down(qlri[0], 4, 8);
					if (BLOCKSIZE/NW > 2) qlri[0] += shfl_down(qlri[0], 2, 4);
					if (BLOCKSIZE/NW > 1) qlri[0] += shfl_down(qlri[0], 1, 2);

				const bool write = ((j % (BLOCKSIZE/NW)) == 0);

			#else  /* WARPSZE == 64 : avoids shared memory bank conflicts on AMD */

				{	real y = yl[itl];
					#pragma unroll
					for (int a=0; a<NACC; a++)  qlri[a] = my_reo[a] * y;
				}
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					real y = yl[itl + k];
					#pragma unroll
					for (int a=0; a<NACC; a++) qlri[a] += my_reo[k+a] * y;
				}

				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination offset in parallel with reduce!
				// reduce the NACC independent accumulators, which are shuffled accross lanes so that they share the same y above
				if (NACC>1) qlri[0] += shfl_xor(qlri[1],1);
				if (NACC==4) {
					qlri[NACC-2] += shfl_xor(qlri[NACC-1],1);
					qlri[0]      += shfl_xor(qlri[NACC-2],2);
				}

				static_assert(BLOCKSIZE/NW <= 4, "Blocksize must not exceed 4*NW");
					// reduce_add within same l is in same warp too:
					if (BLOCKSIZE/NW > 2) qlri[0] += shfl_down(qlri[0], 4*LSPAN, 8*LSPAN);
					if (BLOCKSIZE/NW > 1) qlri[0] += shfl_down(qlri[0], 2*LSPAN, 4*LSPAN);

			#endif

				if (k0==WARPSZE) {
					if ((BLOCKSIZE==WARPSZE  ||  j<WARPSZE) && (l+j+LSPAN<=llim))  ak[j] = al[j];
					al+=WARPSZE;	k0=0;
				}

					if ( write && ((l+(ll>>1))<=llim) ) {	// write result
						#if NLAT_2 * NF_A <= 512  &&  !defined( ILEG_ISHIOKA )
							//if (S==0)	qlri[0] *= xlm[ofs_to_be_determined + l+(ll>>1)];	// this can be done here without the need for another kernel... maybe ?
						#endif
						#if NLAT_2 <= BLKSZE_A
							// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
							ql[ql_ofs]   = qlri[0];
						#else
							atomicAdd_sht(ql+ql_ofs, qlri[0]);
						#endif
					}

			l+=LSPAN;
		}

	}
  #endif
}
