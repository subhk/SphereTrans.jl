/*
 * Copyright (c) 2010-2024 Centre National de la Recherche Scientifique.
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

// Various CUDA kernels for SHTns

/// Maximum number of threads per block that should be used.
#define MAX_THREADS_PER_BLOCK 256

// adjustment for cuda
#undef SHT_L_RESCALE_FLY
#define SHT_L_RESCALE_FLY 1800

// when possible, allows to fuse sh2ishioka into leg_m_kernel, reducing memory traffic
#define SHT_ALLOW_SH2ISH_FUSE 1


/// Macro to check for cuda error and print details
#define CUDA_ERROR_CHECK cuda_error_check(__FILE__, __LINE__)
bool cuda_error_check(const char* fname, int l)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR %s:%d : %s!\n", fname, l, cudaGetErrorString(err));
		return true;
	}
	return false;
}



/// dim0, dim1 : size in complex numbers !
/// BLOCK_DIM_Y must be a power of 2 between 1 and 16
template<int TILE_DIM, int BLOCK_DIM_Y, typename T=double, int MULT=2> __global__ void
transpose_cplx_zero_C2R_kernel(const T* in, T* out, const int dim0, const int dim1, const int mmax_plan, const int mlim,  int idist, int odist)
{
	__shared__ T shrdMem[TILE_DIM][TILE_DIM+1][MULT];		// avoid shared mem conflicts

	const int ly = threadIdx.y;
	const int lx = threadIdx.x / MULT;
	const int ri = (MULT<=1) ? 0 : threadIdx.x & (MULT-1);		// real/imag index

	const int by = TILE_DIM * blockIdx.x;
	const int bx = TILE_DIM * (gridDim.y-1-blockIdx.y);

	int gy = ly + by;
	int gx = lx + bx;

	const int k = (gridDim.z-1-blockIdx.z);

	if (gx < dim0) {
		in += k * idist;
		#pragma unroll
		for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			if (gy_ > mlim) break;
			//shrdMem[ly + repeat][lx^(ly+repeat)][ri] = in[MULT*(gy_ * dim0 + gx) + ri];
			shrdMem[ly + repeat][lx][ri] = in[MULT*(gy_ * dim0 + gx) + ri];
		}
	}

	// transpose tiles:
	gx = ly + bx;	// theta
	gy = lx + by;	// m

	__syncthreads();

	// transpose within tile:
	if (gy > mmax_plan) return;

	out += k * odist;
	T z = {};	// zero
	#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
		int gx_ = gx+repeat;
		if (gx_ >= dim0) break;
		if (gy <= mlim) z = shrdMem[lx][ly + repeat][ri];
		//if (gy <= mlim) z = shrdMem[lx][(ly + repeat)^lx][ri];
		out[MULT*(gx_ * dim1 + gy) + ri] = z;
	}
}

/// dim0, dim1 : size in complex numbers !
/// BLOCK_DIM_Y must be a power of 2 between 1 and 16
template<int TILE_DIM, int BLOCK_DIM_Y, typename real=double, int MULT> __global__ void
transpose_cplx_skip_R2C_kernel(const real* in, real* out, const int dim0, const int dim1, const int mmax,  int idist, int odist)
{
	__shared__ real shrdMem[TILE_DIM][TILE_DIM+1][MULT];		// avoid shared mem conflicts

	const int lx = threadIdx.x/MULT;
	const int ly = threadIdx.y;
	const int ri = (MULT<=1) ? 0 : threadIdx.x & (MULT-1);		// real/imag index

	const int bx = TILE_DIM * blockIdx.x;
	const int by = TILE_DIM * blockIdx.y;

	int gx = lx + bx;	// m
	int gy = ly + by;	// ilat

	const int k = (gridDim.z-1-blockIdx.z);

	if (gx <= mmax) {		// read only data if m<=mmax
		in += k * idist;
		#pragma unroll
		for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			if (gy_ >= dim1) break;
			//shrdMem[ly + repeat][lx^(ly+repeat)][ri] = in[MULT*(gy_ * dim0 + gx) + ri];
			shrdMem[ly + repeat][lx][ri] = in[MULT*(gy_ * dim0 + gx) + ri];
		}
	}

	// transpose tiles:
	gy = ly + bx;	// m
	gx = lx + by;	// ilat

	__syncthreads();
	// transpose within tile:
	if (gx >= dim1) return;
	out += k * odist;
	#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
		int gy_ = gy+repeat;
		if (gy_ <= mmax) 		// write only useful data
			//out[MULT*(gy_ * dim1 + gx) + ri] = shrdMem[lx][(ly + repeat)^lx][ri];
			out[MULT*(gy_ * dim1 + gx) + ri] = shrdMem[lx][(ly + repeat)][ri];
	}
}


/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_zero_C2R(cudaStream_t stream, const void* in, void* out, const int dim0, const int dim1,
		const int mmax_plan, int mlim, int sizeof_real, int nbatch, int idist, int odist)
{
	if (sizeof_real==8) {
		const int tile_dim = 16;
		#if WARPSZE == 64
			const int block_dim_y = 16;		// good performance with 16 on AMD MI200 (MUST be power of 2 between 1 and 16)
		#else
			const int block_dim_y = 8;		// good performance with 8 on nvidia A30 (MUST be power of 2 between 1 and 16)
		#endif
		dim3 blocks((dim1+tile_dim-1)/tile_dim, (dim0+tile_dim-1)/tile_dim, nbatch);
		dim3 threads(tile_dim*2, block_dim_y, 1);
		transpose_cplx_zero_C2R_kernel<tile_dim, block_dim_y,double,2> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax_plan, mlim, idist, odist);
	} else {
		const int tile_dim = 32;
		const int block_dim_y = 16;		// good performance with 16 on nvidia A30 and AMD MI200 (MUST be power of 2 between 1 and tile_dim)
		dim3 blocks((dim1+tile_dim-1)/tile_dim, (dim0+tile_dim-1)/tile_dim, nbatch);
		dim3 threads(tile_dim, block_dim_y, 1);
		transpose_cplx_zero_C2R_kernel<tile_dim, block_dim_y,double,1> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax_plan, mlim, idist/2, odist/2);
	}
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_skip_R2C(cudaStream_t stream, const void* in, void* out, const int dim0, const int dim1,
		const int mmax, int sizeof_real, int nbatch, int idist, int odist)
{
	const int tile_dim = 16;
	#if WARPSZE == 64
		const int block_dim_y = 16;		// good performance with 16 on AMD MI200 (MUST be power of 2 between 1 and 16)
	#else
		const int block_dim_y = 8;		// good performance with 8 on nvidia A30 (MUST be power of 2 between 1 and 16)
	#endif
	dim3 blocks((mmax+tile_dim)/tile_dim, (dim1+tile_dim-1)/tile_dim, nbatch);
	dim3 threads(tile_dim*2, block_dim_y, 1);
	if (sizeof_real==8) {
		dim3 threads(tile_dim*2, block_dim_y, 1);
		transpose_cplx_skip_R2C_kernel<tile_dim, block_dim_y,double, 2> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax, idist, odist);
	} else {
		dim3 threads(tile_dim, block_dim_y, 1);
		transpose_cplx_skip_R2C_kernel<tile_dim, block_dim_y,double, 1> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax, idist/2, odist/2);
	}
}



template<typename real> __global__ void
sh2ishioka_kernel(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql, real* ql_ish,
		const int llim, const int lmax, const int mres, const int S, const int ql_dist=0, const int ql_ish_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;
	const int l  = ll >> 1;
	const int llim_m = llim-m;

	if (l>llim_m) return;		// nothing to do

	// first load matrix coefficients into registers
	const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4 + 3*(ll >> 2);
	real x0 = xlm[x_ofs + (ll&2)];
	real x1 = xlm[x_ofs + 1];

	// address calculation
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres);
	const int b = (blockIdx.z*blockDim.z + threadIdx.z)*NFIELDS;
	ql     += q_ofs + b*ql_dist + ll;
	ql_ish += q_ofs + b*ql_ish_dist + ((im>0) ? ll : l);

	const bool write = (im>0 || (ll&1)==0);
	const bool add2 = ((l&1)==0) && (l+1 <llim_m);
	// loop over NFIELDS different fields
	for (int k=NFIELDS-1; k>=0; k--) {
		real q = ql[k*ql_dist] * x0;
		if (add2) {	// l-m even
			q += ql[k*ql_dist +4] * x1;		// contribution of l+2
		}
		if (write) {
			ql_ish[k*ql_ish_dist] = q;   // coalesced store -- for im=0, compacting real parts together without imaginary part (0)
		}
	}
}

template<typename real> __global__ void
sh2reduced_kernel(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql, real* ql_ish,
		const int llim, const int lmax, const int mres, const int S, const int ql_dist=0, const int ql_ish_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;
	const int l  = ll >> 1;
	const int llim_m = llim-m;

	if (l>llim_m) return;		// nothing to do

	// first load matrix coefficients into registers
	xlm += im*(lmax+3) - (m*(im-1))/2;
	real x0 = xlm[ll>>1];

	// address calculation
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres);
	const int b = (blockIdx.z*blockDim.z + threadIdx.z)*NFIELDS;
	ql     += q_ofs + b*ql_dist + ll;
	ql_ish += q_ofs + b*ql_ish_dist + ((im>0) ? ll : l);

	const bool write = (im>0 || (ll&1)==0);
	// loop over NFIELDS different fields
	for (int k=NFIELDS-1; k>=0; k--) {
		real q = ql[k*ql_dist] * x0;
		if (write) {
			ql_ish[k*ql_ish_dist] = q;   // coalesced store -- for im=0, compacting real parts together without imaginary part (0)
		}
	}
}

/// performs: Ql[2*l] = qq[2*l]*xlm[3*l] + qq[2*l-2]*xlm[3*l+1];   Ql[2*l+1] = qq[2*l+1] * xlm[3*l+2];
/// includes zero-out for unused modes.
template<typename real> __global__ void
ishioka2sh_kernel(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql_ish, real* ql,
	const int llim, const int lmax, const int mmax, const int mres, const int S, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;

	if ((ll>>1) > lmax+S-m) return;		// be sure to include zero-out for llim<l<=lmax AND zero-out for m>mmax

	// first load matrix coefficients into registers
	xlm += 3*im*(2*(lmax+4) -m+mres)/4;
	const int x_ofs = 3*(ll>>2);
	real x0 = xlm[x_ofs + (ll&2)];
	real x1;
	if (x_ofs>0) x1 = xlm[x_ofs-2];

	const int b = (blockIdx.z*blockDim.z + threadIdx.z) * NFIELDS;
	int q_ofs = ll;
	real q = 0.0;
	if (im==0) {
		ql_ish += b*ql_ish_dist + (ll>>1);
		ql += q_ofs + b*ql_dist;
		const bool read = (ll>>1) <= llim-m && ((ll&1)==0);
		const bool add2 = ((ll&2)==0) && (ll >= 4) && read;
		for (int k=NFIELDS-1; k>=0; k--) {
			if (read)  q = ql_ish[k*ql_ish_dist] * x0;	// only real part (ll&1 == 0)
			if (add2) {	// l-m even && real part (ll&3 == 0)
				q += ql_ish[k*ql_ish_dist -2] * x1;		// contribution of l-2
			}
			if (sizeof(real)==4 && ll+S==0) {	// for S==0, add the mean (l==0) as late as possible
				q += ql_ish[k*ql_ish_dist + llim + 1] * x0;
			}
			ql[k*ql_dist] = q;	// coalesced store
		}
	} else {
		q_ofs += im*(((lmax+1+S)*2) -m+mres);
		ql_ish += b*ql_ish_dist + q_ofs;
		ql += q_ofs + b*ql_dist;
		const bool read = (ll>>1) <= llim-m;
		const bool add2 = ((ll&2)==0) && (ll >= 4) && read;
		for (int k=NFIELDS-1; k>=0; k--) {
			if (read)  q = ql_ish[k*ql_ish_dist] * x0;
			if (add2) {	// l-m even
				q += ql_ish[k*ql_ish_dist -4] * x1;		// contribution of l-2
			}
			ql[k*ql_dist] = q;	// coalesced store
		}
	}
}

template<typename real, bool ISHIOKA=true> __global__ void
sphtor2ish_kernel(const real* __restrict__ mx, const real* __restrict__ xlm,
		const real* __restrict__ slm, const real* __restrict__ tlm, real *vlm, real *wlm, 
		const int llim, const int lmax, const int mres, const int ql_dist=0, const int ql_ish_dist=0)
{
	// indices for overlapping blocks:
	const int overlap = (ISHIOKA) ? 8 : 4;
	const int l0 = (blockDim.x-overlap) * blockIdx.x;		// some overlap needed
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	int ll = l0 + j - 2;

	extern __shared__ double sl_[];
	real* const sl = (real*) sl_;		// size blockDim.x
	real* const tl = sl + blockDim.x;	// size blockDim.x
	real* const M  = sl + 2*blockDim.x;	// size blockDim.x

	const int m = im*mres;
	const int llim_m = llim-m;
	const int ofs = im*(((lmax+1)<<1) -m + mres) + ll;
	ll >>= 1;

	real v = 0.0;
	real w = 0.0;
	real mm = 0.0;
	if ( (ll >= 0) && (ll <= llim_m) ) {
		mm = mx[ofs];
		if (slm) v = slm[ofs + b*ql_dist];
		if (tlm) w = tlm[ofs + b*ql_dist];
	}
	M[j] = mm;
	sl[j] = v;
	tl[j] = w;

	__syncthreads();

	const real mimag = m * (j - (j^1));
	if ((j<blockDim.x-4) && (ll <= llim_m)) {
		real ml = M[j|1];
		real mu = M[(j|1)+1];
		v = mimag*tl[(j^1)+2]  +  (ml*v + mu*sl[j+4]);
		w = mimag*sl[(j^1)+2]  -  (ml*w + mu*tl[j+4]);
	}

	int x_ofs;
	const int j2 = j - (j>>1);	//(j>>1)+(j&1);		==> only for ISHIOKA
  if (ISHIOKA) {
	__syncthreads();

	if ((j&2)==0) {
		sl[j2] = v;
		tl[j2] = w;
	}

	x_ofs = (3*im*(2*(lmax+4) -m+mres)>>2) + 3*((l0+j) >> 2);

	__syncthreads();
  }

	if ((j < blockDim.x-overlap) && (ll <= llim_m)) {
		real x0;
		if (ISHIOKA) x0 = xlm[x_ofs + (j&2)];   //M[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
		else         x0 = xlm[im*(lmax+3) - (m*(im-1))/2 + ((l0+j)>>1)];
		v *= x0;
		w *= x0;
		if (ISHIOKA && (j&2)==0) {		// for l-m even
			real x2 = xlm[x_ofs +1];   //M[ix+1];			// contribution of l+2
			v += x2 * sl[j2+2];
			w += x2 * tl[j2+2];
		}
		if (im>0) {
			vlm[ofs+2*im+2 + b*ql_ish_dist] = v;
			wlm[ofs+2*im+2 + b*ql_ish_dist] = w;
		} else if ((j&1)==0) {		// compress complex to real (as imaginary part is zero)
			vlm[(ofs>>1)+1 + b*ql_ish_dist] = v;	// coalesced store
			wlm[(ofs>>1)+1 + b*ql_ish_dist] = w;	// coalesced store
		}
	}
}

template<typename real, bool ISHIOKA=true> __global__ void
ish2sphtor_kernel(const real* __restrict__ mx, const real* __restrict__ xlm, const real* __restrict__ vlm, const real* __restrict__ wlm, 
	real *slm, real *tlm, const int llim, const int lmax, const int mres, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	const int l0 = (blockDim.x-4) * blockIdx.x;		// some overlap needed

	int l  = (l0 + j) >> 1;
	const int m = im*mres;
	const int q_ofs = im*(((lmax+1)*2) -m+mres);
	const int llim_m_p1 = llim+1-m;

	extern __shared__ double vl_[];			// size blockDim.x
	real* const vl = (real*) vl_;
	real* const wl = vl + blockDim.x+2;		// size blockDim.x

	real v = 0.0;
	real w = 0.0;
	{
		if (l<=llim_m_p1) {
			real x,x2;
			if (ISHIOKA) {
				const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4;
				x = xlm[x_ofs + 3*(l>>1) + (j&2)];
				x2 = (l>=2) ? xlm[x_ofs + 3*(l>>1) -2] : 0.0;
			} else {
				const int x_ofs = im*(lmax+3) - (m*(im-1))/2;
				x = xlm[x_ofs + l];
			}
			if (im!=0) {
				const int i = q_ofs + 2*im + b*ql_ish_dist + l0+(j^1);		// xchg real and imag
				v = vlm[i] * x;
				w = wlm[i] * x;
				if (ISHIOKA && ((j&2)==0) && (l>0)) {	// l-m even and l-m>2
					v += vlm[i-4] * x2;		// contribution of l-2
					w += wlm[i-4] * x2;
				}
			} else if (j&1) {
				const int i = q_ofs + b*ql_ish_dist + l;
				v = vlm[i] * x;
				w = wlm[i] * x;
				if (ISHIOKA && ((j&2)==0) && (l>0)) {	// l-m even and l-m>2
					v += vlm[i-2] * x2;		// contribution of l-2
					w += wlm[i-2] * x2;
				}
			}
		}
		vl[(j^1)+2] = v;
		wl[(j^1)+2] = w;
	}
	if (l==0) {  vl[j] = 0.0;	wl[j] = 0.0; }


	l += m;
	real ml,mu, ll_1;
	if ((l <= llim) && (l>0)) {
		ll_1 = 1 / (real)(l*(l+1));
		mu = mx[q_ofs + l0 + (j|1)];    //M[2*(j>>1)+3];
		ml = mx[q_ofs + l0 + (j|1) -3];	//M[2*(j>>1)+0];
	}

	__syncthreads();

	if ((j<blockDim.x-2) &&  (j >= ((blockIdx.x == 0) ? 0 : 2))) {
		real v2 = 0.0;
		real w2 = 0.0;
		if ((l <= llim) && (l>0)) {
			const real mimag = m * ((j^1) -j);
			v2 = mimag*w  +  (ml*vl[j] + mu*vl[j+4]);
			w2 = mimag*v  -  (ml*wl[j] + mu*wl[j+4]);
			v2 *= ll_1;
			w2 *= ll_1;
		}
		if (l <= lmax) {	// fill with zeros up to lmax (and l=0 too).
			slm[q_ofs+l0+j + b*ql_dist] = v2;
			tlm[q_ofs+l0+j + b*ql_dist] = w2;
		}
	}
}

void set_block_size_ish(int n_elem_x, int howmany_z, int& blksze_x, int& blksze_z, int &nblk_z)
{
	blksze_z = 1;		nblk_z = howmany_z;
	if (howmany_z % 4 == 0) {	blksze_z = 4;	nblk_z /= 4;  }
	else if (howmany_z % 3 == 0) {  blksze_z = 3;	nblk_z /= 3;  }
	else if (howmany_z % 2 == 0) {  blksze_z = 2;	nblk_z /= 2;  }

	if (blksze_z > 1) {
		blksze_x = (((n_elem_x+1)/2+WARPSZE-1)/WARPSZE) * WARPSZE;
		if (blksze_x > MAX_THREADS_PER_BLOCK/2) blksze_x = MAX_THREADS_PER_BLOCK/2;
	} else {
		blksze_x = ((n_elem_x+WARPSZE-1)/WARPSZE) * WARPSZE;
		if (blksze_x > MAX_THREADS_PER_BLOCK) blksze_x = MAX_THREADS_PER_BLOCK;
	}
}

template<typename real=double>
void sh2ishioka_gpu(shtns_cfg shtns, std::complex<real>* d_Qlm, std::complex<real>* d_Qlm_ish, int llim, int mmax, int S=0)
{
	int blksze, blksze_z, nblk_z;
	const int nelem_max = (llim+1+S)*2;
	int nfields = shtns->howmany;
	if (shtns->nlm > 256*1024 || nfields <= 5) {	// enough work to saturate the GPU with 1 z-block, or only few fields
		set_block_size_ish( nelem_max, 1, blksze, blksze_z, nblk_z);
	} else {
		nblk_z = nfields;	nfields=1;
		for (int k=4; k>=2; k--) {		// first factorization
			if (nblk_z % k == 0) { nblk_z /= k;	nfields=k;	break; }
		}
		set_block_size_ish( nelem_max, nblk_z, blksze, blksze_z, nblk_z);		// second factor into blocks
	}
	dim3 blocks((nelem_max+blksze-1)/blksze, mmax+1, nblk_z);
	dim3 threads(blksze, 1, blksze_z);
	const real* xlm = (real*) shtns->d_xlm;
	if (shtns->kernel_flags & CUSHT_NO_ISHIOKA) {	// no ishioka
		sh2reduced_kernel <<< blocks, threads, 0, shtns->comp_stream >>>
			(nfields, xlm, (real*) d_Qlm, (real*) d_Qlm_ish, llim, shtns->lmax, shtns->mres, S, shtns->spec_dist*2, shtns->nlm_stride);
	} else {
		sh2ishioka_kernel <<< blocks, threads, 0, shtns->comp_stream >>>
			(nfields, xlm, (real*) d_Qlm, (real*) d_Qlm_ish, llim, shtns->lmax, shtns->mres, S, shtns->spec_dist*2, shtns->nlm_stride);
	}
	CUDA_ERROR_CHECK;
}

/// performs: Ql[2*l] = qq[2*l]*xlm[3*l] + qq[2*l-2]*xlm[3*l+1];   Ql[2*l+1] = qq[2*l+1] * xlm[3*l+2];
/// includes zero-out for unused modes.
template<typename real> __global__ void
reduced2sh_kernel(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql_ish, real* ql,
	const int llim, const int lmax, const int mmax, const int mres, const int S, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;

	if ((ll>>1) > lmax+S-m) return;		// be sure to include zero-out for llim<l<=lmax AND zero-out for m>mmax

	// first load matrix coefficients into registers
	xlm += im*(lmax+3) - (m*(im-1))/2;
	real x0 = xlm[ll>>1];

	const int b = (blockIdx.z*blockDim.z + threadIdx.z) * NFIELDS;
	int q_ofs = ll;
	real q = 0.0;
	if (im==0) {
		ql_ish += b*ql_ish_dist + (ll>>1);
		ql += q_ofs + b*ql_dist;
		const bool read = (ll>>1) <= llim && ((ll&1)==0);
		for (int k=NFIELDS-1; k>=0; k--) {
			if (read)  q = ql_ish[k*ql_ish_dist] * x0;	// only real part (ll&1 == 0)
			if (sizeof(real)==4 && ll+S==0) {	// for S==0, add the mean (l==0) as late as possible
				q += ql_ish[k*ql_ish_dist + llim + 1] * x0;
			}
			ql[k*ql_dist] = q;	// coalesced store
		}
	} else {
		q_ofs += im*(((lmax+1+S)*2) -m+mres);
		ql_ish += b*ql_ish_dist + q_ofs;
		ql += q_ofs + b*ql_dist;
		const bool read = (ll>>1) <= llim-m;
		for (int k=NFIELDS-1; k>=0; k--) {
			if (read)  q = ql_ish[k*ql_ish_dist] * x0;	// only real part (ll&1 == 0)
			ql[k*ql_dist] = q;	// coalesced store
		}
	}
}

template<typename real=double>
void ishioka2sh_gpu(shtns_cfg shtns, std::complex<real>* d_Qlm_ish, std::complex<real>* d_Qlm, int llim, int mmax, int S=0)
{
	int blksze, blksze_z, nblk_z;
	const int nelem_max = (shtns->lmax+1+S)*2;
	int nfields = shtns->howmany;
	if (shtns->nlm > 256*1024 || nfields <= 5) {	// enough work to saturate the GPU with 1 z-block, or only few fields
		set_block_size_ish( nelem_max, 1, blksze, blksze_z, nblk_z);
	} else {
		nblk_z = nfields;	nfields=1;
		for (int k=4; k>=2; k--) {		// first factorization
			if (nblk_z % k == 0) { nblk_z /= k;	nfields=k;	break; }
		}
		set_block_size_ish( nelem_max, nblk_z, blksze, blksze_z, nblk_z);		// second factor into blocks
	}

	dim3 blocks((nelem_max+blksze-1)/blksze, shtns->mmax+1, nblk_z);
	dim3 threads(blksze, 1, blksze_z);
	const real* xlm = (real*) shtns->d_x2lm;
	if (shtns->kernel_flags & CUSHT_NO_ISHIOKA) {
		reduced2sh_kernel <<< blocks, threads, 0, shtns->comp_stream >>>
			(nfields, xlm, (real*) d_Qlm_ish, (real*) d_Qlm, llim, shtns->lmax, mmax, shtns->mres, S, shtns->nlm_stride, shtns->spec_dist*2);
	} else {
		ishioka2sh_kernel <<< blocks, threads, 0, shtns->comp_stream >>>
			(nfields, xlm, (real*) d_Qlm_ish, (real*) d_Qlm, llim, shtns->lmax, mmax, shtns->mres, S, shtns->nlm_stride, shtns->spec_dist*2);
	}
	CUDA_ERROR_CHECK;
}

template<typename real=double>
void sphtor2scal_gpu(shtns_cfg shtns, std::complex<real>* d_Slm, std::complex<real>* d_Tlm, std::complex<real>* d_Vlm, std::complex<real>* d_Wlm, int llim, int mmax)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	const int overlap = (shtns->kernel_flags & CUSHT_NO_ISHIOKA) ? 4 : 8;
	dim3 blocks((2*(shtns->lmax+3)+blksze-overlap-1)/(blksze-overlap), mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	if (shtns->kernel_flags & CUSHT_NO_ISHIOKA) {
		sphtor2ish_kernel<real, false> <<< blocks, threads, blksze*3*sizeof(real), shtns->comp_stream >>>
			((real*) shtns->d_mx_stdt, (real*) shtns->d_xlm, (real*) d_Slm, (real*) d_Tlm, (real*) d_Vlm, (real*) d_Wlm, llim, shtns->lmax, shtns->mres, shtns->spec_dist*2, shtns->nlm_stride);
	} else
	sphtor2ish_kernel <<< blocks, threads, blksze*3*sizeof(real), shtns->comp_stream >>>
		((real*) shtns->d_mx_stdt, (real*) shtns->d_xlm, (real*) d_Slm, (real*) d_Tlm, (real*) d_Vlm, (real*) d_Wlm, llim, shtns->lmax, shtns->mres, shtns->spec_dist*2, shtns->nlm_stride);
	CUDA_ERROR_CHECK;
}

template<typename real=double>
void scal2sphtor_gpu(shtns_cfg shtns, std::complex<real>* d_Vlm, std::complex<real>* d_Wlm, std::complex<real>* d_Slm, std::complex<real>* d_Tlm, int llim)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	const int overlap = 4;
	dim3 blocks((2*(shtns->lmax+3)+blksze-overlap-1)/(blksze-overlap), shtns->mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	if (shtns->kernel_flags & CUSHT_NO_ISHIOKA) {
		ish2sphtor_kernel<real, false> <<< blocks, threads, (blksze+2)*2*sizeof(real), shtns->comp_stream >>>
			((real*) shtns->d_mx_van, (real*) shtns->d_x2lm, (real*) d_Vlm, (real*) d_Wlm, (real*)d_Slm, (real*)d_Tlm, llim, shtns->lmax, shtns->mres, shtns->nlm_stride, shtns->spec_dist*2);
	} else
	ish2sphtor_kernel <<< blocks, threads, (blksze+2)*2*sizeof(real), shtns->comp_stream >>>
		((real*) shtns->d_mx_van, (real*) shtns->d_x2lm, (real*) d_Vlm, (real*) d_Wlm, (real*)d_Slm, (real*)d_Tlm, llim, shtns->lmax, shtns->mres, shtns->nlm_stride, shtns->spec_dist*2);
	CUDA_ERROR_CHECK;
}

