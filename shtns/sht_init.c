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

/********************************************************************
 * SHTns : Spherical Harmonic Transform for numerical simulations.  *
 *    written by Nathanael Schaeffer / CNRS                         *
 ********************************************************************/

/** \internal \file SHT.c
 * \brief main source file for SHTns.
 * This files contains initialization code and also some partial transforms (point or latitudinal evaluations)
 */

#include <stdio.h>
#include <string.h>
// global variables definitions
#include "sht_private.h"

#include <time.h>		// for the clock() function
// cycle counter from FFTW
#include "fftw3/cycle.h"

// chained list of sht_setup : start with NULL
shtns_cfg sht_data = NULL;

#ifdef _OPENMP
  #include <omp.h>
  int omp_threads = 1;	// multi-thread disabled by default.
  #if HAVE_LIBFFTW3_OMP
	#define OMP_FFTW
  #endif
#else
  #define omp_threads 1
#endif

static int verbose = 0;		// runtime verbosity control: 0 no output, 1 output, 2 debug mode, 3 full
void shtns_verbose(int v) {
	const char* vv = getenv("SHTNS_VERBOSE");
	if (vv && strlen(vv) > 0)		v = atoi(vv);		// environment variable takes precedence.
	if (v>=0) verbose = v;		// negative input means don't touch it; useful for just reading environment variable if any
}

#ifdef SHTNS_ISHIOKA
#define _SHTNS_ID_ _SIMD_NAME_ ",ishioka"
#else
#define _SHTNS_ID_ _SIMD_NAME_
#endif

#ifndef SHTNS_VER
#define SHTNS_VER PACKAGE_VERSION
#endif

/// \internal Abort program with error message.
static void shtns_runerr(const char * error_text)
{
	printf("*** [SHTns] Run-time error : %s\n",error_text);
	exit(1);
}

/* PUBLIC useful functions */

/// returns the l=0, m=0 SH coefficient corresponding to a uniform value of 1.
double sh00_1(shtns_cfg shtns) {
	return shtns->Y00_1;
}
/// returns the l=1, m=0 SH coefficient corresponding to cos(theta).
double sh10_ct(shtns_cfg shtns) {
	return shtns->Y10_ct;
}
/// returns the l=1, m=1 SH coefficient corresponding to sin(theta).cos(phi).
double sh11_st(shtns_cfg shtns) {
	return shtns->Y11_st;
}
/// returns the l,m SH coefficient corresponding to unit energy.
double shlm_e1(shtns_cfg shtns, int l, int m) {
	double x = shtns->Y00_1/sqrt(4.*M_PI);
	if (SHT_NORM == sht_schmidt) x *= sqrt(2*l+1);
	if ((m!=0)&&((shtns->norm & SHT_REAL_NORM)==0)) x *= sqrt(0.5);
	return(x);
}

/// \code return (mmax+1)*(lmax+1) - mres*(mmax*(mmax+1))/2; \endcode */
/// \ingroup init
long nlm_calc(long lmax, long mmax, long mres)
{
	if (mmax*mres > lmax) mmax = lmax/mres;
	return (mmax+1)*(lmax+1) - ((mmax*mres)*(mmax+1))/2;	// this is wrong if lmax < mmax*mres
}

/// \code return mmax*mmax*mres + (2*mmax+1)*(lmax-mmax*mres+1); \endcode */
/// \ingroup init
long nlm_cplx_calc(long lmax, long mmax, long mres)
{
//	return 2*nlm_calc(lmax,mmax,mres) - (lmax+1);
	if (mmax*mres > lmax) mmax = lmax/mres;
	return (2*mmax+1)*(lmax+1) - (mmax*mres)*(mmax+1);	// this is wrong if lmax < mmax*mres
}

/// recover the im index (order/mres) from the lm offset into SH array.
/// a bit costly, as it involves a sqrt + a div when mres>1
/// tested up to lmax=65535 (16 bits) and all mres up to mres=65535
int im_from_lm(long lm, int lmax, int mres) {
	if LIKELY(mres==1) {	// avoids division
		double b = 2*lmax +3;
		double delta = b*b - (lm*8+1);   		// +1 to make sure we don't have a finite precision issue when truncating to integer afterwards
		int m2 = b - sqrt(delta);
		return m2 >> 1;
	} else {
		double b = 2*lmax + 2 + mres;
		double z = 1.0/(mres+mres);
		double delta = b*b - (mres*lm*8+1);		// +1 to make sure we don't have a finite precision issue when truncating to integer afterwards
		int im = (b - sqrt(delta))*z;
		return im;
	}
}


/*  LEGENDRE FUNCTIONS  */
#include "sht_legendre.c"


/// \internal return the smallest power of 2 larger than n.
static int next_power_of_2(int n)
{
	int f = 1;
	if ( (n<=0) || (n>(1<<(sizeof(int)*8-2))) ) return 0;
	while (f<n) f*=2;
	return f;
}

/// \internal find the closest integer that is larger than n and that contains only prime factors up to fmax.
/// fmax is 7 for optimal FFTW fourier transforms.
/// return only even integers for n>fmax.
static int fft_int(int n, int fmax)
{
	int k,f;

	if (n<=fmax) return n;
	if (fmax<2) return 0;
	if (fmax==2) return next_power_of_2(n);

	n -= 2-(n&1);		// only even n
	do {
		n+=2;	f=2;
		while ((2*f <= n) && ((n&f)==0)) f *= 2;		// no divisions for factor 2.
		k=3;
		while ((k<=fmax) &&  (k*f <= n)) {
			while ((k*f <= n) && (n%(k*f)==0)) f *= k;
			k+=2;
		}
	} while (f != n);

	k = next_power_of_2(n);			// what is the closest power of 2 ?
	if ((k-n)*33 < n) return k;		// rather choose power of 2 if not too far (3%)

	return n;
}

void shtns_profiling(shtns_cfg shtns, int on) {
	shtns->cpu_timer = (on != 0) ? 0 : -1;		// enable or disable cpu timing: 0=on, -1=off
	#ifdef SHTNS_GPU
		cushtns_profiling(shtns, on);
	#endif
}

double shtns_profiling_read_time(shtns_cfg shtns, double* time_1, double* time_2)
{
	double t1=0.0;	double t2=0.0;
	#ifdef SHTNS_GPU
	cushtns_profiling_read_time(shtns, &t1, &t2);	// return intermediate kernel time (fft and legendre separated)
	#endif
	if (time_1) *time_1 = t1;	if (time_2) *time_2 = t2;
	return shtns->cpu_timer;		// return total CPU time (including transfers)
}


/*	SHT FUNCTIONS  */
#include "sht_func.c"

#include "sht_com.c"

/*
	INTERNAL INITIALIZATION FUNCTIONS
*/

// sht algorithms (hyb, fly1, ...)
enum sht_algos { SHT_ODD, SHT_MEM, SHT_SV,
	SHT_FLY1, SHT_FLY2, SHT_FLY3, SHT_FLY4, SHT_FLY6, SHT_FLY8,
	SHT_GPU1, SHT_GPU2, SHT_GPU3, SHT_GPU4,
	SHT_OMP1, SHT_OMP2, SHT_OMP3, SHT_OMP4, SHT_OMP6, SHT_OMP8,
	SHT_OMP1A, SHT_OMP2A, SHT_OMP3A, SHT_OMP4A, SHT_OMP6A, SHT_OMP8A,
	SHT_NALG };

char* sht_name[SHT_NALG] = {"odd", "mem", "s+v", "fly1", "fly2", "fly3", "fly4", "fly6", "fly8", "gpu1", "gpu2", "gpu3", "gpu4",
	"omp1a", "omp2a", "omp3a", "omp4a", "omp6a", "omp8a",  "omp1b", "omp2b", "omp3b", "omp4b", "omp6b", "omp8b",   };
char* sht_type[SHT_NTYP] = {"syn", "ana", "vsy", "van", "gsp", "gto", "v3s", "v3a" };
char* sht_var[SHT_NVAR] = {"std", "m" };
int sht_npar[SHT_NTYP] = {2, 2, 4, 4, 3, 3, 6, 6};

extern void* ffly[6][SHT_NTYP];
extern void* ffly_m[6][SHT_NTYP];
extern void* ffly_m0[6][SHT_NTYP];
extern void* fodd[SHT_NTYP];
#ifdef _OPENMP
extern void* fomp_a[6][SHT_NTYP];
extern void* fomp_b[6][SHT_NTYP];
#endif
#ifdef SHTNS_GPU
extern void* fgpu[4][SHT_NTYP];
#endif

// big array holding all sht functions, variants and algorithms
void* sht_func[SHT_NVAR][SHT_NALG][SHT_NTYP];

/// \internal use on-the-fly alogorithm (guess without measuring)
static void set_sht_fly(shtns_cfg shtns, int typ_start)
{
	int algo = SHT_FLY2;
	if ((shtns->nthreads > 1) && (sht_func[0][SHT_OMP2][typ_start])) algo = SHT_OMP2;
	for (int it=typ_start; it<SHT_NTYP; it++) {
		for (int v=0; v<SHT_NVAR; v++)
			shtns->ftable[v][it] = sht_func[v][algo][it];
	}
	if (shtns->nlat & 1) {		// odd nlat handled separately (uses other functions)
		for (int it=typ_start; it<SHT_NTYP; it++)
			shtns->ftable[0][it] = fodd[it];
	}
}

/// \internal use gpu alogorithm where possible
static void set_sht_gpu(shtns_cfg shtns, int typ_start)
{
	int algo = SHT_GPU1;
	for (int it=typ_start; it<SHT_NTYP; it++) {
		for (int v=0; v<SHT_NVAR; v++)
			if (sht_func[v][algo][it] != NULL)
				shtns->ftable[v][it] = sht_func[v][algo][it];
	}
}


/// \internal copy all algos to sht_func array (should be called by set_grid before choosing variants).
/// if nphi is 1, axisymmetric algorithms are used.
static void init_sht_array_func(shtns_cfg shtns)
{
	int it, j;
	int alg_lim = SHT_FLY8;

	if (shtns->nlat_2 < 8*VSIZE2) {		// limit available on-the-fly algorithm to avoid overflow (and segfaults).
		it = shtns->nlat_2 / VSIZE2;
		switch(it) {
			case 0 : alg_lim = SHT_FLY1-1; break;
			case 1 : alg_lim = SHT_FLY1; break;
			case 2 : alg_lim = SHT_FLY2; break;
			case 3 : alg_lim = SHT_FLY3; break;
			case 4 : ;
			case 5 : alg_lim = SHT_FLY4; break;
			default : alg_lim = SHT_FLY6;
		}
	}
	alg_lim -= SHT_FLY1;

	memset(sht_func, 0, SHT_NVAR*SHT_NTYP*SHT_NALG*sizeof(void*) );		// zero out.
	sht_func[SHT_STD][SHT_SV][SHT_TYP_3SY] = SHqst_to_spat_2l;
	sht_func[SHT_STD][SHT_SV][SHT_TYP_3AN] = spat_to_SHqst_2l;
	sht_func[SHT_M][SHT_SV][SHT_TYP_3SY] = SHqst_to_spat_2ml;
	sht_func[SHT_M][SHT_SV][SHT_TYP_3AN] = spat_to_SHqst_2ml;
	memcpy(sht_func[SHT_STD][SHT_ODD], fodd, sizeof(void*)*SHT_NTYP);

	if (shtns->nphi==1) {		// axisymmetric transform requested.
		for (int j=0; j<=alg_lim; j++) {
			memcpy(sht_func[SHT_STD][SHT_FLY1 + j], &ffly_m0[j], sizeof(void*)*SHT_NTYP);
			memcpy(sht_func[SHT_M][SHT_FLY1 + j], &ffly_m[j], sizeof(void*)*SHT_NTYP);
		}
	} else {
		for (int j=0; j<=alg_lim; j++) {
			if (shtns->nthreads <= 2) {		// don't consider non-parallel algos for large number of threads.
				memcpy(sht_func[SHT_STD][SHT_FLY1 + j], &ffly[j], sizeof(void*)*SHT_NTYP);
			}
			memcpy(sht_func[SHT_M][SHT_FLY1 + j], &ffly_m[j], sizeof(void*)*SHT_NTYP);
		  #ifdef _OPENMP
			memcpy(sht_func[SHT_STD][SHT_OMP1 + j], &fomp_a[j], sizeof(void*)*SHT_NTYP);
			memcpy(sht_func[SHT_STD][SHT_OMP1A + j], &fomp_b[j], sizeof(void*)*SHT_NTYP);
			memcpy(sht_func[SHT_M][SHT_OMP1 + j], &ffly_m[j], sizeof(void*)*SHT_NTYP);		// no omp algo for SHT_M, use fly instead
		  #endif
		}
	}
	  #ifdef SHTNS_GPU
		for (int j=0; j<4; j++) {
			memcpy(sht_func[SHT_STD][SHT_GPU1+j], &fgpu[j], sizeof(void*)*SHT_NTYP);
		}
	  #endif

	set_sht_fly(shtns, 0);	// default transform is FLY
}


/// \internal returns an aproximation of the memory usage in mega bytes for the scalar matrices.
/// \ingroup init
static double sht_mem_size(int lmax, int mmax, int mres, int nlat)
{
	double s = 1./(1024*1024);
	s *= ((nlat+1)/2) * sizeof(double) * nlm_calc(lmax, mmax, mres);
	return s;
}

/// \internal return the number of shtns config that contain a reference to the memory location *pp.
static int ref_count(shtns_cfg shtns, void* pp)
{
	shtns_cfg s2 = sht_data;
	void* p;
	long int nref, offset;

	if ((pp==NULL) || (shtns==NULL)) return -1;		// error.

	p = *(void**)pp;			// the pointer to memory location that we want to free
	if (p == NULL) return 0;	// nothing to do.

	offset = (char*)pp - (char*)shtns;			// relative location in shtns_info structure.
	nref = 0;		// reference count.
	while (s2 != NULL) {		// scan all configs.
		if ( *(void**)(((char*)s2) + offset) == p ) nref++;
		s2 = s2->next;
	}
	return nref;		// will be >= 1 (as shtns is included in search)
}

/// \internal check if the memory location *pp is referenced by another sht config before freeing it.
/// returns the number of other references, or -1 on error. If >=1, the memory location could not be freed.
/// If the return value is 0, the ressource has been freed.
static int free_unused(shtns_cfg shtns, void* pp)
{
	int n = ref_count(shtns, pp);	// howmany shtns config do reference this memory location ?
	if (n <= 0) return n;		// nothing to free.
	if (n == 1) {				// no other reference found...
		void** ap = (void**) pp;
		free(*ap);		// ...so we can free it...
		*ap = NULL;		// ...and mark as unaloccated.
	}
	return (n-1);
}

/// \internal allocate arrays for SHT related to a given grid.
static void alloc_SHTarrays(shtns_cfg shtns)
{
	const int blk_sze = (VSIZE2 > 2) ? VSIZE2 : 2;
	const int sze = ((NLAT+blk_sze-1)/blk_sze)*blk_sze;		// align on vector
	shtns->ct = (double *) VMALLOC( sizeof(double) * sze*3 );			/// ct[] (including st and st_1)
	shtns->st = shtns->ct + sze;		shtns->st_1 = shtns->ct + 2*sze;

	if (verbose>1) printf("          Memory used for Ylm and Zlm matrices = %.3f Mb x2\n",3.0*sizeof(double)*NLM*NLAT_2/(1024.*1024.));
}


/// \internal free arrays allocated by alloc_SHTarrays.
static void free_SHTarrays(shtns_cfg shtns)
{
	if (ref_count(shtns, &shtns->ct) == 1)	VFREE(shtns->ct);
	shtns->ct = NULL;		shtns->st = NULL;
}

#ifndef HAVE_FFTW_COST
	// substitute undefined symbol in mkl and fftw older than 3.3
	#define fftw_cost(a) 0.0
#endif

/// \internal initialize FFTs using FFTW.
/// \param[in] layout defines the spatial layout (see \ref spat).
/// returns the number of double to be allocated for a spatial field.
static void planFFT(shtns_cfg shtns, int layout)
{
	double cost_fft_ip, cost_fft_oop, cost_ifft_ip, cost_ifft_oop;
	cplx *ShF;
	double *Sh;
	const int nfft = NPHI;
	int theta_inc, phi_inc;
	const int howmany = shtns->howmany;

	if (NPHI <= 2*MMAX) shtns_runerr("the sampling condition Nphi > 2*Mmax is not met.");

	#ifdef OMP_FFTW
		if ((shtns->fftw_plan_mode & (FFTW_EXHAUSTIVE | FFTW_PATIENT)) && (omp_threads > 1)) {
			shtns->fftw_plan_mode = FFTW_PATIENT;
			fftw_plan_with_nthreads(omp_threads);
		} else fftw_plan_with_nthreads(shtns->nthreads);
	#endif

	// default layout:
	phi_inc = shtns->nlat;
	if ((layout & SHT_ALLOW_PADDING) && NPHI>1) {	// handle padding
		int pad = 0;
		#ifndef SHTNS_GPU
		if ((phi_inc % 64 == 0) && (NPHI * phi_inc > 512))
			pad = 8;			// we add some padding, to avoid cache bank conflicts.
		#elif SHTNS_GPU==2
		const long stride_bytes = phi_inc * shtns->sizeof_real;
		if (NPHI * stride_bytes > 32*1024) {	// if the full fft does not fit in 32 kb (the L1 cache size)
			if (stride_bytes % 32)	pad = 32 - (stride_bytes % 32);		// always align on 32 bytes
			if ((stride_bytes+pad) % 8192 == 0) {
				pad += 64;		// avoid multiple of 8 kb
			} else {
				if (NPHI*(stride_bytes+pad) < 8192*1024) {		// full data (including padding) fits in L2
					if (stride_bytes % 64)  pad = 64 - (stride_bytes % 64);		// align on 64 bytes
					if ((stride_bytes+pad) % 128 == 0) pad += 64;		// add padding to avoid L2 cache bank / channel / whatever conflicts on AMD GPUs (large impact on performance).
				} else
				if ((2*MMAX+1)*stride_bytes < 8192*1024*1.2) {		// if an fft entirely fits in the L2 cache (8 Mb for AMD MI100 and MI200)
					if ((stride_bytes+pad) % 128 == 0) pad += 32;	// add padding to avoid L2 cache bank / channel / whatever conflicts on AMD GPUs (large impact on performance).
				}
			}
			pad /= shtns->sizeof_real;		// in units of real
		}
		#endif
		const char* env_pad = getenv("SHTNS_PAD");    if (env_pad) pad = atoi(env_pad);		// override default with SHTNS_PAD environment variable
		phi_inc += pad;
	}
	shtns->k_stride_a = 1;		shtns->m_stride_a = phi_inc;		// default strides
	shtns->nlat_padded = phi_inc;		// stride between phi in spectral domain
	shtns->nspat = NPHI * phi_inc * howmany;		// default spatial size to be allocated for a transform call
	shtns->spat_dist = NPHI * phi_inc;

	if (NPHI==1) 	// no FFT needed.
	{
		shtns->fft_mode = FFT_NONE;		// no FFT
		if (verbose) printf("        => no fft : Mmax=0, Nphi=1, Nlat=%d, Nbatch=%d\n",NLAT,howmany);
		return;
	}

	shtns->layout = layout;		// store the data-layout for future reference (by python interface).
	/* NPHI > 1 */
	theta_inc=1;	// SHT_NATIVE_LAYOUT is the default.
	if (layout & SHT_PHI_CONTIGUOUS) {
		phi_inc=1;  theta_inc=NPHI;
		shtns->nspat = NPHI * NLAT;		// no padding, no batching.
		#ifdef SHTNS_GPU
		if (shtns->mmax == NPHI/2) shtns->nspat += NLAT;		// on GPU, a little bit of extra space is needed for odd NPHI, to store the Fourier coefficients.
		#endif
		shtns->nlat_padded = NLAT;

		shtns->spat_dist = shtns->nspat;
		shtns->nspat *= howmany;
	}

	if (verbose) {
		printf("        => using FFTW : Mmax=%d, Nphi=%d, Nlat=%d, Nbatch=%d  ",MMAX,NPHI,NLAT, howmany);
		if (NPHI <= (SHT_NL_ORDER+1)*MMAX)	printf("     !! Warning : anti-aliasing condition Nphi > %d*Mmax is not met !\n", SHT_NL_ORDER+1);
		if (NPHI != fft_int(NPHI,7))		printf("     !! Warning : Nphi is not optimal for FFTW !\n");
	}

// Allocate dummy Spatial Fields.
	ShF = (cplx *) VMALLOC(shtns->nspat * sizeof(cplx));		// for complex-valued fields
	Sh = (double *) VMALLOC(shtns->nspat * sizeof(cplx));

	if (NLAT & 1) {		// odd nlat => c2r transforms
		if (verbose) printf("(odd layout: phi_inc=%d, theta_inc=%d)\n",phi_inc,theta_inc);
		if (howmany != 1) shtns_runerr("batch transform not supported for odd nlat\n");
		shtns->fft_mode = FFT_OOP | ((phi_inc==1) ? FFT_PHI_CONTIG_ODD : FFT_THETA_CONTIG_ODD);
		const int ncplx = NPHI/2 +1;
		shtns->fftc = fftw_plan_many_dft_r2c(1, &nfft, NLAT, Sh, &nfft, phi_inc, theta_inc, ShF, &ncplx, NLAT, 1, FFTW_ESTIMATE);
		shtns->ifftc = fftw_plan_many_dft_c2r(1, &nfft, NLAT, ShF, &ncplx, NLAT, 1, Sh, &nfft, phi_inc, theta_inc, FFTW_ESTIMATE);
	}
// complex fft for fly transform is a bit different.
	if (layout & SHT_PHI_CONTIGUOUS) {		// out-of-place split dft
		if ((NLAT & 1) == 0) {
			if (verbose) printf("(phi-contiguous layout: phi_inc=%d, theta_inc=%d)\n",phi_inc,theta_inc);
			fftw_iodim dim, many[2];
			shtns->fft_mode = FFT_PHI_CONTIG_SPLIT | FFT_OOP;
			dim.n = NPHI;    		dim.os = 1;				dim.is = NLAT;		// complex transpose
			many[0].n = NLAT/2;		many[0].os = 2*NPHI;	many[0].is = 2;
			many[1].n = howmany;		many[1].os = shtns->spat_dist;	many[1].is = shtns->spat_dist;
			shtns->ifftc = fftw_plan_guru_split_dft(1, &dim,  (howmany==1) ? 1 : 2, many, ((double*)ShF)+1, (double*)ShF, Sh+NPHI, Sh, shtns->fftw_plan_mode);

			// legacy analysis fft
			//dim.n = NPHI;    	dim.is = 1;			dim.os = NLAT;
			//many.n = NLAT/2;	many.is = 2*NPHI;	many.os = 2;
			// new internal
			dim.n = NPHI;    		dim.is = 1;				dim.os = 2;		// split complex, but without global transpose (faster).
			many[0].n = NLAT/2;		many[0].is = 2*NPHI;	many[0].os = 2*NPHI;
			shtns->fftc = fftw_plan_guru_split_dft(1, &dim, (howmany==1) ? 1 : 2, many,  Sh+NPHI, Sh, ((double*)ShF)+1, (double*)ShF, shtns->fftw_plan_mode);
			shtns->k_stride_a = NPHI;		shtns->m_stride_a = 2;
			shtns->nlat_padded = NLAT;

		/*	if (shtns->nthreads > 1) {
				fftw_plan_with_nthreads(1);
				// FOR MKL only:
				//fftw3_mkl.number_of_user_threads = shtns->nthreads;        // required to call the fft of mkl from multiple threads.
				// try to divide NLAT/2 into threads.
				int nblk = (NLAT/2) / shtns->nthreads;
				printf("omp block size (split) = %d\n", nblk);
				if (nblk * shtns->nthreads != NLAT/2) shtns_runerr("not divisible");

				dim.n = NPHI;    	dim.os = 1;			dim.is = NLAT;		// complex transpose
				many.n = nblk;		many.os = 2*NPHI;	many.is = 2;
				shtns->ifftc_block = fftw_plan_guru_split_dft(1, &dim, 1, &many, ((double*)ShF)+1, (double*)ShF, Sh+NPHI, Sh, shtns->fftw_plan_mode);

				dim.n = NPHI;    	dim.is = 1;			dim.os = 2;		// split complex, but without global transpose (faster).
				many.n = nblk;		many.is = 2*NPHI;	many.os = 2*NPHI;
				shtns->fftc_block = fftw_plan_guru_split_dft(1, &dim, 1, &many,  Sh+NPHI, Sh, ((double*)ShF)+1, (double*)ShF, shtns->fftw_plan_mode);
				fftw_plan_with_nthreads(shtns->nthreads);
			}	*/
		}

		// for complex transform it is much simpler (out-of-place):
		shtns->ifft_cplx = fftw_plan_many_dft(1, &nfft, NLAT, ShF, &nfft, NLAT, 1, (cplx*)Sh, &nfft, 1, NPHI, FFTW_BACKWARD, shtns->fftw_plan_mode);
		shtns->fft_cplx =  fftw_plan_many_dft(1, &nfft, NLAT, ShF, &nfft, 1, NPHI, (cplx*)Sh, &nfft, NLAT, 1, FFTW_BACKWARD, shtns->fftw_plan_mode);
	} else {	//if (layout & SHT_THETA_CONTIGUOUS) {		// use only in-place here, supposed to be faster.
		if ((NLAT & 1)==0) {
			if (verbose) printf("(theta-contiguous layout: phi_inc=%d, theta_inc=%d)\n",phi_inc,theta_inc);
			shtns->fft_mode = FFT_THETA_CONTIG;
			if (howmany==1) {
				shtns->ifftc = fftw_plan_many_dft(1, &nfft, shtns->nlat_2 * howmany, ShF, &nfft, phi_inc/2, 1, ShF, &nfft, phi_inc/2, 1, FFTW_BACKWARD, shtns->fftw_plan_mode);
			} else {
				fftw_iodim dim, many[2];
				dim.n = NPHI;    		dim.os = shtns->nlat_padded/2;		dim.is = shtns->nlat_padded/2;
				many[0].n = NLAT/2;		many[0].os = 1;				many[0].is = 1;
				many[1].n = howmany;	many[1].os = shtns->nlat_padded/2 * NPHI;	many[1].is = shtns->nlat_padded/2 * NPHI;
				shtns->ifftc = fftw_plan_guru_dft(1, &dim, 2, many, ShF, ShF, FFTW_BACKWARD, shtns->fftw_plan_mode);
			}
			shtns->fftc = shtns->ifftc;		// same thing, with m>0 and m<0 exchanged.

		/*	if (shtns->nthreads > 1) {
				fftw_plan_with_nthreads(1);
				// FOR MKL only:
				//fftw3_mkl.number_of_user_threads = shtns->nthreads;        // required to call the fft of mkl from multiple threads.
				// try to divide NLAT/2 into threads.
				int nblk = (NLAT/2) / shtns->nthreads;
				printf("omp block size = %d\n", nblk);
				if (nblk * shtns->nthreads != NLAT/2) shtns_runerr("not divisible");
				shtns->ifftc_block = fftw_plan_many_dft(1, &nfft, nblk, ShF, &nfft, NLAT/2, 1, ShF, &nfft, NLAT/2, 1, FFTW_BACKWARD, shtns->fftw_plan_mode);
				shtns->fftc_block = shtns->ifftc_block;		// same thing, with m>0 and m<0 exchanged.
				fftw_plan_with_nthreads(shtns->nthreads);
			}	*/
		}

		// complex-values spatial fields (in-place):
		shtns->ifft_cplx = fftw_plan_many_dft(1, &nfft, NLAT, ShF, &nfft, phi_inc, 1, ShF, &nfft, phi_inc, 1, FFTW_BACKWARD, shtns->fftw_plan_mode);
		shtns->fft_cplx = shtns->ifft_cplx;		// same thing, with m>0 and m<0 exchanged.
	}
	VFREE(Sh);		VFREE(ShF);

	if (verbose) {
		if (NPHI <= (SHT_NL_ORDER+1)*MMAX)	printf("     !! Warning : anti-aliasing condition Nphi > %d*Mmax is not met !\n", SHT_NL_ORDER+1);
		if (NPHI != fft_int(NPHI,7))		printf("     !! Warning : Nphi is not optimal for FFTW !\n");
	}
	if (verbose>1) printf("          fftw cost ifftc=%lg,  fftc=%lg\n",fftw_cost(shtns->ifftc), fftw_cost(shtns->fftc));
	if (verbose>2) {
		printf("\n *** fftc plan : \n");
		if (shtns->fftc) fftw_print_plan(shtns->fftc);
		printf("\n *** ifftc plan :\n");
		if (shtns->ifftc) fftw_print_plan(shtns->ifftc);
		printf("\n");
	}
}


/// \internal Sets the value tm[im] used for polar optimiation on-the-fly.
static void PolarOptimize(shtns_cfg shtns, double eps)
{
	for (int im=0;im<=MMAX;im++)	shtns->tm[im] = 0;

	if (eps > 0.0) {
		double y[LMAX+1];
		for (int im=1;im<=MMAX;im++) {
			int m = im*MRES;
			int it = shtns->tm[im-1] -1;	// tm[im] is monotonic.
			double v;
			do {
				it++;
				legendre_sphPlm_array(shtns, LMAX, im, shtns->ct[it], y+m);
				v = 0.0;
				for (int l=m; l<=LMAX; l++) {
					double ya = fabs(y[l]);
					if ( v < ya )	v = ya;
				}
			} while (v < eps);
			shtns->tm[im] = it;
		}
		if (verbose) printf("        + polar optimization threshold = %.1e\n",eps);
		if (verbose>1) {
			printf("          tm[im]=");
			int im=0;
			if (verbose <= 2  &&  MMAX >= 256) {	// don't print everything
				for (im=0;im<=50;im++)	printf(" %d",shtns->tm[im]);
				printf(" ...");
				im = MMAX-15;
			}
			for (;im<=MMAX;im++)	printf(" %d",shtns->tm[im]);
			printf("\n");
		}
	}
}

/// \internal Generate a grid (including weights)
static void grid_weights(shtns_cfg shtns, double latdir)
{
	long int it;
	double iylm_fft_norm;
	double xg[NLAT], stg[NLAT], wg[NLAT];	// gauss points and weights.
	const int overflow = 8*VSIZE2-1;
	const unsigned char grid = shtns->grid;

	shtns->wg = VMALLOC((NLAT_2 +overflow+VSIZE2) * sizeof(double));	// quadrature weights, double precision.
	shtns->wg += VSIZE2;	// reserve space before the weight array to store a normalization constant; to keep alignement, we reserve VSIZE2 doubles

	iylm_fft_norm = 1.0;	// FFT/SHT normalization for zlm (4pi normalized)
	if ((SHT_NORM != sht_fourpi)&&(SHT_NORM != sht_schmidt))  iylm_fft_norm = 4*M_PIl;	// FFT/SHT normalization for zlm (orthonormalized)
	iylm_fft_norm /= (2*NPHI);
	if (grid == GRID_GAUSS) {
		if (verbose) {
			printf("        => using Gauss nodes\n");
			if (2*NLAT <= (SHT_NL_ORDER +1)*LMAX) printf("     !! Warning : Gauss-Legendre anti-aliasing condition 2*Nlat > %d*Lmax is not met.\n",SHT_NL_ORDER+1);
		}
		gauss_nodes(xg,stg,wg,NLAT);	// generate gauss nodes and weights : ct = ]1,-1[ = cos(theta)
	} else if (grid == GRID_REGULAR) {
		if (verbose) {
			printf("        => using Regular nodes (Chebychev) with Fejer quadrature\n");
			if (NLAT <= (SHT_NL_ORDER +1)*LMAX) printf("     !! Warning : Regular-Fejer anti-aliasing condition Nlat > %d*Lmax is not met.\n",SHT_NL_ORDER+1);
		}
		fejer1_nodes(xg,stg,wg,NLAT);
	} else if (grid == GRID_POLES) {
		if (verbose) {
			printf("        => using Regular nodes including poles, with Clenshaw-Curtis quadrature\n");
			if (NLAT <= (SHT_NL_ORDER +1)*LMAX) printf("     !! Warning : Regular-Clenshaw-Curtis anti-aliasing condition Nlat > %d*Lmax is not met.\n",SHT_NL_ORDER+1);
		}
		clenshaw_curtis_nodes(xg,stg,wg,NLAT);
	} else shtns_runerr("unknown grid.");
	if (NLAT&1) wg[NLAT/2] *= 0.5;		// odd NLAT : adjust weigth of middle point.
	for (it=0; it<NLAT; it++) {
		shtns->ct[it] = latdir * xg[it];
		shtns->st[it] = stg[it];
		shtns->st_1[it] = 1.0/stg[it];
	}
	if (shtns->st[0] == 0.0)  shtns->st_1[0] = 0.0;
	if (shtns->st[NLAT-1] == 0.0)  shtns->st_1[NLAT-1] = 0.0;

	{	// *** perform some sanity checks, by computing simple integrals ***
		double s=0, x2=0, st2=0;
		for (long i=0; i<NLAT_2; i++) {		// sum symmetric contributions together (have same weights, increasing with i)
			int i2 = NLAT-1-i;
			s += wg[i] + wg[i2];						// sum of weights == 2
			x2 += wg[i]*xg[i]*xg[i] + wg[i2]*xg[i2]*xg[i2];		// integral of x2 == 2/3
			st2 += wg[i]*stg[i]*stg[i] + wg[i2]*stg[i2]*stg[i2];		// integral fo sin2(theta) == 4/3
		}
		// compute deviation from exact value:
		s = s - 2.0;
		x2 = x2*1.5 - 1.;
		st2 = st2*0.75 - 1.;
		if (verbose>1) {
			printf("          Sum of weights = 2 + %g (should be 2)\n", s);
			printf("          Applying quadrature rule to 3/2.x^2 = 1 + %g (should be 1)\n", x2);
			printf("          Applying quadrature rule to 3/4.sin2(theta) = 1 + %g (should be 1)\n", st2);
		} else if (fabs(s)+fabs(x2)+fabs(st2) > 1e-14)	shtns_runerr("Bad quadrature accuracy.");
	}

	shtns->wg[-1] = 1.0/iylm_fft_norm;		// store the inverse of the norm included in gauss weights
	for (it=0; it<NLAT_2; it++)
		shtns->wg[it] = wg[it]*iylm_fft_norm;		// faster double-precision computations.
	for (it=NLAT_2; it < NLAT_2 +overflow; it++) shtns->wg[it] = 0.0;		// padding for multi-way algorithm.

	if ((verbose>1) && (grid == GRID_GAUSS)) {
		printf(" NLAT=%d, NLAT_2=%d\n",NLAT,NLAT_2);
	// TEST if gauss points are ok.
		double tmax = 0.0;
		for (it = 0; it<NLAT_2; it++) {
			double t = legendre_Pl(NLAT, shtns->ct[it]);
			if (t>tmax) tmax = t;
	//		printf("i=%d, x=%12.12g, p=%12.12g\n",it,ct[it],t);
		}
		printf("          max zero at Gauss nodes for Pl[l=NLAT] : %g\n",tmax);
		if (NLAT_2 < 100) {
			printf("          Gauss nodes :");
			for (it=0;it<NLAT_2; it++)
				printf(" %g",shtns->ct[it]);
			printf("\n");
		}
	}
}


/* TEST AND TIMING FUNCTIONS */

/// check if an IEEE754 double precision number is finite (works also when compiled with -ffinite-math).
static int isNotFinite(double x) {
	union { double d; unsigned long long i; } mem = { x };
	return (mem.i & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL;		// nan or inf
}

/// \internal return the max error for a back-and-forth SHT transform.
/// this function is used to internally measure the accuracy.
double SHT_error(shtns_cfg shtns, int vector)
{
	cplx *Tlm0=0, *Slm0=0, *Tlm=0, *Slm=0;
	double *Sh=0, *Th=0;
	double t, tmax, n2,  err;
	long int i, jj, nlm_cplx;
	
	srand( 42 );	// init random numbers.
	
	const size_t spec_sze = sizeof(cplx) * shtns->spec_dist * shtns->howmany;
	Slm0 = (cplx *) VMALLOC(spec_sze);
	Slm = (cplx *) VMALLOC(spec_sze);
	Sh = (double *) VMALLOC( NSPAT_ALLOC(shtns) * sizeof(double) );
	if ((Sh==0) || (Slm==0) || (Slm0==0)) shtns_runerr("not enough memory.");
	if (vector) {
		Tlm0 = (cplx *) VMALLOC(spec_sze);
		Tlm = (cplx *) VMALLOC(spec_sze);
		Th = (double *) VMALLOC( NSPAT_ALLOC(shtns) * sizeof(double) );
		if ((Th==0) || (Tlm==0) || (Tlm0==0)) shtns_runerr("not enough memory.");
	}

// m = nphi/2 is also real if nphi is even.
	nlm_cplx = ( MMAX*2 == NPHI ) ? LiM(shtns, MRES*MMAX,MMAX) : NLM;
	t = 1.0 / (RAND_MAX/2);
	for (i=0; i<NLM; i++) {
		if ((i<=LMAX)||(i>=nlm_cplx)) {		// m=0 or m*2=nphi : real random data
			Slm0[i] = t*((double) (rand() - RAND_MAX/2));
			if (vector) Tlm0[i] = t*((double) (rand() - RAND_MAX/2));
		} else {							// m>0 : complex random data
			Slm0[i] = t*((double) (rand() - RAND_MAX/2)) + I*t*((double) (rand() - RAND_MAX/2));
			if (vector) Tlm0[i] = t*((double) (rand() - RAND_MAX/2)) + I*t*((double) (rand() - RAND_MAX/2));
		}
	}

	SH_to_spat(shtns, Slm0,Sh);		// scalar SHT
	spat_to_SH(shtns, Sh, Slm);
	for (i=0, tmax=0., n2=0., jj=0; i<NLM; i++) {		// compute error
		t = cabs(Slm[i] - Slm0[i]);
		n2 += t*t;
		if (t>tmax) { tmax = t; jj = i; }
	}
	err = tmax;
	if (verbose>1) printf("        scalar SH - poloidal   rms error = %.3g  max error = %.3g for l=%hu,lm=%ld\n",sqrt(n2/NLM),tmax,shtns->li[jj],jj);

	if (vector) {
		for (i=1; i<NLM; i++) {
			double nrm = 1.0/shtns->li[i];		// approx unit norm for vector with unit energy in each mode
			Tlm0[i] *= nrm;		Slm0[i] *= nrm;
		}
		Slm0[0] = 0.0; 	Tlm0[0] = 0.0;		// l=0, m=0 n'a pas de signification sph/tor
		SHsphtor_to_spat(shtns, Slm0, Tlm0, Sh, Th);		// vector SHT
		spat_to_SHsphtor(shtns, Sh, Th, Slm, Tlm);
		for (i=0, tmax=0., n2=0., jj=0; i<NLM; i++) {		// compute error
			t = cabs(Slm[i] - Slm0[i]);
			if (i>0) t *= shtns->li[i];		// relative error: account for mean spectrum of unit energy
			n2 += t*t;
			if (t>tmax) { tmax = t; jj = i; }
		}
		if (tmax > err) err = tmax;
		if (verbose>1) printf("        vector SH - spheroidal rms error = %.3g  max error = %.3g for l=%hu,lm=%ld\n",sqrt(n2/NLM),tmax,shtns->li[jj],jj);
		for (i=0, tmax=0., n2=0., jj=0; i<NLM; i++) {		// compute error
			t = cabs(Tlm[i] - Tlm0[i]);
			if (i>0) t *= shtns->li[i];		// relative error: account for mean spectrum of unit energy
			n2 += t*t;
			if (t>tmax) { tmax = t; jj = i; }
		}
		if (tmax > err) err = tmax;
		if (verbose>1) printf("                  - toroidal   rms error = %.3g  max error = %.3g for l=%hu,lm=%ld\n",sqrt(n2/NLM),tmax,shtns->li[jj],jj);

		//for (int i=0; i<NLM; i++) {
		//	printf("l=%d err=%.3g %.3g \t %g,%g (%g,%g) \t %g,%g (%g,%g)\n",shtns->li[i], cabs(Slm[i] - Slm0[i]), cabs(Tlm[i] - Tlm0[i]), creal(Slm[i]),cimag(Slm[i]), creal(Slm0[i]),cimag(Slm0[i]), creal(Tlm[i]),cimag(Tlm[i]), creal(Tlm0[i]),cimag(Tlm0[i]));
		//}
	}

	if (Th) VFREE(Th);    if (Tlm) VFREE(Tlm);    if (Tlm0) VFREE(Tlm0);
	VFREE(Sh);  VFREE(Slm);  VFREE(Slm0);
	return(err);		// return max error.
}


/// \internal measure time used for a transform function
static double get_time(shtns_cfg shtns, int nloop, int npar, char* name, void *fptr, void *i1, void *i2, void *i3, void *o1, void *o2, void *o3, int l)
{
	double t;
	int i;
	ticks tik0, tik1;

	if (fptr == NULL) return(0.0);

	tik1 = getticks();
	for (i=0; i<nloop; i++) {
		switch(npar) {
			case 2: (*(pf2l)fptr)(shtns, i1,o1, l); break;			// l may be discarded.
			case 3: (*(pf3l)fptr)(shtns, i1,o1,o2, l); break;
			case 4: (*(pf4l)fptr)(shtns, i1,i2,o1,o2, l); break;
			default: (*(pf6l)fptr)(shtns, i1,i2,i3, o1,o2,o3, l); break;
		}
		if (i==0) tik0 = getticks();
	}
	if (nloop == 1) {
		t = elapsed(tik0, tik1);
	} else {
		tik1 = getticks();
		t = elapsed(tik1, tik0)/(nloop-1);		// discard first iteration.
	}
	if (verbose>1) {  printf("  t(%s) = %.3g",name,t);	fflush(stdout);  }
	return t;
}


/// \internal choose fastest between on-the-fly and gauss algorithms.
/// *nlp is the number of loops. If zero, it is set to a good value.
/// returns time without dct / best time with dct (or 0 if no dct available).
static void choose_best_sht(shtns_cfg shtns, int* nlp, int vector)
{
	cplx *Qlm=0, *Slm=0, *Tlm=0;
	double *Qh=0, *Sh=0, *Th=0;
	int m, i, i0, minc, nloop, alg_end;
	int typ_lim = SHT_NTYP;		// time every type.
	double t0, t, tt, r;
	double tdct, tnodct;
	clock_t tcpu;
	const int on_the_fly_only = 1;		// only on-the-fly.
	int otf_analys = (shtns->wg != NULL);			// on-the-fly analysis supported.

	if (NLAT < VSIZE2*4) return;			// on-the-fly not possible for NLAT_2 < 2*NWAY (overflow).

	size_t nspat = sizeof(double) * NSPAT_ALLOC(shtns);
	size_t nspec = sizeof(cplx)* shtns->spec_dist * shtns->howmany;
	if (nspec>nspat) nspat=nspec;
	Sh = (double *) VMALLOC(nspat);		Slm = (cplx *) VMALLOC(nspec);
	if ((Sh==0) || (Slm==0)) shtns_runerr("not enough memory.");
	if (vector) {
		Th = (double *) VMALLOC(nspat);				Qh = (double *) VMALLOC(nspat);
		Tlm = (cplx *) VMALLOC(nspec);	Qlm = (cplx *) VMALLOC(nspec);
		if ( (Th==0) || (Qh==0) || (Tlm==0) || (Qlm==0) ) vector = 0;
	}

	for (i=0;i<NLM;i++) {
		const int l = shtns->li[i];
		const double l_2_ = shtns->l_2[l];
		Slm[i] = l_2_ + 0.5*l_2_*I;
		if (vector) {
			Tlm[i] = 0.5*l_2_ + l_2_*I;
			Qlm[i] = 3.*l_2_ + 2.*l_2_*I;
		}
	}

	if (verbose) {
		printf("        finding optimal algorithm");	fflush(stdout);
	}

	const int ref_alg = (shtns->nthreads == 1) ? SHT_FLY2 : SHT_OMP2;
	if (*nlp <= 0) {
		// find good nloop by requiring less than 3% difference between 2 consecutive timings.
		m=0;	nloop = 1;                     // number of loops to get timings.
		r = 0.0;	tt = 1.0;
		do {
			if ((r > 0.03)||(tt<0.1)) {
				m = 0;		nloop *= 3;
			} else 	m++;
			tcpu = clock();
			t0 = get_time(shtns, nloop, 2, "", sht_func[SHT_STD][ref_alg][SHT_TYP_SSY], Slm, Tlm, Qlm, Sh, Th, Qh, LMAX);
			tcpu = clock() - tcpu;		tt = 1.e-6 * tcpu;
			if (tt >= SHT_TIME_LIMIT) break;			// we should not exceed some time-limit
			t = get_time(shtns, nloop, 2, "", sht_func[SHT_STD][ref_alg][SHT_TYP_SSY], Slm, Tlm, Qlm, Sh, Th, Qh, LMAX);
			r = fabs(2.0*(t-t0)/(t+t0));
			if (verbose>1) {
				printf(", nloop=%d, r=%g, m=%d (real time = %g s)\n",nloop,r,m,tt);
				if (tt >= 0.01) break;		// faster timing in debug mode.
			}
			if (verbose==1) {	printf(".");	fflush(stdout);	}
		} while((nloop<10000)&&(m < 3));
		*nlp = nloop;
	} else {
		nloop = *nlp;
	}
	if (verbose>1) printf(" => nloop=%d (takes %g s)\n",nloop, tt);
	if (vector == 0)	typ_lim = SHT_TYP_VSY;		// time only scalar transforms.
//	if (tt > 3.0)		typ_lim = SHT_TYP_VSY;		// time only scalar transforms.
//	if (tt > 10.0)	goto done;		// timing this will be too slow...

	int ityp = 0;	do {
		if (ityp == 2) nloop = (nloop+1)/2;		// scalar ar done.
		t0 = 1e100;
		i0 = SHT_MEM;
		if (on_the_fly_only) i0 = SHT_SV;		// only on-the-fly (SV is then also on-the-fly)
		alg_end = SHT_NALG;
		if (shtns->nthreads <= 1) alg_end = SHT_OMP1;		// no OpenMP with 1 thread.
		if ((ityp&1) && (otf_analys == 0)) alg_end = SHT_FLY1;		// no on-the-fly analysis for regular grid.
		for (i=i0, m=0;	i<alg_end; i++) {
			if (sht_func[0][i][ityp] != NULL) m++;		// count number of algos
		}
		if (m >= 2) {		// don't time if there is only 1 algo !
			if (verbose>1) {  printf("finding best %s ...",sht_type[ityp]);	fflush(stdout);  }
			i = i0-1;		i0 = -1;
			while (++i < alg_end) {
				void *pf = sht_func[0][i][ityp];
				if (pf != NULL) {
					if (ityp&1) {	// analysis
						t = get_time(shtns, nloop, sht_npar[ityp], sht_name[i], pf, Sh, Th, Qh, Slm, Tlm, Qlm, LMAX);
					} else {
						t = get_time(shtns, nloop, sht_npar[ityp], sht_name[i], pf, Slm, Tlm, Qlm, Sh, Th, Qh, LMAX);
					}
					if (i < SHT_FLY1) t *= 1.03;	// 3% penality for memory based transforms.
				#ifdef _OPENMP
					if ((shtns->nthreads > 1) && ((i >= SHT_OMP1)||(i == SHT_SV))) t *= 1.3;	// 30% penality for openmp transforms.
				#endif
					if (t < t0) {	i0 = i;		t0 = t;		if (verbose>1) printf("*");	}
				}
			}
			if (i0 >= 0) {
				for (int iv=0; iv<SHT_NVAR; iv++) {
					if (sht_func[iv][i0][ityp]) shtns->ftable[iv][ityp] = sht_func[iv][i0][ityp];
					if (ityp == 4) {		// only one timing for both gradients variants.
						if (sht_func[iv][i0][ityp+1]) shtns->ftable[iv][ityp+1] = sht_func[iv][i0][ityp+1];
					}
				}
				if (verbose==1) {	printf(".");	fflush(stdout);	}
				if (verbose>1) printf(" => %s\n",sht_name[i0]);
			}
		}
		if (ityp == 4) ityp++;		// skip second gradient
	} while(++ityp < typ_lim);

done:
	if (verbose) printf("\n");
	if (Qlm) VFREE(Qlm);		if (Tlm) VFREE(Tlm);
	if (Qh)  VFREE(Qh);			if (Th)  VFREE(Th);
	if (Slm) VFREE(Slm);	 	if (Sh)  VFREE(Sh);
}


const char* shtns_get_build_info() {
	const int nmax=159;
	static char s[160];	// a reasonable buffer size, set to nmax+1
	int n = snprintf(s, nmax,
  #ifndef SHTNS4MAGIC
	"[SHTns " SHTNS_VER "] built "
  #else
	"[SHTns " SHTNS_VER "] built for MagIC "
  #endif
	__DATE__ ", " __TIME__  ", id: ");
  #ifdef SHTNS_GIT
	if (strlen(SHTNS_GIT) > 0) n += snprintf(s+n, nmax-n, SHTNS_GIT ",");
  #endif
	n += snprintf(s+n, nmax-n, _SHTNS_ID_);
  #ifdef _OPENMP
	n += snprintf(s+n, nmax-n, ",openmp");
  #endif
  #ifdef SHTNS_GPU
	snprintf(s+n, nmax-n, (SHTNS_GPU == 2) ? ",hip" : ",cuda");
  #endif
	s[nmax]=0;
	return s;
}

void shtns_print_version() {
	printf("%s\n",shtns_get_build_info());
}


void fprint_ftable(FILE* fp, void* ftable[SHT_NVAR][SHT_NTYP])
{
	for (int iv=0; iv<SHT_NVAR; iv++) {
		fprintf(fp, "\n  %4s:",sht_var[iv]);
		void** f = ftable[iv];
		for (int it=0; it<SHT_NTYP; it++) {
			if (f[it] != NULL) {
				for (int ia=0; ia<SHT_NALG; ia++)
					if (sht_func[iv][ia][it] == f[it]) {
						fprintf(fp, "%5s ",sht_name[ia]);	break;
					}
			} else  fprintf(fp, " none ");
		}
	}
}

void shtns_print_cfg(shtns_cfg shtns)
{
	printf("Lmax=%d, Mmax*Mres=%d, Mres=%d, Nlm=%d  [%d threads, ",LMAX, MMAX*MRES, MRES, NLM, shtns->nthreads);
	#ifdef SHTNS_GPU
		if (shtns->d_clm) printf("gpu fp%d/fp%d, ",shtns->sizeof_real*8, shtns->sizeof_real_g*8);
	#endif
	if (shtns->norm & SHT_REAL_NORM) printf("'real' norm, ");
	if (shtns->norm & SHT_NO_CS_PHASE) printf("no Condon-Shortley phase, ");
	if (shtns->robert_form) printf("Robert form, ");
	if (SHT_NORM == sht_fourpi) printf("4.pi normalized]\n");
	else if (SHT_NORM == sht_schmidt) printf("Schmidt semi-normalized]\n");
	else printf("orthonormalized]\n");
	if (shtns->ct == NULL)	return;		// no grid is set

	switch(shtns->grid) {
		case GRID_GAUSS : printf("Gauss grid");	 break;
		case GRID_REGULAR : printf("Regular grid");	 break;
		case GRID_POLES : printf("Regular grid including poles");  break;
		default : printf("Unknown grid");
	}
	printf(" : Nlat=%d, Nphi=%d, Nbatch=%d\n", NLAT, NPHI, shtns->howmany);
	printf("      ");
	for (int it=0; it<SHT_NTYP; it++)
		printf("%5s ",sht_type[it]);
	fprint_ftable(stdout, shtns->ftable);
	printf("\n");
	#ifdef SHTNS_GPU
		if (shtns->d_clm) printf("gpu cfg: %s\n", cushtns_get_cfg_info(shtns));
	#endif
}


/// \internal saves config to a file for later restart.
int config_save(shtns_cfg shtns, int req_flags)
{
	int err = 0;
	
	if (shtns->ct == NULL) return -1;		// no grid set

	if (shtns->nphi > 1) {
		FILE* f = fopen("shtns_cfg_fftw","w");
		if (f) {
			fftw_export_wisdom_to_file(f);
			fclose(f);
		} else err -= 2;
	}

	FILE *fcfg = fopen("shtns_cfg","a");
	if (fcfg != NULL) {
		fprintf(fcfg, "%s %s %d %d %d %d %d %d %d %d %d %d",SHTNS_VER, _SHTNS_ID_, shtns->lmax, shtns->mmax, shtns->mres, shtns->nphi, shtns->nlat, shtns->grid, shtns->nthreads, req_flags, shtns->nlorder, -1);
		fprint_ftable(fcfg, shtns->ftable);
		fprintf(fcfg,"\n");
		fclose(fcfg);
	} else err -= 4;

	if (verbose && err < 0) fprintf(stderr,"! Warning ! SHTns could not save config\n");
	return err;
}

/// \internal try to load config from a file 
int config_load(shtns_cfg shtns, int req_flags)
{
	void* ft2[SHT_NVAR][SHT_NTYP];		// pointers to transform functions.
	int lmax2, mmax2, mres2, nphi2, nlat2, grid2, nthreads2, req_flags2, nlorder2, mtr_dct2;
	int found = 0;
	char version[32], simd[32], alg[8];

	if (shtns->ct == NULL) return -1;		// no grid set

	if ((req_flags & 255) == sht_quick_init) req_flags += sht_gauss - sht_quick_init;		// quick_init uses gauss.

	FILE *fcfg = fopen("shtns_cfg","r");
	if (fcfg != NULL) {
		while(1) {
			int i=fscanf(fcfg, "%30s %30s %d %d %d %d %d %d %d %d %d %d",version, simd, &lmax2, &mmax2, &mres2, &nphi2, &nlat2, &grid2, &nthreads2, &req_flags2, &nlorder2, &mtr_dct2);
			if (i<12) break;
			for (int iv=0; iv<SHT_NVAR; iv++) {
				if ( fscanf(fcfg, "%7s", alg) == 0 ) break;
				for (int it=0; it<SHT_NTYP; it++) {
					if ( fscanf(fcfg, "%7s", alg) == 0 ) break;
					ft2[iv][it] = 0;
					for (int ia=0; ia<SHT_NALG; ia++) {
						if (strcmp(alg, sht_name[ia]) == 0) {
							ft2[iv][it] = sht_func[iv][ia][it];
							break;
						}
					}
				}
			}
			if (feof(fcfg)) break;
			if ((shtns->lmax == lmax2) && (shtns->mmax == mmax2) && (shtns->mres == mres2) && (shtns->nthreads == nthreads2) &&
			  (shtns->nphi == nphi2) && (shtns->nlat == nlat2) && (shtns->grid == grid2) &&  (req_flags == req_flags2) &&
			  (shtns->nlorder == nlorder2) && (strcmp(simd, _SHTNS_ID_)==0)) {
				if (verbose > 0) printf("        + using saved config\n");
				if (verbose > 1) {
					fprint_ftable(stdout, ft2);
					printf("\n");
				}
				for (int iv=0; iv<SHT_NVAR; iv++)
				for (int it=0; it<SHT_NTYP; it++)
					if (ft2[iv][it]) shtns->ftable[iv][it] = ft2[iv][it];		// accept only non-null pointer
				found = 1;
				break;
			}
		}
		fclose(fcfg);
		return found;
	} else {
			if (verbose) fprintf(stderr,"! Warning ! SHTns could not load config\n");
		return -2;		// file not found
	}
}

/// \internal returns 1 if val cannot fit in dest (unsigned)
#define IS_TOO_LARGE(val, dest) (sizeof(dest) >= sizeof(val)) ? 0 : ( ( val >= (1ULL<<(8*sizeof(dest))) ) ? 1 : 0 )

/// \internal returns the size that must be allocated for an shtns_info.
#define SIZEOF_SHTNS_INFO(mmax) ( sizeof(struct shtns_info) + (mmax+1)*( sizeof(unsigned short) ) )

/* PUBLIC INITIALIZATION & DESTRUCTION */

/** \addtogroup init Initialization functions.
*/
///@{

/*! This sets the description of spherical harmonic coefficients.
 * It tells SHTns how to interpret spherical harmonic coefficient arrays, and it sets usefull arrays.
 * Returns the configuration to be passed to subsequent transform functions, which is basicaly a pointer to a \ref shtns_info struct.
 * \param lmax : maximum SH degree that we want to describe.
 * \param mmax : number of azimutal wave numbers.
 * \param mres : \c 2.pi/mres is the azimutal periodicity. \c mmax*mres is the maximum SH order.
 * \param norm : define the normalization of the spherical harmonics (\ref shtns_norm)
 * + optionaly disable Condon-Shortley phase (ex: \ref sht_schmidt | \ref SHT_NO_CS_PHASE)
 * + optionaly use a 'real' normalization (ex: \ref sht_fourpi | \ref SHT_REAL_NORM)
*/
shtns_cfg shtns_create(int lmax, int mmax, int mres, enum shtns_norm norm)
{
	shtns_cfg shtns, s2;

//	if (lmax < 1) shtns_runerr("lmax must be larger than 1");
	if (lmax < 2) shtns_runerr("lmax must be at least 2");
	if (IS_TOO_LARGE(lmax, shtns->lmax)) shtns_runerr("lmax too large");
	if (mmax*mres > lmax) shtns_runerr("MMAX*MRES should not exceed LMAX");
	if (mres <= 0) shtns_runerr("MRES must be > 0");

	// allocate new setup and initialize some variables (used as flags) :
	const size_t sze = SIZEOF_SHTNS_INFO(mmax);
	shtns = VMALLOC( sze );		// aligned on a cache line: the public part fits in a single cache line
	if (shtns == NULL) return shtns;	// FAIL
	{
		memset(shtns, 0, sze);		// zero initialize everything!
		shtns->tm = (unsigned short*) (shtns + 1);	// tm is stored at the end of the struct...
		shtns->ct = NULL;	shtns->st = NULL;
		shtns->ylm_lat = NULL;	shtns->ct_lat = 2.0;	shtns->ifft_lat = NULL;		shtns->nphi_lat = 0;	// _to_lat data
		shtns->mx_stdt = NULL;	// marks vector transforms as disabled.
		#ifdef SHTNS_GPU
		shtns->d_clm = NULL;		// this marks the gpu as disabled.
		#endif
		#ifdef SHTNS4MAGIC
		shtns->robert_form = 1;		// Robert form by default for MagIC (multiply spatial vector fields by sin(theta))
		#else
		shtns->robert_form = 0;		// no Robert form by default.
		#endif
		shtns->howmany = 1;		// 1 transform by default. Use shtns_set_many() to ask for more.
		shtns->cpu_timer = -1;		// disable timing by default
	}

	shtns->norm = norm;
	const int with_cs_phase = (norm & SHT_NO_CS_PHASE) ? 0 : 1;		/// Condon-Shortley phase (-1)^m is used by default.
	const double mpos_renorm = (norm & SHT_REAL_NORM) ? 0.5 : 1.0;	/// renormalization of m>0.

	// copy sizes.
	shtns->mmax = mmax;		shtns->mres = mres;		shtns->lmax = lmax;
	shtns->nlm = nlm_calc(lmax, mmax, mres);
	shtns->spec_dist = shtns->nlm;	// default value.
	shtns->nlm_cplx = 2*shtns->nlm - (lmax+1);	// = nlm_cplx_calc(lmax, mmax, mres);
	shtns->nthreads = omp_threads;
	if (omp_threads > mmax+1) shtns->nthreads = mmax+1;	// limit the number of threads to mmax+1
	shtns_verbose(-1);	// -1: check if environment variable sets verbosity level
	if (verbose) {
		shtns_print_version();
		printf("        ");		shtns_print_cfg(shtns);
	}

	int larrays_ok=0, legendre_ok=0, l_2_ok=0;
	s2 = sht_data;		// check if some data can be shared ...
	while(s2 != NULL) {
		if ((s2->mmax >= mmax) && (s2->mres == mres)) {
			if (s2->lmax == lmax) {		// we can reuse the l-related arrays (li)
				shtns->li = s2->li;		shtns->mi = s2->mi;
				larrays_ok = 1;
				if (s2->norm == norm) {		// we can reuse the legendre tables.
					shtns->alm = s2->alm;
					shtns->alm2 = s2->alm2;		shtns->glm = s2->glm;		shtns->glm_analys = s2->glm_analys;
					#ifdef SHTNS_ISHIOKA
					shtns->xlm = s2->xlm;		shtns->x2lm = s2->x2lm;
					shtns->clm = s2->clm;
					#endif
					legendre_ok = 1;
				}
			}
		}
		if (s2->lmax >= lmax) {		// we can reuse l_2
			shtns->l_2 = s2->l_2;
			l_2_ok = 1;
		}
		s2 = s2->next;
	}
	if (larrays_ok == 0) {
		// alloc spectral arrays
		shtns->li = (unsigned short *) malloc( 2*NLM*sizeof(unsigned short) );	// NLM defined at runtime.
		shtns->mi = shtns->li + NLM;
		long lm=0;
		for (int im=0; im<=MMAX; im++) {	// init l-related arrays.
			int m = im*MRES;
			for (int l=im*MRES;l<=LMAX;l++) {
				shtns->li[lm] = l;		shtns->mi[lm] = m;
				lm++;
			}
		}
		if (lm != NLM) shtns_runerr("unexpected error");
	}
	if (legendre_ok == 0) {	// this precomputes some values for the legendre recursion.
		if (verbose>1) printf("        > Condon-Shortley phase = %d, normalization = %d\n", with_cs_phase, SHT_NORM);
		legendre_precomp(shtns, SHT_NORM, with_cs_phase, mpos_renorm);
	}
	if (l_2_ok == 0) {
		shtns->l_2 = (double *) malloc( (LMAX+1)*sizeof(double) );
		shtns->l_2[0] = 0.0;	// undefined for l=0 => replace with 0.
		real one = 1.0;
		for (int l=1; l<=LMAX; l++)		shtns->l_2[l] = one/(l*(l+1));
	}

	switch(SHT_NORM) {
		case sht_for_rotations:
		case sht_schmidt:
			shtns->Y00_1 = 1.0;		shtns->Y10_ct = 1.0;
			break;
		case sht_fourpi:
			shtns->Y00_1 = 1.0;		shtns->Y10_ct = sqrt(1./3.);
			break;
		case sht_orthonormal:
		default:
			shtns->Y00_1 = sqrt(4.*M_PI);		shtns->Y10_ct = sqrt(4.*M_PI/3.);
//			Y11_st = sqrt(2.*M_PI/3.);		// orthonormal :  \f$ \sin\theta\cos\phi/(Y_1^1 + Y_1^{-1}) = -\sqrt{2 \pi /3} \f$
	}
	shtns->mpos_scale_analys = 0.5/mpos_renorm;
	shtns->Y11_st = shtns->Y10_ct * sqrt(0.5/mpos_renorm);
	if (with_cs_phase)	shtns->Y11_st *= -1.0;		// correct Condon-Shortley phase

// initialize rotations along arbitrary axes (if applicable).
	if ((lmax == mmax) && (mres == 1))	SH_rotK90_init(shtns);

// save a pointer to this setup and return.
	shtns->next = sht_data;		// reference of previous setup (may be NULL).
	sht_data = shtns;			// keep track of new setup.
	return(shtns);
}

/// Copy a given config but allow a different (smaller) mmax and the possibility to enable/disable fft (beta).
shtns_cfg shtns_create_with_grid(shtns_cfg base, int mmax, int nofft)
{
	shtns_cfg shtns;

	if (mmax > base->mmax) return (NULL);		// fail if mmax larger than source config.

	shtns = VMALLOC( SIZEOF_SHTNS_INFO(mmax) );			// align on cache line
	memcpy(shtns, base, SIZEOF_SHTNS_INFO(mmax) );		// copy all
	shtns->tm = (unsigned short*) (shtns+1);		// tm is stored at the end of the struct.

	if (mmax != shtns->mmax) {
		shtns->mmax = mmax;
		for (int im=0; im<=mmax; im++) {
			shtns->tm[im] = base->tm[im];
		}
		if (mmax == 0) {
			// TODO we may disable fft and replace with a phi-averaging function ...
			// ... then switch to axisymmetric functions :
			// init_sht_array_func(shtns);
			// choose_best_sht(shtns, &nloop, 0);
		}
	}
	if (nofft != 0) {
		shtns->fft_mode = FFT_NONE;		// fft disabled.
	}

// save a pointer to this setup and return.
	shtns->next = sht_data;		// reference of previous setup (may be NULL).
	sht_data = shtns;			// keep track of new setup.
	return(shtns);
}

/// release all resources allocated by a grid.
void shtns_unset_grid(shtns_cfg shtns)
{
	if (ref_count(shtns, &shtns->wg) == 1)	VFREE(shtns->wg - VSIZE2);
	shtns->wg = NULL;
	free_SHTarrays(shtns);
	shtns->nlat = 0;	shtns->nlat_2 = 0;
	shtns->nphi = 0;	shtns->nspat = 0;
}

/// release all resources allocated by a given shtns_cfg. NOT thead-safe.
void shtns_destroy(shtns_cfg shtns)
{
	shtns_cfg s2 = sht_data;
	while (s2 != shtns) {
		if (s2 == 0) 	return;	// shtns not found in list! already freed?
		s2 = s2->next;
	}
	#ifdef SHTNS_GPU
	if (shtns->d_clm) cushtns_release_gpu(shtns);
	#endif
	free_unused(shtns, &shtns->l_2);
	free_unused(shtns, &shtns->alm);
	free_unused(shtns, &shtns->li);
	free_unused(shtns, &shtns->ct_rot);
	#ifdef SHTNS_ISHIOKA
	free_unused(shtns, &shtns->clm);
	#endif
	free_unused(shtns, &shtns->glm);
	if (shtns->fft_rot)  fftw_destroy_plan(shtns->fft_rot);
	if (shtns->ifftc) fftw_destroy_plan(shtns->ifftc);
	if (shtns->fftc != shtns->ifftc) fftw_destroy_plan(shtns->fftc);
	if (shtns->ifft_cplx) fftw_destroy_plan(shtns->ifft_cplx);
	if (shtns->fft_cplx != shtns->ifft_cplx) fftw_destroy_plan(shtns->fft_cplx);

	if (shtns->mx_van == shtns->mx_stdt) shtns->mx_van = NULL;
	else  free_unused(shtns, &shtns->mx_van);
	free_unused(shtns, &shtns->mx_stdt);

	shtns_unset_grid(shtns);

	shtns_cfg* p = &sht_data;
	while (*p != NULL) {
		if (*p == shtns) {
			*p = shtns->next;		// forget shtns
			break;
		}
		p = &((*p)->next);
	}
	VFREE(shtns);
}

/// clear all allocated memory (hopefully) and go back to 0 state. NOT thread-safe.
void shtns_reset() {
	while (sht_data != NULL) 	shtns_destroy(sht_data);
}

#ifndef SHTNS_GPU
// allocation for vector-aligned data. If gpu is enabled, these are replace by pinned memory allocation.
void* shtns_malloc(size_t size) {
	return VMALLOC(size);
}

void shtns_free(void* p) {
	VFREE(p);
}
#endif


static int choose_nlat(int n)
{
	#ifdef SHTNS_GPU
	n = ((n+3)/4) * 4;		// multiple of 4 for GPUs
	#else
	n += (n&1);		// even is better.
	#endif

	#ifndef SHTNS4MAGIC
	n = ((n+(VSIZE2-1))/VSIZE2) * VSIZE2;		// multiple of vector size
	#else
	n = ((n+(2*VSIZE2-1))/(2*VSIZE2)) * (2*VSIZE2);		// multiple of twice the vector size
	#endif
	if (n < VSIZE2*4) n=VSIZE2*4;			// avoid overflow with NLAT_2 < VSIZE2*2
	return n;
}

/*! Initialization of Spherical Harmonic transforms (backward and forward, vector and scalar, ...) of given size.
 * <b>This function must be called after \ref shtns_create and before any SH transform.</b> and sets all global variables and internal data.
 * returns the required number of doubles to be allocated for a spatial field.
 * \param shtns is the config created by \ref shtns_create for which the grid will be set.
 * \param nlat,nphi pointers to the number of latitudinal and longitudinal grid points respectively. If 0, they are set to optimal values.
 * \param nl_order defines the maximum SH degree to be resolved by analysis : lmax_analysis = lmax*nl_order. It is used to set an optimal and anti-aliasing nlat. If 0, the default SHT_DEFAULT_NL_ORDER is used.
 * \param flags allows to choose the type of transform (see \ref shtns_type) and the spatial data layout (see \ref spat)
 * \param eps polar optimization threshold : polar values of Legendre Polynomials below that threshold are neglected (for high m), leading to increased performance (a few percents)
 *  0 = no polar optimization;  1.e-14 = VERY safe;  1.e-10 = safe;  1.e-6 = aggresive, but still good accuracy.
*/
int shtns_set_grid_auto(shtns_cfg shtns, enum shtns_type flags, double eps, int nl_order, int *nlat, int *nphi)
{
	double t;
	int im,m;
	int layout;
	int nloop = 0;
	int n_gauss = 0;
	const int on_the_fly = 1;		// only on-the-fly algos are available
	int quick_init = 0;
	int vector = !(flags & SHT_SCALAR_ONLY);
	int latdir = (flags & SHT_SOUTH_POLE_FIRST) ? -1 : 1;		// choose latitudinal direction (change sign of ct)
	int cfg_loaded = 0;
	int accuracy_check = 1;
	const int req_flags = flags;		// requested flags.

	if (*nlat & 1) quick_init = 1;	// only one type of transform works with nlat odd. NEVER try others.
	#ifdef SHTNS4MAGIC
		if (*nlat % (VSIZE2*2)) shtns_runerr("Nlat must be an even multiple of vector size\n");
	#endif
	if (shtns->howmany != 1) {		// more constraints apply for batched transforms:
		if (shtns->nlat & 1) shtns_runerr("Nlat must be even for a batched transform\n");
	}
	shtns_unset_grid(shtns);		// release grid if previously allocated.
	if (nl_order <= 0) nl_order = SHT_DEFAULT_NL_ORDER;
/*	shtns.lshift = 0;
	if (nl_order == 0) nl_order = SHT_DEFAULT_NL_ORDER;
	if (nl_order < 0) {	shtns.lshift = -nl_order;	nl_order = 1; }		// linear with a shift in l.
*/
	shtns->nspat = 0;
	shtns->nlorder = nl_order;
	layout = flags & 0xFFFF00;
	flags = flags & 255;	// clear higher bits.

	switch (flags) {
		case sht_auto :		flags = sht_gauss;	break;		// only gauss available.
		case sht_reg_fast:	quick_init = 1;
		case sht_reg_dct:	flags = sht_reg_fast; break;
		case sht_gauss_fly :  flags = sht_gauss;  break;
		case sht_quick_init : flags = sht_gauss;  quick_init = 1;  break;
		case sht_reg_poles : quick_init = 1;	break;		// WARNING: quick_init mandatory here, as reg_poles needs NWAY>1 to work (quick_init sets NWAY=2)
		default : break;
	}
	if (layout & SHT_ROBERT_FORM) shtns->robert_form = 1;	// set Robert form
	#ifdef SHTNS4MAGIC
		if (flags == sht_reg_poles) shtns_runerr("Grid cannot include poles with MagIC layout.");
	#endif
	#if SHTNS_GPU
		shtns->sizeof_real = (layout & SHT_FP32) ? 4 : 8;
		if ((layout & SHT_FP32) && (layout & SHT_ALLOW_GPU)) {
			quick_init = 1;		// for now, FP32 only works on GPU anyway, no need to compare to cpu.
		}
		if (((layout & (SHT_ALLOW_GPU | SHT_PHI_CONTIGUOUS)) == SHT_ALLOW_GPU) && (*nlat % 4)) printf("!!! Warning !!! Nlat must be a multiple of 4 to run on GPU, unless phi-contiguous layout is requested\n");
	#endif

	if (vector) {
		// initialize sin(theta).d/dtheta matrix (for vector transforms)
		shtns->mx_stdt = (double*) malloc( 2*NLM*sizeof(double) );		// for vector synthesis
		st_dt_matrix_shifted(shtns, shtns->mx_stdt);
		shtns->mx_van = shtns->mx_stdt;		//  matrices share the same coefficients, namely mx_van[lm] = - mx_stdt[lm^1]  ...
		if (SHT_NORM == sht_schmidt) {	// ... except for Schmidt normalization
			shtns->mx_van = (double*) malloc( 2*NLM*sizeof(double) );		// for vector synthesis
			mul_ct_matrix_shifted(shtns, shtns->mx_van);
			for (long lm=0; lm<NLM; lm++) {		// 2*cos(theta) + sin(theta) d./dtheta = 1/sin(theta). d/dtheta(sin^2(theta) .)
				double ml = 2.*shtns->mx_van[2*lm] + shtns->mx_stdt[2*lm];
				double mu = 2.*shtns->mx_van[2*lm+1] + shtns->mx_stdt[2*lm+1];
				shtns->mx_van[2*lm] = -mu;
				shtns->mx_van[2*lm+1] = -ml;
			}
		}
	}

	if (*nphi == 0) {
		*nphi = fft_int((nl_order+1)*MMAX+1, 7);		// required fft nodes
	}
	if (*nlat == 0) {
		n_gauss = ((nl_order+1)*LMAX)/2 +1;		// required gauss nodes
		n_gauss = choose_nlat( n_gauss );
		if (flags != sht_gauss) {
			m = ((nl_order+1)*(LMAX+1));		// required regular nodes
			m = choose_nlat( m );
			*nlat = m;
		} else *nlat = n_gauss;
		#ifndef SHTNS_GPU
		// don't do this with GPU (not relevant)
		if (((layout & (SHT_ALLOW_PADDING|SHT_PHI_CONTIGUOUS)) == 0) && (shtns->nthreads == 1)) {
			if ((*nlat % 64 == 0) && (*nlat * *nphi > 512)) {		// heuristics to avoid cache bank conflicts.
			#ifndef SHTNS4MAGIC
				*nlat += 8;			// cache line assumed to be 64 bytes == 8 doubles.
			#else
				*nlat += (VSIZE2 > 4) ? 2*VSIZE2 : 8;		// here we also need nlat to be a multiple of 2*VSIZE2.
			#endif
			}
		}
		#endif
	}

	if (quick_init == 0) {		// do not waste too much time finding optimal fftw.
		//shtns->fftw_plan_mode = FFTW_EXHAUSTIVE;		// defines the default FFTW planner mode.
		shtns->fftw_plan_mode = FFTW_PATIENT;		// defines the default FFTW planner mode.
		//fftw_set_timelimit(60.0);		// do not search plans for more than 1 minute (does it work well ???)
		if (*nphi > 512) shtns->fftw_plan_mode = FFTW_PATIENT;
		if (*nphi > 1024) shtns->fftw_plan_mode = FFTW_MEASURE;
	} else {
		shtns->fftw_plan_mode = FFTW_ESTIMATE;
	}

	if (*nlat <= shtns->lmax) shtns_runerr("Nlat must be larger than Lmax");
	if ((flags != sht_gauss)&&(*nlat <= 2*shtns->lmax)) { accuracy_check=0; printf("\033[93m !WARNING! Nlat must be larger than 2*Lmax for analysis to work (sampling theorem)!\033[0m\n"); }
	if (IS_TOO_LARGE(*nlat, shtns->nlat)) shtns_runerr("Nlat too large");
	if (IS_TOO_LARGE(*nphi, shtns->nphi)) shtns_runerr("Nphi too large");

	// copy to plan variables.
	shtns->nphi = *nphi;
	shtns->nlat_2 = (*nlat+1)/2;	shtns->nlat = *nlat;

	if (layout & SHT_LOAD_SAVE_CFG)	{
		FILE* f = fopen("shtns_cfg_fftw","r");
		if (f) {
			fftw_import_wisdom_from_file(f);		// load fftw wisdom.
			fclose(f);
		}
	}
	planFFT(shtns, layout);		// initialize fftw
	init_sht_array_func(shtns);		// array of SHT functions is now set.

	alloc_SHTarrays(shtns);		// allocate dynamic arrays
	shtns->grid = GRID_NONE;
	switch(flags) {
		case sht_gauss : 	 shtns->grid = GRID_GAUSS;	break;
		case sht_reg_poles : shtns->grid = GRID_POLES;	break;
		case sht_reg_fast :  shtns->grid = GRID_REGULAR;  break;
		default:  shtns_runerr("unknown grid (should not happen)");
	}
	grid_weights(shtns, latdir);

	if (NLAT < VSIZE2*4) shtns_runerr("nlat is too small! try setting nlat>=32");		// avoid overflow with NLAT_2 < VSIZE2*2
	PolarOptimize(shtns, eps);
	set_sht_fly(shtns, 0);		// switch function pointers to "on-the-fly" functions.

  #ifdef SHTNS_GPU
	int gpu_ok = -1;
	if ((layout & SHT_ALLOW_GPU) && (NLAT % 4 == 0 || (layout & SHT_PHI_CONTIGUOUS))) {		// gpu requires NLAT multiple of 4, unless phi-contiguous layout is used
		gpu_ok = cushtns_init_gpu(shtns);		// try to initialize cuda gpu
		if (gpu_ok >= 0) {
			int err = init_gpu_staging_buffer(shtns);		// initialize staging buffers for auto-offload feature.
			if (err)  gpu_ok = -1;
		}
		if ((verbose)&&(gpu_ok>=0)) printf("        + GPU #%d successfully initialized.\n", gpu_ok);
	}
	if (gpu_ok < 0) {		// disable the GPU functions
		for (int j=SHT_GPU1; j<=SHT_GPU4; j++) {
			memset(sht_func[SHT_STD][j], 0, sizeof(void*)*SHT_NTYP);
		}
	} else 	set_sht_gpu(shtns, 0);		// switch function pointers to "gpu" functions by default
  #endif

	if ((layout & SHT_LOAD_SAVE_CFG) && (!cfg_loaded)) cfg_loaded = (config_load(shtns, req_flags) > 0);
	const double t_estimate = 5e-10*LMAX*NLAT*MMAX/VSIZE2 * shtns->howmany;		// very rough cost estimate (in seconds for 1 core @ 1Ghz).
	if ((quick_init == 0) && (!cfg_loaded)) {
		choose_best_sht(shtns, &nloop, vector);
		if (layout & SHT_LOAD_SAVE_CFG) config_save(shtns, req_flags);
	} else if (t_estimate >= 0.3*shtns->nthreads)  accuracy_check = 0;	// don't perform accuracy checks for too large transforms (takes too much time).

	if (accuracy_check) {
		t = SHT_error(shtns, vector);		// compute SHT accuracy.
		if (verbose) printf("        + SHT accuracy = %.3g\n",t);
		if (t > ((layout & SHT_FP32) ? 5e-3 : 1.e-6) || isNotFinite(t)) {
			printf("\033[93m Accuracy test failed. Please file a bug report at https://bitbucket.org/nschaeff/shtns/issues \033[0m\n");
			#if (VSIZE2 == 8) && (defined __GNUC__) && !(defined __INTEL_COMPILER)
			printf("\033[93m You may need to upgrade the 'binutils' package, see https://bitbucket.org/nschaeff/shtns/issues/37/ \033[0m\n");
			#endif
			if (verbose < 2) shtns_runerr("bad SHT accuracy");		// stop if something went wrong (but not in debug mode)
		}
	}

	if ((omp_threads > 1)&&(verbose>1)) printf(" nthreads = %d\n",shtns->nthreads);
	if (verbose) printf("        => SHTns is ready.\n");
	return(shtns->nspat);	// returns the number of doubles to be allocated for a spatial field.
}


/*! Initialization of Spherical Harmonic transforms (backward and forward, vector and scalar, ...) of given size.
 * <b>This function must be called after \ref shtns_create and before any SH transform.</b> and sets all global variables.
 * returns the required number of doubles to be allocated for a spatial field.
 * \param shtns is the config created by shtns_create for which the grid will be set.
 * \param flags allows to choose the type of transform (see \ref shtns_type) and the spatial data layout (see \ref spat)
 * \param eps polar optimization threshold : polar values of Legendre Polynomials below that threshold are neglected (for high m), leading to increased performance (a few percents)
 *  0 = no polar optimization;  1.e-14 = VERY safe;  1.e-10 = safe;  1.e-6 = aggresive, but still good accuracy.
 * \param nlat,nphi respectively the number of latitudinal and longitudinal grid points.
*/
int shtns_set_grid(shtns_cfg shtns, enum shtns_type flags, double eps, int nlat, int nphi)
{
	if ((nlat == 0)||(nphi == 0)) shtns_runerr("nlat or nphi is zero !");
	return( shtns_set_grid_auto(shtns, flags, eps, 0, &nlat, &nphi) );
}

/** Batched transforms, with some constraints.
 * Currently only theta-contiguous data is allowed.
 * This function must be called before \ref shtns_set_grid or \ref shtns_set_grid_auto, after which the spatial datat layout will be defined
 * by \c shtns->nlat_padded and \c shtns->nspat as:
 * \code data[(i_batch*shtns->nphi + i_phi)*shtns->nlat_padded + i_theta] \endcode
 * Note that \c shtns->nspat will be the number of spatial points in howmany fields (not in a single field).
 * \param[in] shtns = a plan created by \ref shtns_create that should handle many transforms at once.
 * \param[in] howmany = number of transforms in batch.
 * \param[in] spec_dist = distance between spectral arrays in batch. Spectral data is accessed with \code Qlm[i_batch * spec_dist + lm] \endcode
 * \returns howmany on success, or -1 on failure.
*/
int shtns_set_many(shtns_cfg shtns, const int howmany, long spec_dist)
{
	if (howmany <= 0)	return -1;		// invalid
	if (spec_dist == 0)  spec_dist = shtns->nlm;
	if (spec_dist < shtns->nlm) return -1;	// invalid
	if (shtns->nspat != 0) return -1;		// grid already set!!

	shtns->howmany = howmany;
	shtns->spec_dist = spec_dist;		// distance between spectral fields.
	return howmany;
}

/*! Simple initialization of Spherical Harmonic transforms (backward and forward, vector and scalar, ...) of given size.
 * This function sets all global variables by calling \ref shtns_create followed by \ref shtns_set_grid, with the
 * default normalization and the default polar optimization (see \ref sht_config.h).
 * Returns the configuration to be passed to subsequent transform functions, which is basicaly a pointer to a \ref shtns_info struct.
 * \param lmax : maximum SH degree that we want to describe.
 * \param mmax : number of azimutal wave numbers.
 * \param mres : \c 2.pi/mres is the azimutal periodicity. \c mmax*mres is the maximum SH order.
 * \param nlat,nphi : respectively the number of latitudinal and longitudinal grid points.
 * \param flags allows to choose the type of transform (see \ref shtns_type) and the spatial data layout (see \ref spat)
*/
shtns_cfg shtns_init(enum shtns_type flags, int lmax, int mmax, int mres, int nlat, int nphi)
{
	shtns_cfg shtns = shtns_create(lmax, mmax, mres, SHT_DEFAULT_NORM);
	if (shtns != NULL)
		shtns_set_grid(shtns, flags, SHT_DEFAULT_POLAR_OPT, nlat, nphi);
	return shtns;
}

/// set the use of Robert form. If robert != 0, the vector synthesis returns a field multiplied by sin(theta), while the analysis divides by sin(theta) before the transform.
void shtns_robert_form(shtns_cfg shtns, int robert)
{
	#ifdef SHTNS_GPU
	if (robert != shtns->robert_form  &&  shtns->d_clm) shtns_runerr("[shtns_robert_form] ERROR: must be called before shtns_set_grid!");
	#endif
	shtns->robert_form = robert;
}

/** Enables OpenMP parallel transforms, if available (see \ref compil).
 Call before any initialization of shtns to use multiple threads. Returns the actual number of threads.
 \li If num_threads > 0, specifies the maximum number of threads that should be used.
 \li If num_threads <= 0, maximum number of threads is automatically set to the number of processors.
 \li If num_threads == 1, openmp will be disabled. */
int shtns_use_threads(int num_threads)
{
#ifdef _OPENMP
	int procs = omp_get_num_procs();
	if (num_threads <= 0)  num_threads = omp_get_max_threads();
	else if (num_threads > 4*procs) num_threads = 4*procs;		// limit the number of threads
	omp_threads = num_threads;
#endif
#ifdef OMP_FFTW
	fftw_init_threads();		// enable threads for FFTW.
#endif
	return omp_threads;
}

/// fill the given array with quadrature weights. returns the number of weights written, which
/// may be zero if the grid has no quadrature rule
int shtns_gauss_wts(shtns_cfg shtns, double *wts)
{
	int i = 0;
	if (shtns->wg) {
		const double rescale = shtns->wg[-1];		// weights are stored with a rescaling that depends on SHT_NORM.
		do {
			wts[i] = shtns->wg[i] * rescale;
		} while(++i < shtns->nlat_2);
	}
	return i;
}

///@}


#ifdef SHT_F77_API

/* FORTRAN API */

/** \addtogroup fortapi Fortran API.
* Call from fortran without the trailing '_'.
* see the \link SHT_example.f Fortran example \endlink for a simple usage of SHTns from Fortran language.
*/
///@{

/// Set verbosity level
void shtns_verbose_(int *v)
{
	shtns_verbose(*v);
}

/// Enable threads
void shtns_use_threads_(int *num_threads)
{
	shtns_use_threads(*num_threads);
}

/// Print info
void shtns_print_cfg_()
{
	shtns_print_version();
	if (sht_data) shtns_print_cfg(sht_data);
}

/// Initializes spherical harmonic transforms of given size using Gauss algorithm with default polar optimization.
void shtns_init_sh_gauss_(int *layout, int *lmax, int *mmax, int *mres, int *nlat, int *nphi)
{
	shtns_cfg shtns = shtns_create(*lmax, *mmax, *mres, SHT_DEFAULT_NORM);
	shtns_set_grid(shtns, sht_gauss | *layout, SHT_DEFAULT_POLAR_OPT, *nlat, *nphi);
}

/// Initializes spherical harmonic transforms of given size using Fastest available algorithm and polar optimization.
void shtns_init_sh_auto_(int *layout, int *lmax, int *mmax, int *mres, int *nlat, int *nphi)
{
	shtns_cfg shtns = shtns_create(*lmax, *mmax, *mres, SHT_DEFAULT_NORM);
	shtns_set_grid(shtns, sht_auto | *layout, SHT_DEFAULT_POLAR_OPT, *nlat, *nphi);
}

/// Initializes spherical harmonic transforms of given size using a regular grid and agressive optimizations.
void shtns_init_sh_reg_fast_(int *layout, int *lmax, int *mmax, int *mres, int *nlat, int *nphi)
{
	shtns_cfg shtns = shtns_create(*lmax, *mmax, *mres, SHT_DEFAULT_NORM);
	shtns_set_grid(shtns, sht_reg_fast | *layout, 1.e-6, *nlat, *nphi);
}

/// Initializes spherical harmonic transform SYNTHESIS ONLY of given size using a regular grid including poles.
void shtns_init_sh_poles_(int *layout, int *lmax, int *mmax, int *mres, int *nlat, int *nphi)
{
	shtns_cfg shtns = shtns_create(*lmax, *mmax, *mres, SHT_DEFAULT_NORM);
	shtns_set_grid(shtns, sht_reg_poles | *layout, 0, *nlat, *nphi);
}

/// Defines the size and convention of the transform.
/// Allow to choose the normalization and whether or not to include the Condon-Shortley phase.
/// \see shtns_create
void shtns_set_size_(int *lmax, int *mmax, int *mres, int *norm)
{
	shtns_create(*lmax, *mmax, *mres, *norm);
}

/// Precompute matrices for synthesis and analysis.
/// Allow to choose polar optimization threshold and algorithm type.
/// \see shtns_set_grid
void shtns_precompute_(int *type, int *layout, double *eps, int *nlat, int *nphi)
{
	shtns_set_grid(sht_data, *type | *layout, *eps, *nlat, *nphi);
}

/// Same as shtns_precompute_ but choose optimal nlat and/or nphi.
/// \see shtns_set_grid_auto
void shtns_precompute_auto_(int *type, int *layout, double *eps, int *nl_order, int *nlat, int *nphi)
{
	shtns_set_grid_auto(sht_data, *type | *layout, *eps, *nl_order, nlat, nphi);
}

/// Clear everything
void shtns_reset_() {
	shtns_reset();
}

#define MAX_CONFIGS (4)
shtns_cfg shtns_configs[MAX_CONFIGS];

/// Saves the current shtns configuration with tag n
void shtns_save_cfg_(unsigned *n) {
	if (*n < MAX_CONFIGS)
		shtns_configs[*n] = sht_data;
	else
		fprintf(stderr, "error saving shtns_cfg, tag %u too big\n", *n);
}

/// Loads the previously saved configuration tagged n
void shtns_load_cfg_(unsigned *n) {
	shtns_cfg s0 = sht_data;		// beginning of list
	shtns_cfg s2 = NULL;
	if (*n < MAX_CONFIGS) s2 = shtns_configs[*n];
	if (s2 != NULL) {
		if (s0 == s2) return;		// nothing to do, config already active.
		while (s0 != NULL) {
			shtns_cfg s1 = s0->next;
			if (s1 == s2) {
				s0->next = s1->next;		// remove from list
				s1->next = sht_data;		// place in front of list
				sht_data = s1;
				return;			// done!
			}
			s0 = s1;
		}
	}
	fprintf(stderr, "error loading shtns_cfg, invalid tag (%u)\n", *n);
}

/// returns nlm, the number of complex*16 elements in an SH array.
/// call from fortran using \code call shtns_calc_nlm(nlm, lmax, mmax, mres) \endcode
void shtns_calc_nlm_(int *nlm, const int *const lmax, const int *const mmax, const int *const mres)
{
    *nlm = nlm_calc(*lmax, *mmax, *mres);
}


int shtns_lmidx_fortran(shtns_cfg shtns, const int *const l, const int *const m)
{
	unsigned im = *m;
	unsigned mres = shtns->mres;
	if (mres > 1) {
		unsigned k = im % mres;
		im = im / mres;
		if (k) printf("wrong m");
	}
    return LiM(shtns, *l, im) + 1;	// convert to fortran convention index.
}

/// returns lm, the index in an SH array of mode (l,m).
/// call from fortran using \code call shtns_lmidx(lm, l, m) \endcode
void shtns_lmidx_(int *lm, const int *const l, const int *const m)
{
    *lm = shtns_lmidx_fortran(sht_data, l, m);
}

int shtns_lm2l_fortran(shtns_cfg shtns, const int *lm)
{
	return shtns->li[*lm -1];	// convert from fortran convention index.
}

int shtns_lm2m_fortran(shtns_cfg shtns, const int *lm)
{
	return shtns->mi[*lm -1];	// convert from fortran convention index.
}

/// returns l and m, degree and order of an index in SH array lm.
/// call from fortran using \code call shtns_l_m(l, m, lm) \endcode
void shtns_l_m_(int *l, int *m, const int *const lm)
{
	*l = sht_data->li[*lm -1];	// convert from fortran convention index.
	*m = sht_data->mi[*lm -1];
}

/// fills the given array with the cosine of the co-latitude angle (NLAT real*8)
/// if no grid has been set, the first element will be set to zero.
void shtns_cos_array_(double *costh)
{
	if (sht_data->ct) {
		for (int i=0; i<sht_data->nlat; i++)
			costh[i] = sht_data->ct[i];
	} else costh[0] = 0.0;	// mark as invalid.
}

/// fills the given array with the gaussian quadrature weights ((NLAT+1)/2 real*8).
/// when there is no gaussian grid, the first element is set to zero.
void shtns_gauss_wts_(double *wts)
{
	int i = shtns_gauss_wts(sht_data, wts);
	if (i==0) wts[0] = 0;	// mark as invalid.
}


/** \name Point evaluation of Spherical Harmonics
Evaluate at a given point (\f$cos(\theta)\f$ and \f$\phi\f$) a spherical harmonic representation.
*/
///@{
/// \see SH_to_point for argument description
void shtns_sh_to_point_(double *spat, cplx *Qlm, double *cost, double *phi)
{
	*spat = SH_to_point(sht_data, Qlm, *cost, *phi);
}

/// \see SHqst_to_point for argument description
void shtns_qst_to_point_(double *vr, double *vt, double *vp,
		cplx *Qlm, cplx *Slm, cplx *Tlm, double *cost, double *phi)
{
	SHqst_to_point(sht_data, Qlm, Slm, Tlm, *cost, *phi, vr, vt, vp);
}
///@}

void shtns_sh_zrotate_(cplx* Qlm, double* alpha, cplx* Rlm)
{
	SH_Zrotate(sht_data, Qlm, *alpha, Rlm);
}

void shtns_sh_yrotate_(cplx* Qlm, double* alpha, cplx* Rlm)
{
	SH_Yrotate(sht_data, Qlm, *alpha, Rlm);
}

void shtns_sh_xrotate90_(cplx *Qlm, cplx *Rlm)
{
	SH_Xrotate90(sht_data, Qlm, Rlm);
}

void shtns_sh_yrotate90_(cplx *Qlm, cplx *Rlm)
{
	SH_Yrotate90(sht_data, Qlm, Rlm);
}

void shtns_sh_cplx_zrotate_(cplx* Qlm, double* alpha, cplx* Rlm)
{
	SH_cplx_Zrotate(sht_data, Qlm, *alpha, Rlm);
}

void shtns_sh_cplx_yrotate_(cplx* Qlm, double* alpha, cplx* Rlm)
{
	SH_cplx_Yrotate(sht_data, Qlm, *alpha, Rlm);
}

void shtns_sh_cplx_xrotate90_(cplx *Qlm, cplx *Rlm)
{
	SH_cplx_Xrotate90(sht_data, Qlm, Rlm);
}

void shtns_sh_cplx_yrotate90_(cplx *Qlm, cplx *Rlm)
{
	SH_cplx_Yrotate90(sht_data, Qlm, Rlm);
}




///@}

#endif
