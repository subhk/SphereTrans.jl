/*
 * Copyright (c) 2010-2018 Centre National de la Recherche Scientifique.
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

/// \file time_SHT.c This program performs some spherical harmonic transforms, and displays timings and accuracy.
/// \c make \c time_SHT to compile, and then run.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>		// for the clock() function
#include "fftw3/fftw3.h"

#include <shtns.h>

shtns_cfg shtns;

complex double *Slm, *Slm0, *Tlm, *Tlm0, *Qlm;	// spherical harmonics l,m space
complex double *ShF, *ThF, *NLF;	// Fourier space : theta,m
double *Sh, *Th, *NL;		// real space : theta,phi (alias of ShF)

int LMAX,MMAX,MRES,NLM;
int NLAT = 0;
int NPHI = 0;

// number of SH iterations
int SHT_ITER = 50;		// do 50 iterations by default
int batch = 1;			// perform 1 transform together by default

int error = 0;
#define COLOR_OK  "\033[92m"
#define COLOR_WRN "\033[93m"
#define COLOR_ERR "\033[91m"
#define COLOR_END "\033[0m"

double err_err = 1e-6;
double err_wrn = 1e-9;

void runerr(const char * error_text)
{
	printf("%s\n",error_text);
	exit(1);
}

/// for real-time performance measurements, returns time in mili-seconds.
#ifdef _OPENMP
  #include <omp.h>
  inline double wtime() {  return omp_get_wtime();  }
#else
  #include <sys/time.h>
  double wtime() {			// use gettimeofday
	static long sec_base = -1;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	if (sec_base == -1) sec_base = tv.tv_sec;
	return tv.tv_usec*1e-6 + (tv.tv_sec - sec_base);
  }
#endif


void write_vect(char *fn, double *vec, int N)
{
	FILE *fp;
	int i;
	
	fp = fopen(fn,"w");
	for (i=0;i<N;i++) {
		fprintf(fp,"%.6g ",vec[i]);
	}
	fclose(fp);
}

void write_mx(char *fn, double *mx, int N1, int N2)
{
	FILE *fp;
	int i,j;
	
	fp = fopen(fn,"w");
	for (i=0;i<N1;i++) {
		for(j=0;j<N2;j++) {
			fprintf(fp,"%.6g ",mx[i*N2+j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

/// check if an IEEE754 double precision number is finite (works also with -ffinite-math).
int isNotFinite(double x) {
	union {
		volatile double d;
		volatile long i;
	} mem;

	mem.d = x;
	return (mem.i & 0x7FF0000000000000) == 0x7FF0000000000000;		// nan or inf
}

int allFinite(double* x, int n) {
	for (int i=0; i<n; i++) {
		long a = ((long*) x)[i];
		if ((a & 0x7FF0000000000000) == 0x7FF0000000000000) return 0;		// nan or inf found
	}
	return 1;	// all finite !
}

/// check if an IEEE754 double precision number is finite and normalized.
int isNotNormal(double x) {

	union {
		volatile double d;
		volatile long i;
	} mem;

	mem.d = x;
	long ii = mem.i;
	long exp  = ii & 0x7FF0000000000000;
	long mant = ii & 0x000FFFFFFFFFFFFF;
	if (exp == 0x7FF0000000000000) return 1;		// nan or inf
	if ((exp == 0) && (mant != 0)) return 1;		// denormal
	return 0;
}

void print_error(double err_rms, double err_max, int l_max, int lm_max, int b_max, char* name)
{
	printf("  %s => max error = %g (l=%d,m=%d,lm=%d,batch=%d)   rms error = %g   ",	name, err_max, l_max, im_from_lm(lm_max, LMAX, MRES)*MRES, lm_max, b_max, err_rms);

	if ((err_max > 100*err_err) || (err_rms > err_err) || isNotFinite(err_rms)) {
		printf(COLOR_ERR " **** ERROR **** " COLOR_END "\n");
		error++;
	} else if ((err_max > 100*err_wrn) || (err_rms > err_wrn)) {
		printf(COLOR_WRN "SUSPECT" COLOR_END "\n");
	} else
		printf(COLOR_OK "OK" COLOR_END "\n");
}

double scal_error(complex double *Slm, complex double *Slm0, int ltr)
{
	long int jj,i, nlm_cplx, ib;
	double tmax,t,n2;
	const long nlm0 = shtns->nlm;

	nlm_cplx = (MMAX*2 == NPHI) ? LiM(shtns, MRES*MMAX,MMAX) : nlm0;
// compute error :
	tmax = 0;	n2 = 0;		jj=0;	ib=0;
//	if (!allFinite((double*)Slm, 2*NLM)) printf("NaN, Inf or Denormal detected\n");
	for (int b=0;b<batch;b++) {
		for (i=0;i<nlm0;i++) {
			//if ((isNaN(creal(Slm[i]))) || (isNaN(cimag(Slm[i])))) printf("NaN @ lm=%ld (l=%d)\n",i,shtns->li[i]);
			if ((i <= LMAX)||(i >= nlm_cplx)) {		// m=0, and 2*m=nphi is real
				if (shtns->li[i] <= ltr)	Slm[i+b*NLM] = creal(Slm[i+b*NLM]-Slm0[i+b*NLM]);
				t = fabs(creal(Slm[i+b*NLM]));
			} else {
				if (shtns->li[i] <= ltr)	Slm[i+b*NLM] -= Slm0[i+b*NLM];
				t = cabs(Slm[i+b*NLM]);
			}
			n2 += t*t;
	//		if (isNotFinite(t)) printf("NaN or Inf @ lm=%ld (l=%d)  Slm=%g,%g  Slm0=%g,%g\n",i,shtns->li[i], creal(Slm[i]), cimag(Slm[i]), creal(Slm0[i]), cimag(Slm0[i]));
			if (t>tmax) { tmax = t; jj = i; ib=b; }
		}
	}
	print_error(sqrt(n2/(nlm0*batch)), tmax, shtns->li[jj],jj,ib, "");
	if ((tmax > 1e-7) && (nlm0 < 15)) {
		printf(" orig:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm_cplx)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Slm0[i+ib*NLM]));
			} else {
				printf("  %g,%g",creal(Slm0[i+ib*NLM]),cimag(Slm0[i+ib*NLM]));
			}
		printf("\n diff:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm_cplx)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Slm[i+ib*NLM]));
			} else {
				printf("  %g,%g",creal(Slm[i+ib*NLM]),cimag(Slm[i+ib*NLM]));
			}
		printf("\n");
	}
	return(tmax);
}

double vect_error(complex double *Slm, complex double *Tlm, complex double *Slm0, complex double *Tlm0, int ltr)
{
	long int jj,i,ib;
	double tmax0, tmax,t,n2;
	const long nlm0 = shtns->nlm;

// compute error :
	tmax = 0;	n2 = 0;		jj=0;	ib=0;
	for (int b=0;b<batch;b++) {
		for (i=0;i<nlm0;i++) {
			if ((i <= LMAX)||(i >= LiM(shtns, MRES*(NPHI+1)/2,(NPHI+1)/2))) {
				if (shtns->li[i] <= ltr)	Slm[i+b*NLM] = creal(Slm[i+b*NLM]-Slm0[i+b*NLM]);
				t = fabs(creal(Slm[i+b*NLM]));
			} else {
				if (shtns->li[i] <= ltr)	Slm[i+b*NLM] -= Slm0[i+b*NLM];
				t = cabs(Slm[i+b*NLM]);
			}
			int l = shtns->li[i];
			n2 += t*t * (l*(l+1));             // relative error: account for mean spectrum of unit energy
			if (t>tmax) { tmax = t; jj = i; ib = b; }
		}
	}
	print_error(sqrt(n2/(nlm0*batch)), tmax, shtns->li[jj],jj,ib, "Spheroidal");
	if ((tmax > 1e-4) && (nlm0 < 15)) {
		printf(" orig:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm0)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Slm0[i]));
			} else {
				printf("  %g,%g",creal(Slm0[i]),cimag(Slm0[i]));
			}
		printf("\n diff:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm0)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Slm[i]));
			} else {
				printf("  %g,%g",creal(Slm[i]),cimag(Slm[i]));
			}
		printf("\n");
	}
//	write_vect("Slm",Slm,NLM*2);
	tmax0 = tmax;

// compute error :
	tmax = 0;	n2 = 0;		jj=0;	ib=0;
	for (int b=0;b<batch;b++) {
		for (i=0;i<nlm0;i++) {
			if ((i <= LMAX)||(i >= LiM(shtns, MRES*(NPHI+1)/2,(NPHI+1)/2))) {
				if (shtns->li[i] <= ltr)	Tlm[i+b*NLM] = creal(Tlm[i+b*NLM]- Tlm0[i+b*NLM]);
				t = fabs(creal(Tlm[i+b*NLM]));
			} else {
				if (shtns->li[i] <= ltr)	Tlm[i+b*NLM] -= Tlm0[i+b*NLM];
				t = cabs(Tlm[i+b*NLM]);
			}
			int l = shtns->li[i];
			n2 += t*t * (l*(l+1));             // relative error: account for mean spectrum of unit energy
			if (t>tmax) { tmax = t; jj = i; ib=b; }
		}
	}
	print_error(sqrt(n2/(nlm0*batch)), tmax, shtns->li[jj],jj,ib, "Toroidal");
	if ((tmax > 1e-4) && (nlm0 < 15)) {
		printf(" orig:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm0)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Tlm0[i]));
			} else {
				printf("  %g,%g",creal(Tlm0[i]),cimag(Tlm0[i]));
			}
		printf("\n diff:");
		for (i=0; i<nlm0;i++)
			if ((i <= LMAX)||(i >= nlm0)) {		// m=0, and 2*m=nphi is real
				printf("  %g",creal(Tlm[i]));
			} else {
				printf("  %g,%g",creal(Tlm[i]),cimag(Tlm[i]));
			}
		printf("\n");
	}
//	write_vect("Tlm",Tlm,NLM*2);
	return(tmax > tmax0 ? tmax : tmax0);
}

void test_SH_point(int vector)
{
	long int jj,i;
	double ts2, ta2=0;

	for (i=0;i<NLM;i++) Slm[i] = Slm0[i];	// restore test case...

	ts2 = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		double v = SH_to_point(shtns, Slm, 0.8, 0.76);
	}
	ts2 = wtime() - ts2;

  if (vector) {
	ta2 = wtime();
	for (jj=1; jj< SHT_ITER; jj++) {
		double vr, vt, vp;
		SHqst_to_point(shtns, Slm, Slm0, Tlm0, 0.8, 0.76, &vr, &vt, &vp);
	}
	ta2 = wtime() - ta2;
  }

	ts2 *= 1000./SHT_ITER;	// ms per eval
	ta2 *= 1000./SHT_ITER;	// ms per eval
	printf("   SHT_to_point time = %f ms [scalar], %f ms [3D vector]\n", ts2, ta2);
	printf("   SHT_to_point time = %f ms [scalar]", ts2);
	if (vector) printf(", %f ms [3D vector]", ta2);
	printf("\n");
	return;
}


void test_SHT()
{
	long int jj,i;
	clock_t tcpu;
	double ts, ta, ts2, ta2;
	double ts_leg, ts_fft, ts_wall, ta_leg, ta_fft, ta_wall;
	double gflop = 1e-6 * (NLAT*(NLM*4 +(MMAX+1)*2 + MMAX*log2(MMAX+1) + 5*NPHI*log2(NPHI)));		// Million floating point ops

	for (i=0;i<NLM*batch;i++) Slm[i] = Slm0[i];	// restore test case...

	tcpu = clock();
	ts2 = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SH_to_spat_time(shtns, Slm,Sh);
	}
	ts2 = wtime() - ts2;
	tcpu = clock() - tcpu;
	ts = tcpu / (1000.*SHT_ITER);
	ts_wall = shtns_profiling_read_time(shtns, &ts_leg, &ts_fft);

	for (i=0;i<NLM*batch;i++) Slm[i] = 1e30 -I*1e31;	// fill destination with garbage

	tcpu = clock();
	ta2 = wtime();
	spat_to_SH_time(shtns, Sh,Slm);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SH_time(shtns, Sh,Tlm);
	}
	ta2 = wtime() - ta2;
	tcpu = clock() - tcpu;
	ta = tcpu / (1000.*SHT_ITER*batch);
	ta_wall = shtns_profiling_read_time(shtns, &ta_fft, &ta_leg);
	
	ts2 *= 1000./(SHT_ITER*batch);
	ta2 *= 1000./(SHT_ITER*batch);
  #ifdef _OPENMP
	printf("   SHT time (lmax=%d): \t synthesis = %.5f ms [cpu %.3f] [%.3f Gflops] \t analysis = %.5f ms [cpu %.3f] [%.3f Gflops] \n", LMAX, ts2, ts, gflop/ts2, ta2, ta, gflop/ta2);
  #else
	printf("   SHT time (lmax=%d): \t synthesis = %f ms [%f Gflops] \t analysis = %f ms [%f Gflops] \n", LMAX, ts2, gflop/ts2, ta2, gflop/ta2);
  #endif
	printf("   split time (ms) (lmax=%d): \t synthesis leg = %f, fft = %f, total = %f \t analysis ileg = %f, fft = %f, wall = %f\n", LMAX, ts_leg*1000, ts_fft*1000, ts_wall*1000, ta_leg*1000, ta_fft*1000, ta_wall*1000);
	double err_max = scal_error(Slm, Slm0, LMAX);
	{	// record timings and accuracy in machine readable file
		FILE* fp = fopen("sht_timings.txt","a");
		fprintf(fp, "%d %d %d %d %d %d\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.3g #lmax mmax mres nlat nphi batch sy leg fft an ileg ifft err %s\n", LMAX,MMAX,MRES,NLAT,NPHI,batch, ts_wall*1000, ts_leg*1000, ts_fft*1000, ta_wall*1000, ta_leg*1000, ta_fft*1000, err_max, shtns_get_build_info());
		fclose(fp);
	}
	return;
}

void test_SHT_accuracy()
{
	for (int i=0;i<NLM*batch;i++) Slm[i] = Slm0[i];	// restore test case...
	for (int i=0;i<NLM*batch;i++) Tlm[i] = 1e30 - I*1e31;	// put garbage everywhere in destination
	for (int jj=0; jj< SHT_ITER; jj++) {
		SH_to_spat(shtns, Slm,Sh);
		spat_to_SH(shtns, Sh, Tlm);
		scal_error(Tlm, Slm0, LMAX);
	}
	return;
}

void test_SHT_m0()
{
	long int jj,i;
	double ts, ta;

	for (i=0;i<NLM*batch;i++) Slm[i] = Slm0[i];	// restore test case...

	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SHsph_to_spat(shtns, Slm,Sh,NULL);
	}
	ts = wtime() - ts;

	ta = wtime();
	SHtor_to_spat(shtns, Slm, NULL, Sh);
	for (jj=1; jj< SHT_ITER; jj++) {
		SHtor_to_spat(shtns, Slm, NULL, Sh);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   SHT time : \t spheroidal = %f ms \t torodial = %f ms\n", ts, ta);

	return;
}

void test_SHT_l(int ltr)
{
	int jj,i;
	double ts, ta;

	for (i=0;i<NLM*batch;i++) Slm[i] = Slm0[i];	// restore test case...

	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SH_to_spat_l(shtns, Slm,Sh,ltr);
	}
	ts = wtime() - ts;

	for (i=0;i<NLM*batch;i++) Slm[i] = 1e30 -I*1e31;	// fill destination with garbage

	ta = wtime();
		spat_to_SH_l(shtns, Sh,Slm,ltr);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SH_l(shtns, Sh,Tlm,ltr);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   SHT time truncated at l=%d : synthesis = %f ms, analysis = %f ms\n", ltr, ts, ta);

	scal_error(Slm, Slm0, ltr);

	if (LMAX < 256) {
		SH_to_spat_l(shtns, Slm0,Sh,ltr);
		spat_to_SH(shtns, Sh, Slm);
		scal_error(Slm, Slm0, ltr);		// check if the synthesis did not lead to l>ltr
	}
	return;
}

void test_SHT_vect_l(int ltr)
{
	int jj,i;
	double ts, ta;

	complex double *S2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *T2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = Slm0[i];	Tlm[i] = Tlm0[i];
	}
	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SHsphtor_to_spat_l(shtns, Slm,Tlm,Sh,Th,ltr);
	}
	ts = wtime() - ts;

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = 1e30 - I*1e31;		Tlm[i] = 1e20 + I*1e19;	// fill destination with garbage
	}

	ta = wtime();
		spat_to_SHsphtor_l(shtns, Sh,Th,Slm,Tlm, ltr);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SHsphtor_l(shtns, Sh,Th,S2,T2, ltr);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   vector SHT time trucated at l=%d : \t synthesis %f ms \t analysis %f ms\n", ltr, ts, ta);

	shtns_free(T2);	shtns_free(S2);
	vect_error(Slm, Tlm, Slm0, Tlm0, ltr);

	if (LMAX < 256) {
		SHsphtor_to_spat_l(shtns, Slm0,Tlm0,Sh,Th,ltr);
		spat_to_SHsphtor(shtns, Sh,Th,Slm,Tlm);
		vect_error(Slm, Tlm, Slm0, Tlm0, ltr);		// check if the synthesis did not actually produce l>ltr
	}
	return;
}

void test_SHT_vect()
{
	int jj,i;
	double ts, ta;

	complex double *S2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *T2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = Slm0[i];	Tlm[i] = Tlm0[i];
	}
	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SHsphtor_to_spat(shtns, Slm,Tlm,Sh,Th);
	}
	ts = wtime() - ts;

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = 1e30 - I*1e31;		Tlm[i] = 1e20 + I*1e19;	// fill destination with garbage
	}

	ta = wtime();
		spat_to_SHsphtor(shtns, Sh,Th,Slm,Tlm);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SHsphtor(shtns, Sh,Th,S2,T2);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   vector SHT time (lmax=%d) : \t synthesis %f ms \t analysis %f ms\n", LMAX, ts, ta);

	shtns_free(T2);	shtns_free(S2);
	vect_error(Slm, Tlm, Slm0, Tlm0, LMAX);
	return;
}

void test_SHT_vect3d_l(int ltr)
{
	int jj,i;
	double ts, ta;
	
	complex double *Q2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *S2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *T2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	
	for (i=0;i<NLM*batch;i++) {
		Slm[i] = Slm0[i];	Tlm[i] = Tlm0[i];	Qlm[i] = Tlm0[i];
	}

	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SHqst_to_spat_l(shtns, Qlm,Slm,Tlm,NL,Sh,Th, ltr);
	}
	ts = wtime() - ts;

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = 1e30 - I*1e31;		Tlm[i] = 1e20 + I*1e19;		Qlm[i] = 3e25 -I*1e25;	// fill destination with garbage
	}

	ta = wtime();
		spat_to_SHqst_l(shtns, NL,Sh,Th,Qlm,Slm,Tlm, ltr);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SHqst_l(shtns, NL,Sh,Th,Q2,S2,T2, ltr);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   3D vector SHT time : \t synthesis %f ms \t analysis %f ms\n", ts, ta);

	shtns_free(T2);	shtns_free(S2);	shtns_free(Q2);
	vect_error(Slm, Tlm, Slm0, Tlm0, ltr);
	scal_error(Qlm, Tlm0, ltr);

	if (LMAX < 256) {
		SHqst_to_spat_l(shtns, Tlm0,Slm0,Tlm0,NL,Sh,Th, ltr);
		spat_to_SHqst(shtns, NL,Sh,Th,Qlm,Slm,Tlm);
		vect_error(Slm, Tlm, Slm0, Tlm0, ltr);		// check if the synthesis did not actually produce l>ltr
		scal_error(Qlm, Tlm0, ltr);		// check if the synthesis did not lead to l>ltr
	}

	return;
}

void test_SHT_vect3d()
{
	int jj,i;
	double ts, ta;
	
	complex double *Q2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *S2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	complex double *T2 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	
	for (i=0;i<NLM*batch;i++) {
		Slm[i] = Slm0[i];	Tlm[i] = Tlm0[i];	Qlm[i] = Tlm0[i];
	}

	ts = wtime();
	for (jj=0; jj< SHT_ITER; jj++) {
		SHqst_to_spat(shtns, Qlm,Slm,Tlm,NL,Sh,Th);
	}
	ts = wtime() - ts;

	for (i=0;i<NLM*batch;i++) {
		Slm[i] = 1e30 - I*1e31;		Tlm[i] = 1e20 + I*1e19;		Qlm[i] = 3e25 -I*1e25;	// fill destination with garbage
	}

	ta = wtime();
		spat_to_SHqst(shtns, NL,Sh,Th,Qlm,Slm,Tlm);
	for (jj=1; jj< SHT_ITER; jj++) {
		spat_to_SHqst(shtns, NL,Sh,Th,Q2,S2,T2);
	}
	ta = wtime() - ta;
	ts *= 1000./(SHT_ITER*batch);	// ms per eval
	ta *= 1000./(SHT_ITER*batch);	// ms per eval
	printf("   3D vector SHT time (lmax=%d): \t synthesis %f ms \t analysis %f ms\n", LMAX, ts, ta);

	shtns_free(T2);	shtns_free(S2);	shtns_free(Q2);
	vect_error(Slm, Tlm, Slm0, Tlm0, LMAX);
	scal_error(Qlm, Tlm0, LMAX);
	return;
}

/*
fftw_plan ifft_in, ifft_out;
fftw_plan fft_in, fft_out;
fftw_plan fft_tr, ifft_tr;


// we want to test if in-place is faster than out-of place or not.
init_fft_tests()
{
	complex double *ShF, *Shout;
	double *Sh;
	int nfft, ncplx, nreal;
	unsigned fftw_plan_mode = FFTW_EXHAUSTIVE;		// defines the default FFTW planner mode.

	nfft = NPHI;
	ncplx = NPHI/2 +1;
	nreal = 2*ncplx;

// Allocate dummy Spatial Fields.
	ShF = (complex double *) shtns_malloc(ncplx * NLAT * sizeof(complex double));
	Sh = (double *) ShF;

// IFFT : unnormalized
	ifft_in = fftw_plan_many_dft_c2r(1, &nfft, NLAT, ShF, &ncplx, NLAT, 1, Sh, &nreal, NLAT, 1, fftw_plan_mode);
	if (ifft_in == NULL) printf("ifft_in failed\n");
// FFT : must be normalized.
	fft_in = fftw_plan_many_dft_r2c(1, &nfft, NLAT, Sh, &nreal, NLAT, 1, ShF, &ncplx, NLAT, 1, fftw_plan_mode);
	if (fft_in == NULL) printf("fft_in failed\n");
printf("in-place done\n");
	printf("** ifft in-place :\n");	fftw_print_plan(ifft_in);
	printf("\n** fft in-place :\n");	fftw_print_plan(fft_in);

	Shout = (complex double *) shtns_malloc(ncplx * NLAT * sizeof(complex double));
	ifft_out = fftw_plan_many_dft_c2r(1, &nfft, NLAT, Shout, &ncplx, NLAT, 1, Sh, &nfft, NLAT, 1, fftw_plan_mode);
	if (ifft_out == NULL) printf("ifft_out failed\n");
	fft_out = fftw_plan_many_dft_r2c(1, &nfft, NLAT, Sh, &nfft, NLAT, 1, Shout, &ncplx, NLAT, 1, fftw_plan_mode);
	if (fft_out == NULL) printf("fft_out failed\n");
printf("\nout-of-place done\n");
	printf("** ifft out-of-place :\n");	fftw_print_plan(ifft_out);
	printf("\n** fft out-of-place :\n");	fftw_print_plan(fft_out);

	ifft_tr = fftw_plan_many_dft_c2r(1, &nfft, NLAT, Shout, &ncplx, NLAT, 1, Sh, &nfft, 1, NPHI, fftw_plan_mode);
	if (ifft_out == NULL) printf("ifft_out failed\n");
	fft_tr = fftw_plan_many_dft_r2c(1, &nfft, NLAT, Sh, &nfft, 1, NPHI, Shout, &ncplx, NLAT, 1, fftw_plan_mode);
	if (fft_out == NULL) printf("fft_out failed\n");
printf("\ntranspose done\n");
	printf("** ifft + transpose :\n");	fftw_print_plan(ifft_tr);
	printf("\n** fft + transpose :\n"); fftw_print_plan(fft_tr);

	shtns_free(Shout);	shtns_free(ShF);
}

do_fft_tests()
{
	complex double *Sho;
	int jj;
	clock_t tcpu;

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		fftw_execute_dft_c2r(ifft_in, ShF, (double *) ShF);
	}
	tcpu = clock() - tcpu;
	printf("  ifft in-place : %d\n", (int) tcpu);

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		fftw_execute_dft_r2c(fft_in, (double *) ShF, ShF);
	}
	tcpu = clock() - tcpu;
	printf("  fft in-place : %d\n", (int) tcpu);

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		Sho = (complex double *) shtns_malloc( (NPHI/2+1) * NLAT * sizeof(complex double));
		fftw_execute_dft_c2r(ifft_out, Sho, (double *) ShF);
		shtns_free(Sho);
	}
	tcpu = clock() - tcpu;
	printf("  ifft out-of-place (+malloc) : %d\n", (int) tcpu);

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		Sho = (complex double *) shtns_malloc( (NPHI/2+1) * NLAT * sizeof(complex double));
		fftw_execute_dft_r2c(fft_out, (double *) ShF, Sho);
		shtns_free(Sho);
	}
	tcpu = clock() - tcpu;
	printf("  fft out-of-place (+malloc) : %d\n", (int) tcpu);

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		Sho = (complex double *) shtns_malloc( (NPHI/2+1) * NLAT * sizeof(complex double));
		fftw_execute_dft_c2r(ifft_tr, Sho, (double *) ShF);
		shtns_free(Sho);
	}
	tcpu = clock() - tcpu;
	printf("  ifft transpose (+malloc) : %d\n", (int) tcpu);

	tcpu = clock();
	for (jj=0; jj< SHT_ITER; jj++) {
		Sho = (complex double *) shtns_malloc( (NPHI/2+1) * NLAT * sizeof(complex double));
		fftw_execute_dft_r2c(fft_tr, (double *) ShF, Sho);
		shtns_free(Sho);
	}
	tcpu = clock() - tcpu;
	printf("  fft transpose (+malloc) : %d\n", (int) tcpu);

}
*/

void usage()
{
	printf("\nUsage: time_SHT lmax [options] \n");
	printf("        where lmax is the maxiumum spherical harmonic degree.\n");
	printf("** available options :\n");
	printf(" -mmax=<mmax> : defines the maximum spherical harmonic order <mmax>\n");
	printf(" -nphi=<nphi> : defines the number of azimutal (longitude) point\n");
	printf(" -nlat=<nlat> : defines the number of grid points in theta (latitude)\n");
	printf(" -mres=<mres> : the azimutal periodicity (1 for no symmetry; 2 for two-fold symmetry, ...)\n");
	printf(" -polaropt=<thr> : set the threshold for polar optimization. 0 for no polar optimization, 1.e-6 for agressive.\n");
	printf(" -iter=<n> : set the number of back-and-forth transforms to compute timings and errors.\n");
	printf(" -gauss : force gauss grid\n");
	printf(" -fly : force gauss grid with on-the-fly computations only\n");
	printf(" -quickinit : force gauss grid and fast initialiation time (but suboptimal fourier transforms)\n");
	printf(" -vector : time and test also vector transforms (2D and 3D)\n");
	printf(" -reg : use regular grid\n");
	printf(" -regpoles : use regular grid including poles\n");
	printf(" -oop : force out-of-place transform\n");
	printf(" -transpose : force transpose data (ie phi varies fastest)\n");
	printf(" -nlorder : define non-linear order to be resolved.\n");
	printf(" -schmidt : use schmidt semi-normalization.\n");
	printf(" -4pi : use 4pi normalization.\n");
	printf(" -robert : use Robert form, ie spatial vector fields are multiplied by sin(colatitude).\n");
	printf(" -loadsave : load and save config for faster startup.\n");
	printf(" -nogpu : disable GPU offload.\n");
	printf(" -nopadding : disable padding (may reduce performance).\n");
  #ifdef _OPENMP
	printf(" -nth=<n> : use n threads.\n");
  #endif
}

int main(int argc, char *argv[])
{
	complex double t1, t2;
	double t,tmax,n2;
	int nthreads = 0;
	int i,im,m,l;
	clock_t tcpu;
	double e0,e1;
	double polaropt = 1.e-8;		// default for polar optimization.
	enum shtns_type shtmode = sht_auto;		// default to "auto" (fastest) mode.
	enum shtns_norm shtnorm = sht_orthonormal;		// default to "orthonormal" SH.
	int layout = SHT_NATIVE_LAYOUT;
	int layout_opts = SHT_ALLOW_PADDING | SHT_ALLOW_GPU;		// allow padding and GPU by default
	int noltr = 0;
	int nlorder = 0;
	int point = 0;
	int vector = 0;
	int robert_form = -1;
	int accuracy_test = 0;
	char name[20];
	FILE* fw;

	srand( 42 );	// initialise les nombres aléatoires.
	shtns_verbose(2);		// output some diagnostics.

	printf("time_SHT performs some spherical harmonic transforms, and displays timings and accuracy.\n");
	if (argc < 2) {
		usage();	exit(1);
	}

//	first argument is lmax, and is mandatory.
	sscanf(argv[1],"%lf",&t);	LMAX=t;
	MMAX=-1;	MRES=1;

	for (i=2; i<argc; i++) {		// parse command line
		sscanf(argv[i],"-%[^=]=%lf",name,&t);
		if (strcmp(name,"mmax") == 0) MMAX = t;
		if (strcmp(name,"mres") == 0) MRES = t;
		if (strcmp(name,"nlat") == 0) NLAT = t;
		if (strcmp(name,"nphi") == 0) NPHI = t;
		if (strcmp(name,"polaropt") == 0) polaropt = t;
		if (strcmp(name,"iter") == 0) SHT_ITER = t;
		if (strcmp(name,"nth") == 0) nthreads = t;
		if (strcmp(name,"gauss") == 0) shtmode = sht_gauss;		// force gauss grid.
		if (strcmp(name,"fly") == 0) shtmode = sht_gauss_fly;		// force gauss grid with on-the-fly computation.
		if (strcmp(name,"reg") == 0) shtmode = sht_reg_fast;	// force regular grid.
		if (strcmp(name,"regpoles") == 0) shtmode = sht_reg_poles;	// force regular grid.
		if (strcmp(name,"quickinit") == 0) shtmode = sht_quick_init;	// Gauss grid and fast initialization time, but suboptimal fourier transforms.
		if (strcmp(name,"schmidt") == 0) shtnorm = sht_schmidt | SHT_NO_CS_PHASE;
		if (strcmp(name,"4pi") == 0) shtnorm = sht_fourpi | SHT_REAL_NORM;
		if (strcmp(name,"oop") == 0) layout = SHT_THETA_CONTIGUOUS;
		if (strcmp(name,"transpose") == 0) layout = SHT_PHI_CONTIGUOUS;
		if (strcmp(name,"nlorder") == 0) nlorder = t;
		if (strcmp(name,"vector") == 0) vector = 1;
		if (strcmp(name,"point") == 0) point = 1;
		if (strcmp(name,"loadsave") == 0) layout_opts |= SHT_LOAD_SAVE_CFG;
		if (strcmp(name,"robert") == 0) robert_form = t;
		if (strcmp(name,"nopadding") == 0) layout_opts &= ~SHT_ALLOW_PADDING;		// Disable padding.
		if (strcmp(name,"nogpu") == 0) layout_opts &= ~SHT_ALLOW_GPU;		// Disable gpu.
		if (strcmp(name,"float") == 0) layout_opts |= SHT_FP32;		// use float instead of double
		if (strcmp(name,"accuracy") == 0) accuracy_test = 1;			// Perform an accuracy test instead of a speed test.
		if (strcmp(name,"batch") == 0) batch = -1;		// Perform several transforms together.
		if (strcmp(name,"noltr") == 0) noltr = 1;
	}

	if (layout_opts & SHT_FP32) { err_err = 3e-3;  err_wrn = 3e-5; }
	if (vector == 0) layout_opts |= SHT_SCALAR_ONLY;
	printf("loadsave = %d\n", !!(layout_opts & SHT_LOAD_SAVE_CFG));
	if (MMAX == -1) MMAX=LMAX/MRES;
	shtns_use_threads(nthreads);		// 0 : means automatically chooses the number of threads.
	shtns = shtns_create(LMAX, MMAX, MRES, shtnorm);
	if (robert_form >= 0) shtns_robert_form(shtns, robert_form);		// keep the default robert_form, unless specified on command line (usefull when built4magic)
	NLM = ((shtns->nlm + 3)/4) *4;		// align on 64 bytes (cache line)
	if (batch == -1) {
		batch = SHT_ITER;		SHT_ITER = 1;
		int r = shtns_set_many(shtns, batch, NLM);
		if (r<0) printf("ERROR batch\n");
	}
	shtns_set_grid_auto(shtns, shtmode | layout | layout_opts, polaropt, nlorder, &NLAT, &NPHI);

	shtns_profiling(shtns, 1);		// enable internal profiling
	shtns_print_cfg(shtns);

/*
	t1 = 1.0+2.0*I;
	t2 = 1.0-I;
	printf("test : %f, %f, %f, %f\n",creal(t1),cimag(t1), creal(t2),cimag(t2));

	(double) t1 = 8.0 +I;
	(double) t2 = 8.1;
	printf("test : %f, %f, %f, %f\n",creal(t1),cimag(t1), creal(t2),cimag(t2));
*/
//	write_vect("cost",ct,NLAT);
//	write_vect("sint",st,NLAT);

	printf("nspat = %d, nlat = %d, nlat_padded = %d, iter=%d, batch=%d\n",shtns->nspat, shtns->nlat, shtns->nlat_padded, SHT_ITER, batch);
	ShF = (complex double *) shtns_malloc( shtns->nspat * sizeof(double));
	Sh = (double *) ShF;
	if (ShF == NULL) runerr("memory allocation 1 failed");
	if (vector) {
		ThF = (complex double *) shtns_malloc( shtns->nspat * sizeof(double));
		Th = (double *) ThF;
		NLF = (complex double *) shtns_malloc( shtns->nspat * sizeof(double));
		NL = (double *) NLF;
		if ((ThF == NULL)||(NLF == NULL)) runerr("memory allocation 2 failed");
	}

	Slm0 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	Slm = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	Tlm = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
	if ((Slm0 == NULL)||(Slm == NULL)||(Tlm == NULL)) runerr("memory allocation 3 failed");
	if (vector) {
		Tlm0 = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
		Qlm = (complex double *) shtns_malloc(sizeof(complex double)* NLM * batch);
		if ((Tlm0 == NULL)||(Qlm == NULL)) runerr("memory allocation 4 failed");
	}

// perform fft tests.
//	init_fft_tests();
//	do_fft_tests();
//	exit(0);

  if (NLM*batch < 10000) {
// SH_to_spat
	for (i=0;i<NLM*batch;i++) {
		Slm[i] = 0.0;
		if (vector) Tlm[i] = 0.0;
	}
	for (i=0;i<shtns->nspat;i++) {
		Sh[i] = 0.0;
	}
	for (int b=0; b<batch; b++) {
		Slm[LiM(shtns, 1,0) + b*NLM] = (b+1)*sh10_ct(shtns);
		if ((MMAX > 0)&&(MRES==1))	Slm[LiM(shtns, 1,1) + b*NLM] = (b+1)*sh11_st(shtns);
	}
//	write_vect("ylm0",Slm, NLM*2);
//	SH_to_spat_ml(shtns, 0,Slm, Sh, LMAX);
//	spat_to_SH_ml(shtns, 0,Sh, Slm, LMAX);
	SH_to_spat(shtns, Slm,Sh);
	write_mx("spat",Sh,NPHI,shtns->nlat_padded);
	if (vector) {
		SHtor_to_spat(shtns, Slm,Sh,Th);
		write_mx("spatt",Sh,NPHI,shtns->nlat_padded);
		write_mx("spatp",Th,NPHI,shtns->nlat_padded);
		SHtor_to_spat_l(shtns, Slm,Sh,Th,LMAX/2);
		write_mx("spatt_l",Sh,NPHI,shtns->nlat_padded);
		write_mx("spatp_l",Th,NPHI,shtns->nlat_padded);
	}

//	SHqst_to_lat(Slm,Slm,Tlm,ct[0],Sh,Th,Th,NPHI/2,LMAX,MMAX);
//	write_vect("spat_lat", Sh, NPHI/2);

/*	for (i=0;i<(NLAT/2)*NPHI;i++) {
		Sh[i] = 0.0;
	}
	SHeo_to_spat(Slm, Sh, 0);
	write_mx("spate",Sh,NPHI,NLAT/2);
	for (i=0;i<(NLAT/2)*NPHI;i++) {
		Th[i] = 0.0;
	}
	SHeo_to_spat(Slm, Th, 1);
	write_mx("spato",Th,NPHI,NLAT/2);

	for (i=0;i<NLM;i++)
		Slm[i] = 0.0;
	spat_to_SHeo(Sh, Slm, 0);
	write_vect("ylme",Slm, NLM*2);
	for (i=0;i<NLM;i++)
		Tlm[i] = 0.0;
	spat_to_SHeo(Th, Slm, 1);
	write_vect("ylmeo",Slm, NLM*2);
*/
// spat_to_SH
	for (im=0;im<NPHI;im++) {
		for (int b=0;b<batch;b++) {
			for (i=0;i<shtns->nlat;i++) Sh[im*shtns->nlat_padded+b*shtns->nlat+i] = (b+1) * shtns->ct[i];
		}
	}
	spat_to_SH(shtns, Sh,Slm);
	write_mx("ylm",(double *)Slm,batch, NLM*2);
  }

// test case...
	printf("generating random test case...\n");
	t = 1.0 / (RAND_MAX/2);
	for (i=0;i<NLM*batch;i++) {
		Slm0[i] = t*((double) (rand() - RAND_MAX/2)) + I*t*((double) (rand() - RAND_MAX/2));
		if (vector) Tlm0[i] = t*((double) (rand() - RAND_MAX/2)) + I*t*((double) (rand() - RAND_MAX/2));
	}
	for (int m=0; m<=MMAX; m++) {		// zero the last one.
	//	Slm0[LiM(shtns, LMAX,m)] = 0.0;	
	//	if (vector) Tlm0[LiM(shtns, LMAX, m)] = 0.0;
	}

	if (point) {
		test_SH_point(vector);
		exit(0);
	}


//	printf("** performing %d scalar SHT with NL evaluation\n", SHT_ITER);
	printf("** performing %d scalar SHT\n", SHT_ITER);
	printf(":: STD\n");
	if (accuracy_test) {
		test_SHT_accuracy();
		exit(error);
	}
	test_SHT();
	if (!noltr) {
		printf(":: LTR\n");
		test_SHT_l(LMAX/2);
	}

	if (vector) {
		for (int b=0;b<batch;b++) {
			for (int i=1;i<NLM;i++) {	// rescale amplitudes for vector transform (unit energy in each mode) -- except l=0
				int l = shtns->li[i];
				double scal = 1.0/sqrt(l*(l+1));
				Slm0[i + b*NLM] *= scal;
				Tlm0[i + b*NLM] *= scal;
			}
			Slm0[LM(shtns, 0,0) + b*NLM] = 0.0;	// l=0, m=0 n'a pas de signification sph/tor
			Tlm0[LM(shtns, 0,0) + b*NLM] = 0.0;	// l=0, m=0 n'a pas de signification sph/tor
			//for (i=0;i<NLM;i++) Slm0[i + b*NLM] = 0.0;	// zero out Slm.
		}

		printf("** performing %d vector SHT\n", SHT_ITER);
		printf(":: STD\n");
		test_SHT_vect();
		if (!noltr) {
			printf(":: LTR\n");
			test_SHT_vect_l(LMAX/2);
		}

		printf("** performing %d 3D vector SHT\n", SHT_ITER);
		printf(":: STD\n");
		test_SHT_vect3d();
		if (!noltr) {
			printf(":: LTR\n");
			test_SHT_vect3d_l(LMAX/2);
		}

		if (NPHI == 1) {		// test the special m=0 transforms
			printf("** performing %d m=0 gradient SHT\n", SHT_ITER);
			printf(":: STD\n");
			test_SHT_m0();
		}
	}

	{	// test forward and backward separately:
		const double a00 = sh00_1(shtns);
		const double a10 = sh10_ct(shtns);
		const cplx a11 = (MRES==1 && MMAX>0) ? sh11_st(shtns)*(cos(M_PI/12)+I*sin(M_PI/12)) : 0.0;	// with a phase shift.

		for (int ip=0; ip<NPHI; ip++) {
			double y11 = (MRES==1 && MMAX>0) ? cos(ip*2.*M_PI/NPHI + M_PI/12.) : 0;		// cos(phi + pi/12)
			for (int it=0;it<NLAT; it++) {
				long idx = (layout & SHT_PHI_CONTIGUOUS) ? it*shtns->nphi + ip : ip*shtns->nlat_padded + it;	// phi-contiguous or theta-contiguous layouts
				Sh[idx] = 1.0 + shtns->ct[it] + y11*shtns->st[it];
			}
		}
		spat_to_SH(shtns, Sh, Slm);
		double err = 0.0;
		int lm_max = -1;
		for (int lm=0; lm<shtns->nlm; lm++) {
			double t = cabs(Slm[lm]);
			if (lm==LiM(shtns,1,1)) {	t = cabs(Slm[lm] - a11);	if (t>1e-12) printf(COLOR_ERR "l=1,m=1 error = %g" COLOR_END "\n", t);  }
			if (lm==0) {  t = cabs(Slm[lm] - a00);	if (t>1e-12) printf(COLOR_ERR "l=0,m=0 error = %g" COLOR_END "\n", t);  }
			if (lm==1) {  t = cabs(Slm[lm] - a10);	if (t>1e-12) printf(COLOR_ERR "l=1,m=0 error = %g" COLOR_END "\n", t);  }
			if (err < t) {	err = t;	lm_max = lm;  }
		}
		printf("** Test forward transform (analysis) spat_to_SH() :: max err = %g   at lm=%d   ", err, lm_max);
		if (err > err_wrn) {		printf(COLOR_ERR "**** ERROR ****" COLOR_END "\n");	error++;	}
		else printf(COLOR_OK "OK" COLOR_END "\n");
		//write_vect("spec_analys_test.txt", (double*) Slm, 2*NLM);

		for (int lm=0; lm<NLM; lm++)	Slm[lm] = 0.0;
		Slm[0] = a00;
		Slm[1] = a10;
		if (MRES==1 && MMAX>0) Slm[LiM(shtns,1,1)] = a11;
		SH_to_spat(shtns, Slm, Sh);
		err = 0.0;
		for (int ip=0; ip<NPHI; ip++) {
			double y11 = (MRES==1 && MMAX>0) ? cos(ip*2.*M_PI/NPHI + M_PI/12.) : 0;		// cos(phi + pi/12)
			for (int it=0;it<NLAT; it++) {
				long idx = (layout & SHT_PHI_CONTIGUOUS) ? it*shtns->nphi + ip : ip*shtns->nlat_padded + it;	// phi-contiguous or theta-contiguous layouts
				double t = fabs(Sh[idx] - (1.0 + shtns->ct[it] + y11*shtns->st[it]));
				if (err < t) err = t;
			}
		}
		printf("** Test backward transform (synthesis) SH_to_spat() :: max err = %g   ", err);
		if (err > err_wrn) {		printf(COLOR_ERR "**** ERROR ****" COLOR_END "\n");	error++;	}
		else printf(COLOR_OK "OK" COLOR_END "\n");
	}

	{	// test Legendre only:
		const int im = (MMAX > 0) ? 1 : 0;
		if (im == 0) {
			shtns_free(ShF);
			ShF = shtns_malloc(sizeof(cplx) * shtns->nlat);  Sh = (double*) ShF;
			for (int i=0; i<=LMAX; i++) Slm0[i] = creal(Slm0[i]);
			if (vector) {
				shtns_free(ThF);
				ThF = (complex double *) shtns_malloc( sizeof(cplx) * shtns->nspat);  Th = (double *) ThF;
				for (int i=0; i<=LMAX; i++) Tlm0[i] = creal(Tlm0[i]);
			}
		}
		memset(Slm+im*(LMAX+1), 0, sizeof(cplx)*(LMAX-im*MRES+1));
		if (vector) {
			memset(Tlm+im*(LMAX+1), 0, sizeof(cplx)*(LMAX-im*MRES+1));
			SHsphtor_to_spat_ml(shtns, im, Slm0+im*(LMAX+1), Tlm0+im*(LMAX+1), (cplx*) Sh, (cplx*) Th, LMAX);
			spat_to_SHsphtor_ml(shtns, im, (cplx*) Sh, (cplx*) Th, Slm+im*(LMAX+1), Tlm+im*(LMAX+1), LMAX);
		} else {
			SH_to_spat_ml(shtns, im, Slm0+im*(LMAX+1), (cplx*) Sh, LMAX);
			spat_to_SH_ml(shtns, im, (cplx*) Sh, Slm+im*(LMAX+1), LMAX);
		}
		double err = 0.0;
		double err_v = 0.0;
		for (int i=0; i<=LMAX-im*MRES; i++) {
			double t = cabs(Slm[i+im*(LMAX+1)]-Slm0[i+im*(LMAX+1)]);
			err += t*t;
			if (vector) {
				double t = cabs(Tlm[i+im*(LMAX+1)]-Tlm0[i+im*(LMAX+1)]);
				err_v += t*t;
			}
			if (t > 1e-6) {
				printf("l=%d, Slm=%g,%g   Slm0=%g,%g\n", i, creal(Slm[i+im*(LMAX+1)]), cimag(Slm[i+im*(LMAX+1)]), creal(Slm0[i+im*(LMAX+1)]), cimag(Slm0[i+im*(LMAX+1)]));
				if (vector)
				printf("l=%d, Tlm=%g,%g   Tlm0=%g,%g\n", i, creal(Tlm[i+im*(LMAX+1)]), cimag(Tlm[i+im*(LMAX+1)]), creal(Tlm0[i+im*(LMAX+1)]), cimag(Tlm0[i+im*(LMAX+1)]));
			}
		}
		printf("** Test Legendre only (m=%d) :: err = %g   ",im*MRES,sqrt(err));
		if (sqrt(err) > 1e-4) {		printf(COLOR_ERR "**** ERROR ****" COLOR_END "\n");	error++;	}
		else printf(COLOR_OK "OK" COLOR_END "\n");
		if (vector) {
			printf("** Test Legendre only Vector (m=%d) :: err = %g   ",im*MRES,sqrt(err_v));
			if (sqrt(err_v) > 1e-4) {		printf(COLOR_ERR "**** ERROR ****" COLOR_END "\n");	error++;	}
			else printf(COLOR_OK "OK" COLOR_END "\n");
		}
	}

	shtns_create(LMAX, MMAX, MRES, shtnorm);		// test memory allocation and management.
//	shtns_create_with_grid(shtns, MMAX/2, 1);

// free memory and resources (to track memory leaks)
	shtns_free(Slm);		shtns_free(ShF);
	if (vector) {
		shtns_free(Qlm);		shtns_free(Tlm);		
		shtns_free(Slm0);		shtns_free(Tlm0);
		shtns_free(NLF);		shtns_free(ThF);
	}

	shtns_reset();
	fftw_cleanup();
	return error;
}
