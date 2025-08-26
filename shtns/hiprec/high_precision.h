/*
 * Copyright (c) 2024 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 * 
 * nathanael.schaeffer@cnrs.fr
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

/* real-real arithmetic for higher precision results: a simple collection of functions.
 * See Joldes+ 2017, doi:10.1145/3121432, for details on many algorithms
 * and also Muller & Rideau 2022, doi:10.1145/3484514 for some tighter bounds.
 * See also Rump & Lange 2020, doi:10.1145/3290955 for pair arithmetic (skipping "normalization")
 * 
 * In addition to classic operations, there are some:
 * 	- *_no_norm() variants that skip the final normalization step.
 *    It is often not needed to perform this normalization after each operation,
 *    especially if it is followed by an addition that does it "for free".
 *  - *_ordered() variants that assume first operand a is larger or equal in magnitude than
 *    second operand b.
 * 
 * Note that a fast (hardware) FMA operation is desirable. Otherwise it will be significantly slower.
*/

//#define FP_WORD 32

#include <math.h>	// for fma, sqrt

#if FP_WORD == 32
	typedef float real1;
#else
	typedef double real1;
#endif

#if defined( __AVX2__ ) || defined( FP_FAST_FMA ) || defined( __ARM_FEATURE_FMA )
	// rely on standard "fma" function being translated to hardware instruction
	#define HIPREC_NO_FMA 0
#else
	// no hardware instruction for fma, use our software implementation, which is
	// not accurate to last bit, but reasonably fast (faster than fully accurate software implementation)
	// and does not check for infs, nans, etc...
	#define HIPREC_NO_FMA 1
#endif

#ifdef __FAST_MATH__
#error "high_precision can't work with -ffast-math  (or /fp:fast). Possible options are: -fno-math-errno -fno-trapping-math -freciprocal-math -ffinite-math-only"
#endif

/* ADDITIONS */

// Also known as "2Sum". 6 ops (latency of 5 ops)
inline real1 add_err(real1 a, real1 b, real1* err)
{
	real1 s = a+b;
	real1 v = s-a;
	*err = (a-(s-v)) + (b-v);
	return s;
}

// Also known as "Fast2Sum". Assumes |a|>=|b|, faster than add_err (3 ops)
inline real1 add_ordered_err(real1 a, real1 b, real1* err)
{
	real1 s = a+b;
	*err = b - (s-a);
	return s;
}

// rh,rl = ah,al + bh,bl   with accuracy guaranteed < 3ulp
// 20 ops, (latency 14)
void dd_add(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 sh,sl,th,tl;
	sh = add_err(ah,bh, &sl);
	th = add_err(al,bl, &tl);
	sl += th;
	sh = add_ordered_err(sh,sl, &sl);	// normalize
	sl += tl;
	*rh = add_ordered_err(sh,sl, rl);
}

// rh,rl = ah,al + b   with accuracy guaranteed < 2ulp
// 10 ops
void dd_add_d(real1* rh, real1* rl, real1 ah, real1 al, real1 b)
{
	real1 sh,sl;
	sh = add_err(ah,b, &sl);
	sl += al;
	*rh = add_ordered_err(sh,sl, rl);	// normalize
}

// rh,rl = ah,al + bh,bl 	Assuming |ah|>=|bh|
// 14 ops (latency 11)
void dd_add_ordered(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 sh,sl,th,tl;
	sh = add_ordered_err(ah,bh, &sl);
	th = add_ordered_err(al,bl, &tl);
	sl += th;
	sh = add_ordered_err(sh,sl, &sl);	// normalize
	sl += tl;
	*rh = add_ordered_err(sh,sl, rl);
}

// rh,rl = ah,al + b   Assuming |ah|>=|b|
// 7 ops
void dd_add_d_ordered(real1* rh, real1* rl, real1 ah, real1 al, real1 b)
{
	real1 sh,sl;
	sh = add_ordered_err(ah,b, &sl);
	sl += al;
	*rh = add_ordered_err(sh,sl, rl);	// normalize
}

// rh,rl = ah,al + bh,bl   with accuracy guaranteed only if ah and bh have the same sign!!
// WARNING: accuracy guaranteed (4ulp) ONLY IF a AND b ARE OF THE SAME SIGN, see Joldes+ 2017, doi:10.1145/3121432, algo 5
// In practice, only if catastrophic cancellation occurs when adding ah and bh is the relative accuracy degraded; the absolute accuracy is always 4 ulp!
// 11 ops
inline void dd_add_sloppy(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 sh,sl;
	sh = add_err(ah,bh, &sl);
	sl += al+bl;
	*rh = add_ordered_err(sh, sl, rl);
}

// same as above, assuming |ah|>=|bh|
// 8 ops (latency 7)
void dd_add_sloppy_ordered(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 sh,sl;
	sh = add_ordered_err(ah,bh, &sl);
	sl += al+bl;
	*rh = add_ordered_err(sh, sl, rl);
}

/* MULTIPLICATIONS */

/*
	#ifdef __SSE2__
	#include <emmintrin.h>
	#endif

// does not overflow, but subnormals may lead to hi=0, lo=x1 (which is probably ok)
// seems to work, but not proved, unlike actual Veltkamp splitting... (more costly)
inline void split_bits_x2(double x1, double* h1, double *l1, double x2, double* h2, double* l2)
{
	//const unsigned long long mask = 0xFFFFFFFFF8000000ull;
	const unsigned long long mask =   0xFFFFFFFFF8000000ull;
    #ifdef __SSE2__
	__m128d xx = _mm_set_pd(x1,x2);
	__m128d xh = _mm_castsi128_pd( _mm_add_epi64( _mm_castpd_si128(xx), _mm_set1_epi64x((~mask)>>1)) );	// effective rounding done using integer add
	xh = _mm_and_pd( xh, _mm_set1_pd( *(double*)&mask ) );
	xx = _mm_sub_pd(xx,xh);
	*h1 = xh[0];	*h2 = xh[1];
	*l1 = xx[0];	*l2 = xx[1];
    #else
	long i1 = *(long*)&x1;
	long i2 = *(long*)&x2;
	i1 += (~mask)>>1;		// may overflow
	i2 += (~mask)>>1;		// may overflow
	i1 &= mask;	// simply zeros the 27 lower bits
	i2 &= mask;	// simply zeros the 27 lower bits
	*h1 = *(double*)&i1;
	*h2 = *(double*)&i2;
	*l1 = x1-*h1;
	*l2 = x2-*h2;
    #endif
}

// expected latency: 5 ops
double mul_err_nofma_mask(double a, double b, double* err)
{
	#pragma FP_CONTRACT OFF
	double ah,al, bh,bl;
	double ab = a*b;
	split_bits_x2(a, &ah,&al, b, &bh,&bl);
	double e = ah*bh - ab;	// mostly cancel
	e += ah*bl;
	e += al*bh;			// more cancellation
	*err = e + al*bl;		// al*bl must come last: after cancellation occured above
	return ab;
}
*/

#if HIPREC_NO_FMA

// Proved error-free multiplication, see https://toccata.gitlabpages.inria.fr/toccata/gallery/Dekker.en.html
// expected latency: 8 ops
inline double mul_err_nofma(double x, double y, double* err) {
	const double C=0x8000001;		// 2^27 + 1
	double xy,px,qx,hx,py,qy,hy,tx,ty,r2;

	// first perform Veltkamp splitting:
	px=x*C;		// may overflow
	py=y*C;		// may overflow
	xy = x*y;
	qx=x-px;
	qy=y-py;
	hx=px+qx;
	hy=py+qy;
	tx=x-hx;
	ty=y-hy;

	// then subtract each partaial product in order (exact)
	r2=hx*hy - xy;
	r2+=hy*tx;
	r2+=hx*ty;
	r2+=tx*ty;		// must come last
	*err = r2;
	return xy;
}

// compared to mul_err_nofma(x,x, err), this saves quite some cycles.
inline double square_err_nofma(double x, double* err) {
	const double C=0x8000001;		// 2^27 + 1
	double x2,px,qx,hx,tx,r2;

	// first perform Veltkamp splitting:
	px=x*C;		// may overflow
	x2 = x*x;
	qx=x-px;
	hx=px+qx;
	tx=x-hx;

	// then subtract each partaial product in order (exact)
	r2 = hx*hx - x2;
	qx = hx*tx;
	r2 += (qx+qx);	// 2*hx*tx, exact
	r2 += tx*tx;		// must come last
	*err = r2;
	return x2;
}

// expected latency: 16 ops (when we accept an error in the last bit)
double fma_sloppy(double a, double b, double c)
{
	double e, sl;
    double ab = mul_err_nofma(a,b, &e);
	double sh = add_err(ab,c, &sl);
  #ifndef HIPREC_EXACT_FMA
	sl += e;	// this should be "round to odd" for the result to be exact. See Boldo & Melquiond 2008, doi:10.1109/TC.2007.70819 and https://github.com/JuliaMath/openlibm/blob/master/src/s_fma.c
  #else
	// very costly (branching, bitcast to integer), don't do it
	sl = add_err(sl,e, &e);
	long long hi = *(long long*)&sl;
	long long lo = *(long long*)&e;
	if ((hi&1)==0 && lo!=0) {	// need adjusting when last bit of hi is zero, and lo!=0
		hi += 1 - 2*((hi^lo)>>63);
		sl = *(double*)&hi;
	}
  #endif
	return sh+sl;
}

inline float fmaf_sloppy(float a, float b, float c)
{
	// here also, double rounding may occur, so that the last bit may be wrong.
	return (float)( ((double)a) * ((double)b) + ((double)c) );
}

	#define FMA_FAST fma_slopy
	#warning "slow fma"
#else
	#define FMA_FAST fma
#endif


// Also known as "2Prod". 2 ops thanks to fma.
inline real1 mul_err(real1 a, real1 b, real1* err)
{
  #if HIPREC_NO_FMA == 0
	real1 x = a*b;
	*err = fma(a,b,-x);
	return x;
  #else
	return mul_err_nofma(a,b, err);
  #endif
}

// rh,rl = ah,al * bh,bl
// error bound of 4ulp (see algo 12 in Joldes+ 2017, and Muller&Rideau 2022 for the correct error bound)
// this one has the advantage of NOT requiring normalized inputs to ensure accuracy.
void dd_mul_accurate(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 ph, pl, pll;
	ph = mul_err(ah,bh, &pl);
	pll = al*bl;
	pll += al*bh;
	pll += bl*ah;
	*rh = add_ordered_err(ph,pl+pll, rl);
}

// rh,rl = ah,al * bh,bl
// error bound of 5ulp (see algo 11 in Joldes+ 2017, and Muller&Rideau 2022 for the correct error bound)
void dd_mul(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 ph, pl, pll;
	ph = mul_err(ah,bh, &pl);
	pll = al*bh + bl*ah;
	*rh = add_ordered_err(ph,pl+pll, rl);
}

// same as dd_mul, but does not normalize the number at the end, allowing overlap of rh and rl.
// also, does not require input to be normalized to give an accurate result.
inline void dd_mul_no_norm(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 ph, pl, pll;
	ph = mul_err(ah,bh, &pl);
	pll = al*bl;
	pll += al*bh;
	pll += bl*ah;
	*rl = pl + pll;
	*rh = ph;
}

// rh,rl = ah,al * b
// error bound of 2ulp (see Joldes+ 2017)
void dd_mul_d(real1* rh, real1* rl, real1 ah, real1 al, real1 b)
{
	real1 ph, pl;
	ph = mul_err(ah,b, &pl);
	pl += al*b;
	*rh = add_ordered_err(ph,pl, rl);
}

// same as dd_mul_d, but does not normalize the number at the end, allowing overlap of rh and rl.
void dd_mul_d_no_norm(real1* rh, real1* rl, real1 ah, real1 al, real1 b)
{
	real1 ph, pl;
	ph = mul_err(ah,b, &pl);
	*rl = pl + al*b;
	*rh = ph;
}

/* DIVISIONS */

inline real1 div_err(real1 a, real1 b, real1* err)
{
	real1 q1 = a/b;
  #if HIPREC_NO_FMA == 0
	real1 d = fma(-b,q1,a);		// a - b*q1,  EXACT error term (see Theorem 4 in Boldo and Daumas (2003))
  #else
	real1 bq1, e;
	bq1 = mul_err(b,q1, &e);
	real1 d = a-bq1;	// mostly cancels, exact
	d -= e;
  #endif
	*err = d/b;
	return q1;
}
/*
void dd_div_test((real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 q1 = ah/bh;
  #if HIPREC_NO_FMA == 0
	real1 s1 = fma(-bh,q1,ah);	// most of ah cancels out, same order as al, EXACT (see Theorem 4 in Boldo and Daumas (2003))
	real1 s2 = fma(-bl,q1,al);	// also same order as al
  #else
	real1 th,tl;
	th = mul_err(bh,q1, &tl);			// ph=bh*q1;	pl=fma(bh,q1,-bh*q1)
	real s1 = ah - th;		// mostly cancels, exact
	s1 -= tl;
	real s2 = al - bl*q1;
  #endif
	real1 q2 = (s1+s2)/bh;
	*rh = add_ordered_err(q1,q2,rl);	
}
*/

// rh,rl = (ah+al) / (bh+bl)	algo 17 from Joldes+ 2017, modified/optimze with fma
// (ah+al)/(bh+bl) approx= (ah+al)/bh * (1-bl/bh) = ah/bh + (div_err + (al - bl*(ah/bh)))/bh
void dd_div(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 q1 = ah/bh;
  #if HIPREC_NO_FMA == 0
	real1 s1 = fma(-bh,q1,ah);	// most of ah cancels out, same order as al, EXACT (see Theorem 4 in Boldo and Daumas (2003))
	real1 s2 = fma(-bl,q1,al);	// also same order as al
  #else
	real1 th,tl;
	dd_mul_d(&th,&tl, bh,bl, q1);		// approx ah
	real1 s1 = ah - th;	// exact
	real1 s2 = al - tl;
  #endif
	real1 q2 = (s1+s2)/bh;
	*rh = add_ordered_err(q1,q2,rl);
}

// rh,rl = (ah+al) / (bh+bl)		Same as dd_div, but compute 1/bh to save one div (replaced by 2*, so the difference is not that large)
void dd_div_fast(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 b_1 = 1.0/bh;
	real1 q1 = ah * b_1;
  #if HIPREC_NO_FMA == 0
	real1 s1 = fma(-bh,q1,ah);	// most of ah cancels out, same order as al
	real1 s2 = fma(-bl,q1,al);	// also same order as al
  #else
	real1 th,tl;
	dd_mul_d(&th,&tl, bh,bl, q1);		// approx ah
	real1 s1 = ah - th;	// exact
	real1 s2 = al - tl;
  #endif
	real1 q2 = (s1+s2)*b_1;
	*rh = add_ordered_err(q1,q2,rl);
}

// modified version of CPairDiv from Rump & Lange 2020, doi:10.1145/3290955
void dd_div_no_norm(real1* rh, real1* rl, real1 ah, real1 al, real1 bh, real1 bl)
{
	real1 q1 = ah/bh;
  #if HIPREC_NO_FMA == 0
	real1 s1 = fma(-bh,q1,ah);	// most of ah cancels out, EXACT, but can be smaller than al in a non-normalized context
	real1 s2 = fma(-bl,q1,al);
  #else
	real1 th,tl;
	dd_mul_d(&th,&tl, bh,bl, q1);		// approx ah
	real1 s1 = ah - th;	// exact
	real1 s2 = al - tl;	
  #endif
	real1 q2 = (s1+s2)/(bh+bl);		// bh+bl != bh only if input is not normalized
	*rh = q1;
	*rl = q2;
}


/* SQUARE ROOTS */

inline real1 sqrt_err(real1 a, real1* err)
{
	real1 s = sqrt(a);
  #if HIPREC_NO_FMA == 0
	real1 d = fma(-s,s,a);		// error :  a=s*s+d = s*s*(1+d/(s*s))   ==> sqrt(a) = s*sqrt(1+d/s2) approx= s*(1+d/(2s2)) = s + d/(2*s)
  #else
	real1 s2,e;
	s2 = square_err_nofma(s, &e);		// s2+e = s*s
	real1 d = a-s2;		// mostly cancels, EXACT
	d -= e;
  #endif
	*err = (a==0) ? a : d/(s+s);
	return s;
}

// rh,rl = sqrt(ah+al). Note: a non-normalized input will give a less accurate result
// Algo 8 of Lefevre+ 2023, doi:10.1145/3568672  https://hal.science/hal-03482567/document
// error bound of 3.125 ulp (with fma)
// assumes that if ah==0, then al==0 (that is a normalized input)
void dd_sqrt(real1* rh, real1* rl, real1 ah, real1 al)
{
	real1 s = sqrt(ah);		// sqrt(ah)
  #if HIPREC_NO_FMA == 0
	real1 d = fma(-s,s,ah);		// d = ah-s*s,  same order of magnitude as al
  #else
	real1 s2,e;
	s2 = square_err_nofma(s, &e);	// s2+e = s*s
	real1 d = ah-s2;		// mostly cancels, EXACT
	d -= e;
  #endif
	d += al;			// d = a-s*s
	if (ah != 0) ah = d/(s+s);
	*rh = add_ordered_err(s,ah, rl);
}

// same as dd_sqrt, but assuming ah>0, and skipping normalization
// Note: a non-normalized input will give a less accurate result
void dd_sqrt_nz_no_norm(real1* rh, real1* rl, real1 ah, real1 al)
{
	real1 s = sqrt(ah);		// sqrt(ah)
  #if HIPREC_NO_FMA == 0
	real1 d = fma(-s,s,ah);		// d = ah-s*s,  same order of magnitude as al
  #else
	real1 s2,e;
	s2 = square_err_nofma(s, &e);		// s2+e = s*s
	real1 d = ah-s2;		// mostly cancels, EXACT
	d -= e;
  #endif
	d += al;						// a-s*s = d
	*rl = d/(s+s);
	*rh = s;
}

