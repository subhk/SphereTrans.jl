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

/** \internal \file sht_legendre.c
 \brief Compute legendre polynomials and associated functions.
 
 - Normalization of the associated functions for various spherical harmonics conventions are possible, with or without Condon-Shortley phase.
 - When computing the derivatives (with respect to colatitude theta), there are no singularities.
 - written by Nathanael Schaeffer / CNRS, with some ideas and code from the GSL 1.13 and Numerical Recipies.

 The associated legendre functions are computed using the following recurrence formula :
 \f[  Y_l^m(x) = a_l^m \, x \  Y_{l-1}^m(x) + b_l^m \ Y_{l-2}^m(x)  \f]
 with starting values :
 \f[  Y_m^m(x) = a_m^m \ (1-x^2)^{m/2}  \f]
 and
 \f[  Y_{m+1}^m(x) = a_{m+1}^m \ Y_m^m(x)  \f]
 where \f$ a_l^m \f$ and \f$ b_l^m \f$ are coefficients that depend on the normalization, and which are
 precomputed once for all by \ref legendre_precomp.
*/

// index in alm array for given im.
#define alm_im(shtns, im) (shtns->alm + (im)*(2*(shtns->lmax+1) - ((im)-1)*shtns->mres))

#if SHT_VERBOSE > 1
  #define LEG_RANGE_CHECK
#endif

// use quadmath for initilization (gauss_nodes, recurrence coefficients,...)
//#define SHTNS_QUADMATH
// use double-double arithmetic for high-precision pre-computations:
#define SHTNS_HIPREC

#ifdef SHTNS_QUADMATH
//  #warning "quad precision"
  #include <quadmath.h>
  typedef __float128 real;
  #define SQRT sqrtq
  #define COS cosq
  #define SIN sinq
  #define FABS fabsq
  #define M_PIr M_PIq
  #undef SHTNS_HIPREC
#else
  typedef double real;
  #define SQRT sqrt
  #define COS cos
  #define SIN sin
  #define FABS fabs
  #define M_PIr M_PI
  #ifdef SHTNS_HIPREC
//    #warning "double-double"
    #include "hiprec/high_precision.h"
  #endif
#endif




/// \internal computes sin(t)^n from cos(t). ie returns (1-x^2)^(n/2), with x = cos(t)
/// assumes: -1 <= cost <= 1, n>=0.
/// writes nval, and returns val such as the result is val.SHT_SCALE_FACTOR^(nval)
static double sint_pow_n_ext(double cost, int n, int *nval)
{
	double s2 = 1. - cost*cost;		// sin(t)^2 = 1 - cos(t)^2 >= 0
	int ns2 = 0;
	int nv = 0;

#ifdef LEG_RANGE_CHECK
	if (s2 < 0) return NAN;		// sin(t)^2 < 0 !!!
#endif

	double val = 1.0;		// val >= 0
	if (n&1) val *= sqrt(s2);	// = sin(t)
	while (n >>= 1) {
		if (n&1) {
			if (val < 1.0/SHT_SCALE_FACTOR) {		// val >= 0
				nv--;		val *= SHT_SCALE_FACTOR;
			}
			val *= s2;		nv += ns2;		// 1/S^2 < val < 1
		}
		s2 *= s2;		ns2 += ns2;
		if (s2 < 1.0/SHT_SCALE_FACTOR) {	// s2 >= 0
			ns2--;		s2 *= SHT_SCALE_FACTOR;		// 1/S < s2 < 1
		}
	}
	while ((nv < 0) && (val > 1.0/SHT_SCALE_FACTOR)) {	// try to minimize |nv|
		++nv;	val *= 1.0/SHT_SCALE_FACTOR;
	}
	*nval = nv;
	return val;		// 1/S^2 < val < 1
}


/// \internal Returns the value of a legendre polynomial of degree l and order im*MRES, normalized for spherical harmonics, using recurrence.
/// Requires a previous call to \ref legendre_precomp().
/// Output compatible with the GSL function gsl_sf_legendre_sphPlm(l, m, x)
static double legendre_sphPlm(shtns_cfg shtns, const int l, const int im, const double x)
{
	double *al;
	int m, ny;
	double ymm, ymmp1;

	m = im*MRES;
#ifdef LEG_RANGE_CHECK
	if ( (l>LMAX+1) || (l<m) || (im>MMAX) ) shtns_runerr("argument out of range in legendre_sphPlm");
#endif

	ny = 0;
	al = alm_im(shtns, im);
	ymm = al[0];
	if (m>0) ymm *= sint_pow_n_ext(x, m, &ny);	// ny <= 0

	ymmp1 = ymm;			// l=m
	if (l == m) goto done;

	ymmp1 = al[1] * (x*ymmp1);		// l=m+1
	if (l == m+1) goto done;

	m+=2;	al+=2;
	while ((ny < 0) && (m < l)) {		// take care of scaled values
		ymm   = al[1]*(x*ymmp1) + al[0]*ymm;
		ymmp1 = al[3]*(x*ymm) + al[2]*ymmp1;
		m+=2;	al+=4;
		if (fabs(ymm) > 1.0/SHT_SCALE_FACTOR) {		// rescale when value is significant
			++ny;	ymm *= 1.0/SHT_SCALE_FACTOR;	ymmp1 *= 1.0/SHT_SCALE_FACTOR;
		}
	}
	while (m < l) {		// values are unscaled.
		ymm   = al[1]*(x*ymmp1) + al[0]*ymm;
		ymmp1 = al[3]*(x*ymm) + al[2]*ymmp1;
		m+=2;	al+=4;
	}
	if (m == l) {
		ymmp1 = al[1]*(x*ymmp1) + al[0]*ymm;
	}
done:
	if (ny < 0) {
		if (ny+3 < 0) return 0.0;
		do { ymmp1 *= 1.0/SHT_SCALE_FACTOR; } while (++ny < 0);		// return unscaled values.
	}
	return ymmp1;
}


/// \internal Compute values of legendre polynomials noramalized for spherical harmonics,
/// for a range of l=m..lmax, at given m and x, using recurrence.
/// Requires a previous call to \ref legendre_precomp().
/// Output compatible with the GSL function gsl_sf_legendre_sphPlm_array(lmax, m, x, yl)
/// \param lmax maximum degree computed, \param im = m/MRES with m the SH order, \param x argument, x=cos(theta).
/// \param[out] yl is a double array of size (lmax-m+1) filled with the values.
/// \returns the first degree l of non-zero value.
int legendre_sphPlm_array(shtns_cfg shtns, const int lmax, const int im, const double x, double *yl)
{
	const double *al;
	double ymm, ymmp1;
	int l, ny, lnz;

	const int m = im*MRES;
#ifdef LEG_RANGE_CHECK
	if ( (lmax>LMAX+1) || (lmax<m) || (im>MMAX) ) shtns_runerr("argument out of range in legendre_sphPlm");
#endif

	al = alm_im(shtns, im);
	yl -= m;			// shift pointer
	lnz = m;			// all non-zero a priori

	ny = 0;
	ymm = al[0];
	if (m>0) ymm *= sint_pow_n_ext(x, m, &ny);	// l=m,  ny <= 0

	l=m+2;	al+=2;
	if (ny<0) {
		yl[m] = 0.0;	lnz++;
		if (lmax==m) return lnz;
		ymmp1 = ymm * (al[-1] * x);		// l=m+1
		yl[m+1] = 0.0;	lnz++;
		if (lmax==m+1) return lnz;
		while (l < lmax) {		// values are negligible => discard.
			ymm   = (al[1]*x)*ymmp1 + al[0]*ymm;
			ymmp1 = (al[3]*x)*ymm   + al[2]*ymmp1;
			yl[l] = 0.0;	yl[l+1] = 0.0;
			l+=2;	al+=4;		lnz+=2;
			if (fabs(ymm) > 1.0) {		// rescale when value is significant
				ymm *= 1.0/SHT_SCALE_FACTOR;	ymmp1 *= 1.0/SHT_SCALE_FACTOR;
				if (++ny == 0) goto ny_zero;
			}
		}
		if (l == lmax) {
			yl[l] = 0.0;	lnz++;
		}
		return lnz;
	}
	yl[m] = ymm;
	if (lmax==m) return lnz;
	ymmp1 = ymm * (al[-1] * x);		// l=m+1
	yl[m+1] = ymmp1;
	if (lmax==m+1) return lnz;
  ny_zero:
	while (l < lmax) {		// values are unscaled => store
		ymm   = (al[1]*x)*ymmp1 + al[0]*ymm;
		ymmp1 = (al[3]*x)*ymm   + al[2]*ymmp1;
		yl[l] = ymm;		yl[l+1] = ymmp1;
		l+=2;	al+=4;
	}
	if (l == lmax) {
		yl[l] = (al[1]*x)*ymmp1 + (al[0]*ymm);
	}
	return lnz;
}


/// \internal Compute values of a legendre polynomial normalized for spherical harmonics derivatives, for a range of l=m..lmax, using recurrence.
/// Requires a previous call to \ref legendre_precomp(). Output is not directly compatible with GSL :
/// - if m=0 : returns ylm(x)  and  d(ylm)/d(theta) = -sin(theta)*d(ylm)/dx
/// - if m>0 : returns ylm(x)/sin(theta)  and  d(ylm)/d(theta).
/// This way, there are no singularities, everything is well defined for x=[-1,1], for any m.
/// \param lmax maximum degree computed, \param im = m/MRES with m the SH order, \param x argument, x=cos(theta).
/// \param sint = sqrt(1-x^2) to avoid recomputation of sqrt.
/// \param[out] yl is a double array of size (lmax-m+1) filled with the values (divided by sin(theta) if m>0)
/// \param[out] dyl is a double array of size (lmax-m+1) filled with the theta-derivatives.
/// \returns the first degree l of non-zero value.
int legendre_sphPlm_deriv_array(shtns_cfg shtns, const int lmax, const int im, const double x, const double sint, double *yl, double *dyl)
{
	const double *al;
	double st, y0, y1, dy0, dy1;
	int l, ny, lnz;

	const int m = im*MRES;
#ifdef LEG_RANGE_CHECK
	if ((lmax > LMAX+1)||(lmax < m)||(im>MMAX)) shtns_runerr("argument out of range in legendre_sphPlm_deriv_array");
#endif

	al = alm_im(shtns, im);
	yl -= m;	dyl -= m;			// shift pointers
	lnz = m;			// all non-zero apriori

	ny = 0;
	st = sint;
	y0 = al[0];
	dy0 = 0.0;
	if (m>0) {
		y0 *= sint_pow_n_ext(x, m-1, &ny);
		dy0 = x*m*y0;
		st *= st;		// st = sin(theta)^2 is used in the recurrence for m>0
	}
	y1 = al[1] * (x * y0);		// l=m+1
	dy1 = al[1]*( x*dy0 - st*y0 );

	l=m+2;	al+=2;
	if (ny<0) {
		yl[m] = 0.0;	dyl[m] = 0.0;		lnz++;
		if (lmax==m) return lnz;		// done.
		yl[m+1] = 0.0;	dyl[m+1] = 0.0;		lnz++;
		if (lmax==m+1) return lnz;
		while (l < lmax) {		// values are negligible => discard.
			y0 = (al[1]*x)*y1 + al[0]*y0;
			dy0 = al[1]*(x*dy1 - y1*st) + al[0]*dy0;
			y1 = (al[3]*x)*y0 + al[2]*y1;
			dy1 = al[3]*(x*dy0 - y0*st) + al[2]*dy1;
			yl[l] = 0.0;	yl[l+1] = 0.0;
			dyl[l] = 0.0;	dyl[l+1] = 0.0;
			l+=2;	al+=4;	lnz+=2;
			if (fabs(y0) > 1.0) {		// rescale when value is significant
				y0 *= 1.0/SHT_SCALE_FACTOR;		dy0 *= 1.0/SHT_SCALE_FACTOR;
				y1 *= 1.0/SHT_SCALE_FACTOR;		dy1 *= 1.0/SHT_SCALE_FACTOR;
				if (++ny == 0) goto ny_zero;
			}
		}
		if (l == lmax) {
			yl[l] = 0.0;	dyl[l] = 0.0;	lnz++;
		}
		return lnz;
	}
	yl[m] = y0; 	dyl[m] = dy0;		// l=m
	if (lmax==m) return lnz;		// done.
	yl[m+1] = y1; 	dyl[m+1] = dy1;		// l=m+1
	if (lmax==m+1) return lnz;		// done.
  ny_zero:
	while (l < lmax) {		// values are unscaled => store.
		y0 = al[1]*(x*y1) + al[0]*y0;
		dy0 = al[1]*(x*dy1 - y1*st) + al[0]*dy0;
		y1 = al[3]*(x*y0) + al[2]*y1;
		dy1 = al[3]*(x*dy0 - y0*st) + al[2]*dy1;
		yl[l] = y0;		dyl[l] = dy0;
		yl[l+1] = y1;	dyl[l+1] = dy1;
		l+=2;	al+=4;
	}
	if (l==lmax) {
		yl[l] = al[1]*(x*y1) + al[0]*y0;
		dyl[l] = al[1]*(x*dy1 - y1*st) + al[0]*dy0;
	}
	return lnz;
}

/// Same as legendre_sphPlm_deriv_array for x=0 and sint=1 (equator).
/// Depending on the parity of l, either ylm or dylm/dtheta is zero. So we store only the non-zero values.
static void legendre_sphPlm_deriv_array_equ(shtns_cfg shtns, const int lmax, const int im, double *ydyl)
{
	double *al;
	int l,m;
	double y0, dy1;

	m = im*MRES;
#ifdef LEG_RANGE_CHECK
	if ((lmax > LMAX+1)||(lmax < m)||(im>MMAX)) shtns_runerr("argument out of range in legendre_sphPlm_deriv_array");
#endif

	al = alm_im(shtns, im);
	ydyl -= m;			// shift pointer

	y0 = al[0];
	ydyl[m] = y0;		// l=m
	if (lmax==m) return;		// done.

	dy1 = -al[1]*y0;
	ydyl[m+1] = dy1;		// l=m+1
	if (lmax==m+1) return;		// done.

	l=m+2;	al+=2;
	while (l < lmax) {
		y0 = al[0]*y0;
		dy1 = al[2]*dy1 - al[3]*y0;
		ydyl[l]   = y0;
		ydyl[l+1] = dy1;
		l+=2;	al+=4;
	}
	if (l==lmax) {
		ydyl[l] = al[0]*y0;
	}
}

/// \internal Precompute constants for the recursive generation of Legendre associated functions, with given normalization.
/// this function is called by \ref shtns_set_size, and assumes up-to-date values in \ref shtns.
/// For the same conventions as GSL, use \c legendre_precomp(sht_orthonormal,1);
/// \param[in] norm : normalization of the associated legendre functions (\ref shtns_norm).
/// \param[in] with_cs_phase : Condon-Shortley phase (-1)^m is included (1) or not (0)
/// \param[in] mpos_renorm : Optional renormalization for m>0.
///  1.0 (no renormalization) is the "complex" convention, while 0.5 leads to the "real" convention (with FFTW).
void legendre_precomp(shtns_cfg shtns, enum shtns_norm norm, int with_cs_phase, double mpos_renorm)
{
	double *alm;
	real t1, t2;
	const long int lmax = LMAX+1;		// we go to one order beyond LMAX to resolve vector transforms through scalar ones.

#if HAVE_LONG_DOUBLE_WIDER
	test_long_double();
#endif

	if (with_cs_phase != 0) with_cs_phase = 1;		// force to 1 if !=0

	alm = (double *) malloc( (2*NLM + 2)*sizeof(double) );		//  fits exactly into an array of 2*NLM doubles, + 2 values to allow overflow read.
	if (alm==0) shtns_runerr("not enough memory.");
	shtns->alm = alm;

/// - Compute and store the prefactor (independant of x) of the starting value for the recurrence :
/// \f[  Y_m^m(x) = Y_0^0 \ \sqrt{ \prod_{k=1}^{m} \frac{2k+1}{2k} } \ \ (-1)^m \ (1-x^2)^{m/2}  \f]
	if (norm != sht_orthonormal) {
		t1 = 1.0;
		alm[0] = t1;		/// \f$ Y_0^0 = 1 \f$  for Schmidt or 4pi-normalized 
	} else {
		t1 = 0.25L/M_PIl;
		alm[0] = SQRT(t1);		/// \f$ Y_0^0 = 1/\sqrt{4\pi} \f$ for orthonormal
	}
	t1 *= mpos_renorm;		// renormalization for m>0
	real e=0.0;
	for (int im=1, m=0; im<=MMAX; ++im) {
		while(m<im*MRES) {
			++m;
			real x = ((real)m + 0.5)/m;
		#if HAVE_LONG_DOUBLE_WIDER
			t1 *= x;	// t1 *= (m+0.5)/m;
		#else
			// compensated product algorithm, see Algorithm 3.4 of https://hal.archives-ouvertes.fr/hal-00164607
			// => gets some extra bits of precision at negligible cost (which is dominated by the division above)
			real tt = t1*x;
			real t1e = fma(t1,x,-tt);	// = t1*x -tt  (C99)
			t1 = tt;
			e = fma(e,x,t1e);	// = e*x+t1e (C99) => accumulate error
		#endif
		}
		t2 = SQRT(t1+e);
		if ( m & with_cs_phase ) t2 = -t2;		/// optional \f$ (-1)^m \f$ Condon-Shortley phase.
		alm_im(shtns, im)[0] = t2;
	}

	#ifdef SHTNS_ISHIOKA
		double *clm, *xlm, *ylm;
		const long nlm0 = nlm_calc(LMAX+4, MMAX, MRES);
		const long n_alloc = (norm == sht_schmidt) ? 8 : 5;
		clm = (double *) malloc( n_alloc*nlm0/2 * sizeof(double) );	// a_lm, b_lm
		if (clm==0) shtns_runerr("not enough memory.");
		memset(clm, 0, n_alloc*nlm0/2 * sizeof(double) );	// fill with zeros, because some values will not be set otherwise
		xlm = clm + nlm0;
		shtns->clm = clm;
		shtns->xlm = xlm;
		ylm = xlm;
		if (n_alloc > 5)	ylm = xlm + 3*nlm0/2;
		shtns->x2lm = ylm;
	#endif
	// FOR ALT RECURRENCE:
		const long nlm1 = nlm_calc(LMAX+2, MMAX, MRES);
		double* const glm = (double *) malloc( (((norm==sht_schmidt) ? 3:2)* nlm1 + 2) * sizeof(double) );
		double* const alm2 = glm + nlm1;
		shtns->glm = glm;
		shtns->alm2 = alm2;
		shtns->glm_analys = glm + ((norm==sht_schmidt) ? 2*nlm1 : 0);

/// - Precompute the factors alm of the recurrence relation :
	#pragma omp parallel for private(t1, t2) schedule(dynamic)
	for (int im=0; im<=MMAX; ++im) {
		const long m = im*MRES;
		long lm = im*(2*lmax - (im-1)*MRES);

		if ((norm == sht_schmidt)||(norm == sht_for_rotations)) {		/// <b> For Schmidt semi-normalized </b>
			t2 = SQRT(2*m+1);
			alm[lm] /= t2;		/// starting value divided by \f$ \sqrt{2m+1} \f$ 
			alm[lm+1] = t2;		// l=m+1
			lm+=2;
			for (long l=m+2; l<=lmax; ++l) {
				t1 = SQRT((l+m)*(l-m));
				alm[lm+1] = (2*l-1)/t1;		/// \f[  a_l^m = \frac{2l-1}{\sqrt{(l+m)(l-m)}}  \f]
				alm[lm] = - t2/t1;			/// \f[  b_l^m = -\sqrt{\frac{(l-1+m)(l-1-m)}{(l+m)(l-m)}}  \f]
				t2 = t1;	lm+=2;
			}
		} else {			/// <b> For orthonormal or 4pi-normalized </b>
			t2 = 2*m+1;
			// starting value unchanged.
			alm[lm+1] = SQRT(2*m+3);		// l=m+1
			lm+=2;
			for (long l=m+2; l<=lmax; ++l) {
				t1 = (l+m)*(l-m);
				alm[lm+1] = SQRT(((2*l+1)*(2*l-1))/t1);			/// \f[  a_l^m = \sqrt{\frac{(2l+1)(2l-1)}{(l+m)(l-m)}}  \f]
				alm[lm] = - SQRT(((2*l+1)*t2)/((2*l-3)*t1));	/// \f[  b_l^m = -\sqrt{\frac{2l+1}{2l-3}\,\frac{(l-1+m)(l-1-m)}{(l+m)(l-m)}}  \f]
				t2 = t1;	lm+=2;
			}
		}

	/* ALT RECURRENCE */
	{	const long lm0 = (im>0) ? nlm_calc(LMAX+2, im-1, MRES) : 0;
		const long lm = im*(2*lmax - (im-1)*MRES);
		glm[lm0] = alm[lm];			glm[lm0+1] = alm[lm];
		for (int l=2; l<=lmax-m; l++) 		glm[lm0+l] = glm[lm0+l-2] * alm[lm+2*l-2];
		alm2[lm0] = 1.0;
		alm2[lm0+1] = alm[lm+1];
		for (int l=2; l<=lmax-m; l++)		alm2[lm0+l] = alm[lm+2*l-1] * glm[lm0+l-1]/glm[lm0+l];
		if (norm == sht_schmidt) {
			double* glm_a = shtns->glm_analys;
			for (int l=m; l<=lmax; l++) 	glm_a[lm0+l-m] = glm[lm0+l-m] * (2*l+1);		// normalize analysis for Schmidt
		}
	}
	/* END ALT RECURRENCE */

	#ifdef SHTNS_ISHIOKA
		/// PRE-COMPUTE VALUES FOR NEW RECURRENCE OF ISHIOKA (2018)
		/// see https://doi.org/10.2151/jmsj.2018-019
		const long lm0 = im*(2*lmax - (im-1)*MRES);
		real* elm = (real*) malloc( 3*(LMAX+4-m)/2 * sizeof(real) );
		real* dlm = elm + LMAX+4-m;
		for (long l=m+1; l<=LMAX+4; l++) {		// start at m+1, as for l=m, elm=0
			real num = (l-m)*(l+m);
			real den = (2*l+1)*(2*l-1);
			elm[l-(m+1)] = SQRT( num/den );
		}

		real alpha = 1.0L/elm[0];		// alpha_0
		for (long i=0; i<=(LMAX-m+1)/2; i++) {
			dlm[i] = alpha;
			alpha = (1.0L*(1-2*(i&1)))/(alpha*elm[2*i+2]*elm[2*i+1]);
		}

		//clm[lm0] = alm[lm0];
		real a0m = alm[lm0];		// initial value is the same, but should be cast into alpha_lm (multiply all alpha_lm by alm[lm0])
		long lm1 = (im+1)*(2*lmax - (im)*MRES);
		for (long i=0; i<(LMAX-m+1)/2; i++) {
			real a = dlm[i]*dlm[i] * (1-2*(i&1));
			clm[lm0/2+2*i]   = -a*(elm[2*i]*elm[2*i] + elm[2*i+1]*elm[2*i+1]);
			clm[lm0/2+2*i+1] = a;
			//if (lm0/2 + 2*i+1 >= lm1/2) printf("error at m=%d, i=%d (%ld >= %ld)\n",m,i, lm0/2 + 2*i+1, lm1/2);
		//	dlm[i] *= a0m;
		}
		
		if (norm == sht_schmidt)  a0m *= SQRT(2*m+1);
		
		const long lm3 = 3*im*(2*(LMAX+4) - (im-1)*MRES)/4;
		long lm2 = 3*(im+1)*(2*(LMAX+4) - (im)*MRES)/4;
		// change e(l,m), to be e(l,m) * alpha(l/2,m)
		for (long i=0; i<=(LMAX-m+1)/2; i++) {
			real a = dlm[i] * a0m;		// multiply by coefficient of ymm
			xlm[lm3 + 3*i]   = elm[2*i] * a;
			xlm[lm3 + 3*i+1] = elm[2*i+1] * a;
			xlm[lm3 + 3*i+2] = a;
			//if (lm3 + 3*i+2 >= lm2) printf("error at m=%d, i=%d (%ld >= %ld)\n",m,i, lm3 + 3*i+2, lm2);
		}
		
		if (norm == sht_schmidt) {		// correct for schmidt semi-normalized:
			const long lm3 = 3*im*(2*(LMAX+4) - (im-1)*MRES)/4;
			int l = m;
			real sq = SQRT(2*l+1);
			for (long i=0; i<=(LMAX-m+1)/2; i++) {
				real sq1 = SQRT(2*(l+1)+1);
				real sq2 = SQRT(2*(l+2)+1);
				ylm[lm3 + 3*i]   = xlm[lm3 + 3*i]   * sq;
				ylm[lm3 + 3*i+1] = xlm[lm3 + 3*i+1] * sq2;
				ylm[lm3 + 3*i+2] = xlm[lm3 + 3*i+2] * sq1;
								
				xlm[lm3 + 3*i]   /= sq;
				xlm[lm3 + 3*i+1] /= sq2;
				xlm[lm3 + 3*i+2] /= sq1;
				sq = sq2;
				l+=2;
			}
		}
		free(elm);
	#endif
	}
}

/// \internal returns the value of the Legendre Polynomial of degree l.
/// l is arbitrary, a direct recurrence relation is used, and a previous call to legendre_precomp() is not required.
static double legendre_Pl(const int l, double x)
{
	long int i;
	real p, p0, p1;

	if ((l==0)||(x==1.0)) return ( 1. );
	if (x==-1.0) return ( (l&1) ? -1. : 1. );

	p0 = 1.0;		/// \f$  P_0 = 1  \f$
	p1 = x;			/// \f$  P_1 = x  \f$
	for (i=2; i<=l; ++i) {		 /// recurrence : \f[  l P_l = (2l-1) x P_{l-1} - (l-1) P_{l-2}	 \f] (works ok up to l=100000)
		p = ((x*p1)*(2*i-1) - (i-1)*p0)/i;
		p0 = p1;
		p1 = p;
	}
	return ((double) p1);
}

/// \internal Good approximation of the k-th zero of the Bessel J0 function.
/// Function adapted from ducc https://gitlab.mpcdf.mpg.de/mtr/ducc/-/blob/ducc0/src/ducc0/math/gl_integrator.cc
/// Equation (4.1) from Bogaert 2014, doi:10.1137/140954969
/// Equation 10.21.19 from https://dlmf.nist.gov/10.21#E19 for \nu=0
double bessel_j0_root(int k)
{
	static const double j0_root[] = {	// first 11 zeros of the Bessel J0 function, generated with mpmath: besseljzero(0,1..11)
		2.4048255576957727686, 5.5200781102863106496, 8.653727912911012217,  11.791534439014281614,
		14.930917708487785948, 18.071063967910922543, 21.211636629879258959, 24.352471530749302737,
		27.493479132040254796, 30.634606468431975118, 33.775820213573568684 };
	static const double coeffs[] = {	// coefficients for the polynomial approximation, accurate to machine precision (1 ulp) for k>11:
		2092163573./82575360, -6277237./3440640, 3779./15360, -31./384, 1./8};

	if (k<=11) return j0_root[k-1];		// read from table for k<=11

	// use polynomial approximation for k>11, accurate to machine precision (1 ulp).
	double z = 0.25*M_PI*(4*k-1);
	double r = 1/z;
	double r2 = r*r;
	double p = coeffs[0];
	for (int i=1; i<5; i++) 	p = coeffs[i] + r2*p;
	return z + r*p;
}


/// \internal Generates the abscissa and weights for a Gauss-Legendre quadrature.
/// Newton method from initial Guess to find the zeros of the Legendre Polynome
/// \param x = abscissa, \param st = sin(theta)=sqrt(1-x*x), \param w = weights, \param n points.
/// \note Reference:  Numerical Recipes, Cornell press.
/// \note Reference for initial guesses: Hale & Townsend 2013, doi:10.1137/120889873
void gauss_nodes(double *x, double* st, double *w, const int n)
{
	real* al;
	real* al_e;
	#if defined( SHTNS_HIPREC ) || defined ( SHTNS_QUADMATH )
		const double eps = 1e-25;	//2.3e-16;		// desired precision, minimum = 2.2204e-16 (double)
	#else
		const double eps = 2.3e-16;		// desired precision, minimum = 2.2204e-16 (double)
	#endif

	al = (real*) malloc(sizeof(real)*4*n);
	al_e = al + 2*n;

	// first, we precompute the coefficients as they will be re-used quite a bit:
	#pragma omp parallel for
	for (int l=2; l<=n; l++) {
		#ifndef SHTNS_HIPREC
			real l_1 = ((real)1) / l;
			al[2*l-2] = (2*l-1) * l_1;
			al[2*l-1] = -(l-1) * l_1;
		#else
			al[2*l-2] = div_err(2*l-1,  l,  al_e+2*l-2);
			al[2*l-1] = div_err(-(l-1), l,  al_e+2*l-1);
		#endif
	}

	const long m = n/2;
	#pragma omp parallel for schedule(dynamic)
	for (long i=0;i<m;++i) {
		real z, pp, dz;
		int k=10;		// maximum Newton iteration count to prevent infinite loop.
		if (3*i > 2*m) {	// "interior" region, equation 3.3 from Hale & Townsend 2013
			z = cos((M_PI*(4*i+3))/(4*n+2));
			double n_1 = 0.25/n;
			z -= z*n_1*n_1*n_1*( 8*(n-1)  + n_1 * (26. - (56./3)/(1.-z*z)) );	// initial guess
		} else {	// "boundary region", equation 3.6 of Hale & Townsend 2013, doi:10.1137/120889873
			z = bessel_j0_root(i+1) / (n+0.5);
			z = cos( z + (z/tan(z)-1)/(8*z*(n+0.5)*(n+0.5)) );
		}
		real pp_, z_ = 0;		// corrections (increased precision)
		do {
			real p1 = z;	// P_1
			real p2 = 1.0;	// P_0
			real p1_ = z_;
			real p2_ = 0;
			for(long l=2;l<=n;++l) {		 // recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}	(works ok up to l=100000)
				real p3 = p2;
				p2 = p1;
				//p1 = ((2*l-1)*z*p2 - (l-1)*p3)/l;		// The Legendre polynomial...
				#ifndef SHTNS_HIPREC
					// quad precision float128 = 18x slower than double
					p1 = al[2*l-2]*z*p2 + al[2*l-1]*p3;		// The Legendre polynomial...
				#else
					// "rigorous" dd version with dd_mul and dd_add = 6.7x slower than double, 2.7x faster than float128
					// "fast&sloppy" dd version with dd_mul_no_norm and dd_add_sloppy = 3.6x slower than double, 5x faster than float128
					real p3_ = p2_;
					p2_ = p1_;
					dd_mul_no_norm(&p1,&p1_, z,z_,   al[2*l-2],al_e[2*l-2]);
					dd_mul_no_norm(&p3,&p3_, p3,p3_, al[2*l-1],al_e[2*l-1]);
					dd_mul_no_norm(&p1,&p1_, p1,p1_, p2,p2_);
					dd_add_sloppy(&p1,&p1_, p1,p1_, p3,p3_);
				#endif
			}
			#ifndef SHTNS_HIPREC
				pp = n*(p2-z*p1);			// ... and its (almost) derivative.
				dz = p1*(1.-z*z)/pp;		// increment from Newton's method
				z -= dz;
			#else
				dd_mul_no_norm(&pp,&pp_, z,z_, p1,p1_);		//pp=z*p1
				dd_add(&pp,&pp_, p2,p2_, -pp,-pp_);		//pp=p2-z*p1
				dd_mul_d(&pp,&pp_, pp,pp_, n);		// pp = n*(p2-z*p1)

				dd_mul_no_norm(&p2,&p2_, z,z_, z,z_);		// p2=z*z
				dd_add_d_ordered(&p2,&p2_, 1.0,-p2_, -p2);	// p2=1-z*z
				dd_mul(&p2,&p2_, p2,p2_, p1,p1_);	// p2=p1*(1-z*z)
				dd_div_no_norm(&p2,&p2_, p2,p2_, pp,pp_);	// p2=p1*(1-z*z)/pp
				dz = p2+p2_;	// double-word not needed.
				dd_add_d_ordered(&z, &z_, z,z_, -dz);
			#endif
			--k;
		} while (( fabs(dz) > ((double)z)*eps ) && (k > 0));
		if (k==0 && verbose>1) printf("convergence warning: i=%ld, iter=%d, z=%g, dz=%g, err=%g\n",i,10-k, (double) z, dz, fabs(dz)/((double)(z)) );
		#ifdef SHTNS_HIPREC
			real s2, s2_;
			dd_mul_no_norm(&s2,&s2_, z,z_, z,z_);
			dd_add_d_ordered(&s2,&s2_, 1.0,-s2_, -s2);	// 1-z*z
			dd_mul(&pp,&pp_, pp,pp_, pp,pp_);	// pp*pp
			dd_div(&pp,&pp_, s2,s2_, pp,pp_);	// s2/(pp*pp)
			dd_sqrt(&s2,&s2_, s2,s2_);
		#else
			real s2 = 1.-z*z;
			pp = s2/(pp*pp);		// quadrature weight
			s2 = SQRT(s2);
		#endif
		x[i] = z;		// Build up the abscissas.
		x[n-1-i] = -z;
		w[i] = 2*pp;		// Build up the weights.
		w[n-1-i] = 2*pp;
		st[i] = s2;
		st[n-1-i] = s2;
		//if (eps < 1e-16 && verbose>0) printf("i=%ld, sin(theta)=%g, sqrt(1-z2)=%g, err=%g\n", i, st[i], sqrt(1.-x[i]*x[i]), (st[i] - sqrt(1.-x[i]*x[i]))/st[i] );
	}
	if (n&1) {
		x[n/2]  = 0.0;		// exactly zero.
		st[n/2] = 1.0;
		real p2 = 1.0;	// P_0
		// recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}	(works ok up to l=100000)
		#ifndef SHTNS_HIPREC
			for(long l=2;l<=n;l+=2)  p2 *= (1.0-l)/l;		// The Legendre polynomial...
			p2 *= n;
			p2 = 2.0/(p2*p2);
		#else
			real p2_ = 0;
			for(long l=2;l<=n;l+=2)  dd_mul(&p2, &p2_, p2,p2_, al[2*l-1],al_e[2*l-1]);
			dd_mul_d(&p2, &p2_, p2,p2_, n);
			dd_mul(&p2, &p2_, p2,p2_, p2,p2_);
			dd_div(&p2, &p2_, 2,0, p2,p2_);
		#endif
		w[n/2] = p2;
	}

	free(al);
// as we started with initial guesses, we should check if the gauss points are actually unique and ordered.
	for (long i=m-1; i>0; i--) {
		if (((double) x[i]) >= ((double) x[i-1])) shtns_runerr("bad gauss points");
	}
}

/// Accurate evalutation of Nth-root of unity
/// Expect 0 <= k <= n.  (use k=k%n to enforce)
/// Should work well up to n = 2^48.
cplx exp_2IpiK_N_accurate(long k, long n)
{
	// range reduction to 0..2*pi
	//if ((unsigned) k > n)	k = k % n;		// k modulo n.
	// range reduction from 0..2*pi to 0..pi/4
	int quadrant = 0;	// record the quadrant of the result
	// from 0..2pi to 0..pi
	if (2*k > n) {
		quadrant |= 1;	// change sign of sin
		k = n-k;
	}
	// from 0..pi to 0..pi/2
	if (4*k > n) {
		quadrant |= 2;	// change sign of cos
		k = n - 2*k;
		n *= 2;
	}
	// from 0..pi/2 to 0..pi/4
	if (8*k > n) {		// exchange sin and cos
		quadrant |= 4;
		k = n - 4*k;
		n *= 4;
	}
	double c = 1.0;
	double s = 0.0;
	if (k != 0) {
		if (8*k == n) {		// special value for pi/4
			c = s = sqrt(0.5);
		} else if (12*k == n) {		// special value for pi/6
			c = sqrt(3)*0.5;
			s = 0.5;
		} else {
			// use extra precision here, to get accurate double values in the end.
			long double x = ((long double)(2*k)/n) * M_PIl;		// x should be:  0 <= x <= pi/4
			c = cosl(x);
			s = sinl(x);
		}
	}
	if (quadrant & 4) {
		double t = c;	c = s;	s = t;		// exchange sin and cos
	}
	if (quadrant & 2) c = -c;
	if (quadrant & 1) s = -s;
	return c + I*s;
}


/// \internal Generates the abscissa and weights for a Féjer quadrature (#1).
/// Compute weights via FFT
/// \param x = abscissa, \param w = weights, \param n points.
/// \note Reference: Waldvogel (2006) "Fast Construction of the Fejér and Clenshaw-Curtis Quadrature Rules"
/// requires n > 2*lmax
static void fejer1_nodes(double *x, double *st, double *w, const int n)
{
	double* wf = (double*) VMALLOC( (2*n+4) * sizeof(double) );
	cplx* v1 = (cplx*) (wf + n + (n&1));

	// the nodes
	for (int i=0; i<(n+1)/2; i++) {
		cplx cs = exp_2IpiK_N_accurate(2*i+1, 4*n);
		if (fabs(creal(cs) - cos(M_PI*(0.5+i)/n)) > 1e-15) printf("BAD POINTS\n");
		x[i]      =  creal(cs);		// cos(M_PI*(0.5+i)/n);
		x[n-1-i]  = -creal(cs);
		st[i]     =  cimag(cs);
		st[n-1-i] =  cimag(cs);
	}

	// the weights
	fftw_plan ifft = fftw_plan_dft_c2r_1d(n, v1, wf, FFTW_ESTIMATE);
	for (int k=0; k<n/2+1; k++) {
		cplx cs = exp_2IpiK_N_accurate(k, 2*n);
		double t = (M_PI*k)/n;	// 2*M_PI*k/(2*n)
		if (cabs(cos(t) + I*sin(t) - cs) > 1e-15) printf("BAD WEIGHTS\n");
		//v1[k] = (cos(t) + I*sin(t)) * 2.0/(1.0 - 4.0*k*k);
		v1[k] = cs * 2.0/(1.0 - 4.0*k*k);
	}
	if ((n&1) == 0) v1[n/2] = 0;
	fftw_execute_dft_c2r(ifft,v1,wf);

	for (int k=0; k<n; k++)
		w[k] = wf[k]/n;

	fftw_destroy_plan(ifft);
	free(wf);
}

/// exclude_poles must be 0 or 1 !
/// exclude_poles=0 : grid includes poles (theta=0 and pi) and uses Clenshaw-Curtis quadrature.
/// exclude_poles=1 : grid excludes poles and uses second Fejer quadrature.
static void fejer_cc_nodes(double *x, double* st, double *w, const int n, const int exclude_poles)
{
	const int N = n-1 + 2*exclude_poles;		// n-1 or n+1 : "logical" size of the FFT involved

	// the nodes
	for (int i=0; i<n; i++) {
		cplx cs = exp_2IpiK_N_accurate(i+exclude_poles, 2*N);
		if (fabs(creal(cs) - cos((M_PI*(i+exclude_poles))/N)) > 1e-15) printf("BAD POINTS\n");
		x[i]  = creal(cs);
		st[i] = cimag(cs);
	}

	double* wf = (double*) VMALLOC( (2*N+10) * sizeof(double) );
	cplx* v1 = (cplx*) (wf + N + (N&1));
	fftw_plan ifft = fftw_plan_dft_c2r_1d(N, v1, wf, FFTW_ESTIMATE);

	// the weights
	for (long k=0; k <= (N+1)/2; k++) {
		v1[k] = 2.0/(1 - 4*k*k);
	}
	if (exclude_poles) v1[N/2] = (N-3.0)/(2*(N/2)-1) - 1.0;		// this line switches to Fejer2 quadrature (excluding poles)

	fftw_execute_dft_c2r(ifft,v1,wf);
	
	wf[0] *= 0.5;
	for (int k=0; k<n-1+exclude_poles; k++)  w[k] = wf[k+exclude_poles]/N;
	if (exclude_poles==0)  w[n-1] = w[0];

	fftw_destroy_plan(ifft);
	free(wf);
}

/// Clenshaw-Curtis quadrature requires n > 2*lmax, and includes the poles
static void clenshaw_curtis_nodes(double *x, double* st, double *w, const int n) {
	fejer_cc_nodes(x, st, w, n, 0);
}

/// Féjer #2 quadrature requires n > 2*lmax, and excludes the poles.
static void fejer2_nodes(double *x, double* st, double *w, const int n) {
	fejer_cc_nodes(x, st, w, n, 1);
}

