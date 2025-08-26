! Fortran 2003 interface to the SHTns library (with contribution from G-E. Moulard).

!!sht_gpu.cu:void cu_SH_to_spat(shtns_cfg shtns, cplx* d_Qlm, double *d_Vr, int llim)
!!sht_gpu.cu:void cu_SHsphtor_to_spat(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
!!sht_gpu.cu:void cu_SHqst_to_spat(shtns_cfg shtns, cplx* d_Qlm, cplx* d_Slm, cplx* d_Tlm, double* d_Vr, double* d_Vt, double* d_Vp, int llim)
!!sht_gpu.cu:void cu_spat_to_SH(shtns_cfg shtns, double *d_Vr, cplx* d_Qlm, int llim)
!!sht_gpu.cu:void cu_spat_to_SHsphtor(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, int llim)
!!sht_gpu.cu:void cu_spat_to_SHqst(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, int llim)

  interface

    subroutine cu_spat_to_SH(shtns,Vr,Qlm,ltr) bind(C, name='cu_spat_to_SH')
      import
      type(C_PTR), value :: shtns
      real(C_DOUBLE), intent(inout) :: Vr(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Qlm(*)
      integer(C_INT), value :: ltr
    end subroutine cu_spat_to_SH
    
    subroutine cu_SH_to_spat(shtns,Qlm,Vr,ltr) bind(C, name='cu_SH_to_spat')
      import
      type(C_PTR), value :: shtns
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Qlm(*)
      real(C_DOUBLE), intent(out) :: Vr(*)
      integer(C_INT), value :: ltr
    end subroutine cu_SH_to_spat
    
    subroutine cu_SHsphtor_to_spat(shtns,Slm,Tlm,Vt,Vp,ltr) bind(C, name='cu_SHsphtor_to_spat')
      import
      type(C_PTR), value :: shtns
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Slm(*)
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Tlm(*)
      real(C_DOUBLE), intent(out) :: Vt(*)
      real(C_DOUBLE), intent(out) :: Vp(*)
      integer(C_INT), value :: ltr
    end subroutine cu_SHsphtor_to_spat

    subroutine cu_SHsph_to_spat(shtns,Slm,Vt,Vp,ltr) bind(C, name='cu_SHsph_to_spat')
      import
      type(C_PTR), value :: shtns
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Slm(*)
      real(C_DOUBLE), intent(out) :: Vt(*)
      real(C_DOUBLE), intent(out) :: Vp(*)
      integer(C_INT), value :: ltr
    end subroutine cu_SHsph_to_spat

    subroutine cu_SHtor_to_spat(shtns,Tlm,Vt,Vp,ltr) bind(C, name='cu_SHtor_to_spat')
      import
      type(C_PTR), value :: shtns
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Tlm(*)
      real(C_DOUBLE), intent(out) :: Vt(*)
      real(C_DOUBLE), intent(out) :: Vp(*)
      integer(C_INT), value :: ltr
    end subroutine cu_SHtor_to_spat

    subroutine cu_spat_to_SHsphtor(shtns,Vt,Vp,Slm,Tlm,ltr) bind(C, name='cu_spat_to_SHsphtor')
      import
      type(C_PTR), value :: shtns
      real(C_DOUBLE), intent(inout) :: Vt(*)
      real(C_DOUBLE), intent(inout) :: Vp(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Slm(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Tlm(*)
      integer(C_INT), value :: ltr
    end subroutine cu_spat_to_SHsphtor
    
    subroutine cu_spat_to_SHqst(shtns,Vr,Vt,Vp,Qlm,Slm,Tlm,ltr) bind(C, name='cu_spat_to_SHqst')
      import
      type(C_PTR), value :: shtns
      real(C_DOUBLE), intent(inout) :: Vr(*)
      real(C_DOUBLE), intent(inout) :: Vt(*)
      real(C_DOUBLE), intent(inout) :: Vp(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Qlm(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Slm(*)
      complex(C_DOUBLE_COMPLEX), intent(out) :: Tlm(*)
      integer(C_INT), value :: ltr
    end subroutine cu_spat_to_SHqst
    
    subroutine cu_SHqst_to_spat(shtns,Qlm,Slm,Tlm,Vr,Vt,Vp,ltr) bind(C, name='cu_SHqst_to_spat')
      import
      type(C_PTR), value :: shtns
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Qlm(*)
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Slm(*)
      complex(C_DOUBLE_COMPLEX), intent(inout) :: Tlm(*)
      real(C_DOUBLE), intent(out) :: Vr(*)
      real(C_DOUBLE), intent(out) :: Vt(*)
      real(C_DOUBLE), intent(out) :: Vp(*)
      integer(C_INT), value :: ltr
    end subroutine cu_SHqst_to_spat

    !!void cushtns_set_streams(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream);
    subroutine cushtns_set_streams(shtns,compute_stream,transfer_stream) bind(C, name='cushtns_set_streams')
      import
      type(C_PTR), value :: shtns
      type(C_PTR), value :: compute_stream
      type(C_PTR), value :: transfer_stream
    end subroutine cushtns_set_streams

  end interface
