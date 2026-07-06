c
c     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
c     *                                                               *
c     *                  copyright (c) 1998 by UCAR                   *
c     *                                                               *
c     *       University Corporation for Atmospheric Research         *
c     *                                                               *
c     *                      all rights reserved                      *
c     *                                                               *
c     *                      SPHEREPACK version 3.2                   *
c     *                                                               *
c     *       A Package of Fortran77 Subroutines and Programs         *
c     *                                                               *
c     *              for Modeling Geophysical Processes               *
c     *                                                               *
c     *                             by                                *
c     *                                                               *
c     *                  John Adams and Paul Swarztrauber             *
c     *                                                               *
c     *                             of                                *
c     *                                                               *
c     *         the National Center for Atmospheric Research          *
c     *                                                               *
c     *                Boulder, Colorado  (80307)  U.S.A.             *
c     *                                                               *
c     *                   which is sponsored by                       *
c     *                                                               *
c     *              the National Science Foundation                  *
c     *                                                               *
c     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
c
c
c ... file sshifte.f contains code and documentation for subroutine sshifte
c     and its' initialization subroutine sshifti
c
c ... required spherepack files 
c
c     hrfft.f
c
c     subroutine sshifte(ioff,nlon,nlat,goff,greg,wsav,lsav,work,lwork,ier)
c
c *** purpose
c
c     subroutine sshifte does a highly accurate 1/2 grid increment shift
c     in both longitude and latitude of equally spaced data on the sphere.
c     data is transferred between the nlon by nlat "offset grid" in goff
c     (which excludes poles) and the nlon by nlat+1 "regular grid" in greg
c     (which includes poles).  the transfer can go from goff to greg or from
c     greg to goff (see ioff).  the grids which underly goff and greg are
c     described below.  the north and south poles are at latitude 0.5*pi and
c     -0.5*pi radians respectively where pi = 4.*atan(1.).
c
c *** grid descriptions
c
c     let dlon = (pi+pi)/nlon and dlat = pi/nlat be the uniform grid
c     increments in longitude and latitude
c
c     offset grid
c
c     the "1/2 increment offset" grid (long(j),lat(i)) on which goff(j,i)
c     is given (ioff=0) or generated (ioff=1) is
c
c          long(j) =0.5*dlon + (j-1)*dlon  (j=1,...,nlon)
c
c     and
c
c          lat(i) = -0.5*pi + 0.5*dlat + (i-1)*dlat (i=1,...,nlat)
c
c     the data in goff is "shifted" one half a grid increment in longitude
c     and latitude and excludes the poles.  each goff(j,1) is given at
c     latitude -0.5*pi+0.5*dlat and goff(j,nlat) is given at 0.5*pi-0.5*dlat
c     (1/2 a grid increment away from the poles).  each goff(1,i),goff(nlon,i)
c     is given at longitude 0.5*dlon and 2.*pi-0.5*dlon.
c
c     regular grid
c
c     let dlat,dlon be as above.  then the nlon by nlat+1 grid on which
c     greg(j,i) is generated (ioff=0) or given (ioff=1) is given by
c
c          lone(j) = (j-1)*dlon (j=1,...,nlon)
c
c      and
c
c          late(i) = -0.5*pi + (i-1)*dlat (i=1,...,nlat+1)
c
c     values in greg include the poles and start at zero degrees longitude.
c
c *** remark
c
c     subroutine sshifte can be used in conjunction with subroutine trssph
c     when transferring data from an equally spaced "1/2 increment offset"
c     grid to a gaussian or equally spaced grid (which includes poles) of
c     any resolution.  this problem (personal communication with dennis
c     shea) is encountered in geophysical modeling and data analysis.
c
c *** method
c
c     fast fourier transform software from spherepack2 and trigonometric
c     identities are used to accurately "shift" periodic vectors half a
c     grid increment in latitude and longitude.  latitudinal shifts are
c     accomplished by setting periodic 2*nlat vectors over the pole for each
c     longitude.  when nlon is odd, this requires an additional longitude
c     shift.  longitudinal shifts are then executed for each shifted latitude.
c     when necessary (ioff=0) poles are obtained by averaging the nlon
c     shifted polar values.
c
c *** required spherepack files
c
c     hrfft.f
c
c *** argument description
c
c ... ioff
c
c     ioff = 0 if values on the offset grid in goff are given and values
c              on the regular grid in greg are to be generated.
c
c     ioff = 1 if values on the regular grid in greg are given and values
c              on the offset grid in goff are to be generated.
c
c ... nlon
c
c     the number of longitude points on both the "offset" and "regular"
c     uniform grid in longitude (see "grid description" above).  nlon
c     is also the first dimension of array goff and greg.  nlon determines
c     the grid increment in longitude as dlon = 2.*pi/nlon.  for example,
c     nlon = 144 for a 2.5 degree grid.  nlon can be even or odd and must
c     be greater than or equal to 4.  the efficiency of the computation
c     is improved when nlon is a product of small primes.
c
c ... nlat
c
c     the number of latitude points on the "offset" uniform grid.  nlat+1
c     is the number of latitude points on the "regular" uniform grid (see
c     "grid description" above).  nlat is the second dimension of array goff.
c     nlat+1 must be the second dimension of the array greg in the program
c     calling sshifte.  nlat determines the grid in latitude as pi/nlat.
c     for example, nlat = 36 for a five degree grid.  nlat must be at least 3.
c
c ... goff
c
c     a nlon by nlat array that contains data on the offset grid
c     described above.  goff is a given input argument if ioff=0.
c     goff is a generated output argument if ioff=1.
c
c ... greg
c
c     a nlon by nlat+1 array that contains data on the regular grid
c     described above.  greg is a given input argument if ioff=1.
c     greg is a generated output argument if ioff=0.
c
c ... wsav
c
c     a real saved work space array that must be initialized by calling
c     subroutine sshifti(ioff,nlon,nlat,wsav,ier) before calling sshifte.
c     wsav can then be used repeatedly by sshifte as long as ioff, nlon,
c     and nlat do not change.  this bypasses redundant computations and
c     saves time.  undetectable errors will result if sshifte is called
c     without initializing wsav whenever ioff, nlon, or nlat change.
c
c ... lsav
c
c     the length of the saved work space wsav in the routine calling sshifte
c     and sshifti.  lsave must be greater than or equal to 2*(2*nlat+nlon+16).
c
c ... work
c
c     a real unsaved work space
c
c ... lwork
c
c     the length of the unsaved work space in the routine calling sshifte
c     lwork must be greater than or equal to 2*nlon*(nlat+1) if nlon is even.
c     lwork must be greater than or equal to nlon*(5*nlat+1) if nlon is odd.
c
c ... ier
c
c     indicates errors in input parameters
c
c     = 0 if no errors are detected
c
c     = 1 if ioff is not equal to 0 or 1
c
c     = 1 if nlon < 4
c
c     = 2 if nlat < 3
c
c     = 3 if lsave < 2*(nlon+2*nlat+16)
c
c     = 4 if lwork < 2*nlon*(nlat+1) for nlon even or
c            lwork < nlon*(5*nlat+1) for nlon odd
c
c *** end of sshifte documentation
c
c     subroutine sshifti(ioff,nlon,nlat,lsav,wsav,ier)
c
c     subroutine sshifti initializes the saved work space wsav
c     for ioff and nlon and nlat (see documentation for sshifte).
c     sshifti must be called before sshifte whenever ioff or nlon
c     or nlat change.
c
c ... ier
c
c     = 0 if no errors with input arguments
c
c     = 1 if ioff is not 0 or 1
c
c     = 2 if nlon < 4
c
c     = 3 if nlat < 3
c
c     = 4 if lsav < 2*(2*nlat+nlon+16)
c
c *** end of sshifti documentation
c
      subroutine sshifte(ioff,nlon,nlat,goff,greg,wsav,lsav,
     +                   wrk,lwrk,ier)
      implicit none
      integer ioff,nlon,nlat,n2,nr,nlat2,nlatp1,lsav,lwrk,i1,i2,ier
      integer sshifte_trace_ctx
      real goff(nlon,nlat),greg(nlon,*),wsav(lsav),wrk(lwrk)
      common /sshifte_trace_ctx_blk/ sshifte_trace_ctx
c
c     check input parameters
c
      ier = 1
      if (ioff*(ioff-1).ne.0) return
      ier = 2
      if (nlon.lt.4) return
      ier = 3
      if (nlat .lt. 3) return
      ier = 4
      if (lsav .lt. 2*(2*nlat+nlon+16)) return
      ier = 5
      n2 = (nlon+1)/2
      if (2*n2 .eq. nlon) then
	if (lwrk .lt. 2*nlon*(nlat+1)) return
	i1 = 1
	nr = n2
      else
	if (lwrk .lt. nlon*(5*nlat+1)) return
	i1 = 1+2*nlat*nlon
	nr = nlon
      end if
      ier = 0
      sshifte_trace_ctx = 0
      nlat2 = nlat+nlat
      i2 = i1 + (nlat+1)*nlon
      if (ioff.eq.0) then
	call sshifte_trace_matrix('sshifte.goff_in', goff, nlon, nlat)
	call shftoff(nlon,nlat,goff,greg,wsav,nr,nlat2,
     +               wrk,wrk(i1),wrk(i2))
	call sshifte_trace_matrix('sshifte.greg_out', greg, nlon, nlat+1)
      else
	nlatp1 = nlat+1
	call sshifte_trace_matrix('sshifte.greg_in', greg, nlon, nlatp1)
	call shftreg(nlon,nlat,goff,greg,wsav,nr,nlat2,nlatp1,
     +               wrk,wrk(i1),wrk(i2))
	call sshifte_trace_matrix('sshifte.goff_out', goff, nlon, nlat)
      end if
      end

      subroutine shftoff(nlon,nlat,goff,greg,wsav,nr,nlat2,
     +                   rlat,rlon,wrk)
c
c     shift offset grid to regular grid, i.e.,
c     goff is given, greg is to be generated
c
      implicit none
      integer nlon,nlat,nlat2,n2,nr,j,i,js,isav
      real goff(nlon,nlat),greg(nlon,nlat+1)
      real rlat(nr,nlat2),rlon(nlat,nlon)
      real wsav(*),wrk(*)
      real gnorth,gsouth
      isav = 4*nlat+17
      n2 = (nlon+1)/2
c
c     execute full circle latitude shifts for nlon odd or even
c
      if (2*n2 .gt. nlon) then
c
c     odd number of longitudes
c
	do i=1,nlat
	  do j=1,nlon
	    rlon(i,j) = goff(j,i)
	  end do
	end do
	call sshifte_trace_matrix('shftoff.goff_as_rlon', rlon, nlat, nlon)
c
c       half shift in longitude
c
	call shifth(nlat,nlon,rlon,wsav(isav),wrk)
	call sshifte_trace_matrix('shftoff.rlon_shifted', rlon, nlat, nlon)
c
c       set full 2*nlat circles in rlat using shifted values in rlon
c
	do j=1,n2-1
	  js = j+n2
	  do i=1,nlat
	    rlat(j,i)      = goff(j,i)
	    rlat(j,nlat+i) = rlon(nlat+1-i,js)
	  end do
	end do
	do j=n2,nlon
	  js = j-n2+1
	  do i=1,nlat
	    rlat(j,i)      = goff(j,i)
	    rlat(j,nlat+i) = rlon(nlat+1-i,js)
	  end do
	end do
	call sshifte_trace_matrix('shftoff.rlat_before_lat_shift', rlat,
     +                          nlon, nlat2)
c
c       shift the nlon rlat vectors one half latitude grid
c
	call shifth(nlon,nlat2,rlat,wsav,wrk)
	call sshifte_trace_matrix('shftoff.rlat_after_lat_shift', rlat,
     +                          nlon, nlat2)
c
c       set nonpole values in greg and average for poles
c
	gnorth = 0.0
	gsouth = 0.0
	do j=1,nlon
	  gnorth = gnorth + rlat(j,1)
	  gsouth = gsouth + rlat(j,nlat+1)
	  do i=2,nlat
	    greg(j,i) =  rlat(j,i)
	  end do
	end do
	gnorth = gnorth/nlon
	gsouth = gsouth/nlon
	call sshifte_trace_matrix('shftoff.greg_after_poles', greg, nlon,
     +                          nlat+1)

      else
c
c     even number of longitudes (no initial longitude shift necessary)
c     set full 2*nlat circles (over poles) for each longitude pair (j,js)
c
	do j=1,n2
	  js = n2+j
	  do i=1,nlat
	    rlat(j,i)      = goff(j,i)
	    rlat(j,nlat+i) = goff(js,nlat+1-i)
	  end do
	end do
	call sshifte_trace_matrix('shftoff.even.rlat_before_lat_shift',
     +                          rlat, n2, nlat2)
c
c       shift the n2=(nlon+1)/2 rlat vectors one half latitude grid
c
	call shifth(n2,nlat2,rlat,wsav,wrk)
	call sshifte_trace_matrix('shftoff.even.rlat_after_lat_shift',
     +                          rlat, n2, nlat2)
c
c       set nonpole values in greg and average poles
c
	gnorth = 0.0
	gsouth = 0.0
	do j=1,n2
	  js = n2+j
	  gnorth = gnorth + rlat(j,1)
	  gsouth = gsouth + rlat(j,nlat+1)
	  do i=2,nlat
	    greg(j,i) =  rlat(j,i)
	    greg(js,i) = rlat(j,nlat2-i+2)
	  end do
	end do
	gnorth = gnorth/n2
	gsouth = gsouth/n2
	call sshifte_trace_matrix('shftoff.even.greg_after_poles', greg,
     +                          nlon, nlat+1)
      end if
c
c     set poles
c
      do j=1,nlon
	greg(j,1)      = gnorth
	greg(j,nlat+1) = gsouth
      end do
c
c     execute full circle longitude shift
c
	do j=1,nlon
	  do i=1,nlat
	    rlon(i,j) = greg(j,i)
	  end do
	end do
	call sshifte_trace_matrix('shftoff.final_rlon_before_lon_shift',
     +                          rlon, nlat, nlon)
	call shifth(nlat,nlon,rlon,wsav(isav),wrk)
	call sshifte_trace_matrix('shftoff.final_rlon_after_lon_shift',
     +                          rlon, nlat, nlon)
	do j=1,nlon
	  do i=2,nlat
	    greg(j,i) = rlon(i,j)
	  end do
	end do
	call sshifte_trace_matrix('shftoff.greg_final', greg, nlon, nlat+1)
      end

      subroutine shftreg(nlon,nlat,goff,greg,wsav,nr,nlat2,nlatp1,
     +                   rlat,rlon,wrk)
c
c     shift regular grid to offset grid, i.e.,
c     greg is given, goff is to be generated
c
      implicit none
      integer nlon,nlat,nlat2,nlatp1,n2,nr,j,i,js,isav
      integer sshifte_trace_ctx
      real goff(nlon,nlat),greg(nlon,nlatp1)
      real rlat(nr,nlat2),rlon(nlatp1,nlon)
      real wsav(*),wrk(*)
      common /sshifte_trace_ctx_blk/ sshifte_trace_ctx
      isav = 4*nlat+17
      n2 = (nlon+1)/2
c
c     execute full circle latitude shifts for nlon odd or even
c
      if (2*n2 .gt. nlon) then
c
c     odd number of longitudes
c
	do i=1,nlat+1
	  do j=1,nlon
	    rlon(i,j) = greg(j,i)
	  end do
	end do
	call sshifte_trace_matrix('shftreg.greg_as_rlon_full', rlon, nlat+1,
     +                          nlon)
c
c       half shift in longitude in rlon
c
	call shifth(nlat+1,nlon,rlon,wsav(isav),wrk)
	call sshifte_trace_matrix('shftreg.rlon_full_after_first_lon_shift',
     +                          rlon, nlat+1, nlon)
c
c       set full 2*nlat circles in rlat using shifted values in rlon
c
	do j=1,n2
	  js = j+n2-1
	  rlat(j,1) = greg(j,1)
	  do i=2,nlat
	    rlat(j,i) = greg(j,i)
	    rlat(j,nlat+i) = rlon(nlat+2-i,js)
	  end do
	  rlat(j,nlat+1) = greg(j,nlat+1)
	end do
	do j=n2+1,nlon
	  js = j-n2
	  rlat(j,1) = greg(j,1)
	  do i=2,nlat
	    rlat(j,i) = greg(j,i)
	    rlat(j,nlat+i) = rlon(nlat+2-i,js)
	  end do
	  rlat(j,nlat+1) = greg(j,nlat+1)
	end do
	call sshifte_trace_matrix('shftreg.odd.rlat_before_lat_shift', rlat,
     +                          nlon, nlat2)
c
c       shift the nlon rlat vectors one halflatitude grid
c
	call shifth(nlon,nlat2,rlat,wsav,wrk)
	call sshifte_trace_matrix('shftreg.odd.rlat_after_lat_shift', rlat,
     +                          nlon, nlat2)
c
c       set values in goff
c
	do j=1,nlon
	  do i=1,nlat
	    goff(j,i) =  rlat(j,i)
	  end do
	end do

      else
c
c     even number of longitudes (no initial longitude shift necessary)
c     set full 2*nlat circles (over poles) for each longitude pair (j,js)
c
	do j=1,n2
	  js = n2+j
	  rlat(j,1) = greg(j,1)
	  do i=2,nlat
	    rlat(j,i) = greg(j,i)
	    rlat(j,nlat+i) = greg(js,nlat+2-i)
	  end do
	  rlat(j,nlat+1) = greg(j,nlat+1)
	end do
	call sshifte_trace_matrix('shftreg.even.rlat_before_lat_shift', rlat,
     +                          n2, nlat2)
c
c       shift the n2=(nlon+1)/2 rlat vectors one half latitude grid
c
	sshifte_trace_ctx = 1
	call shifth(n2,nlat2,rlat,wsav,wrk)
	sshifte_trace_ctx = 0
	call sshifte_trace_matrix('shftreg.even.rlat_after_lat_shift', rlat,
     +                          n2, nlat2)
c
c       set values in goff
c
	do j=1,n2
	  js = n2+j
	  do i=1,nlat
	    goff(j,i) =  rlat(j,i)
	    goff(js,i) = rlat(j,nlat2+1-i)
	  end do
	end do
      end if
c
c     execute full circle longitude shift for all latitude circles
c
	do j=1,nlon
	  do i=1,nlat
	    rlon(i,j) = goff(j,i)
	  end do
	  rlon(nlat+1,j) = 0.0
	end do
	call sshifte_trace_matrix('rfull0', rlon, nlat+1, nlon)
	call shifth(nlat+1,nlon,rlon,wsav(isav),wrk)
	call sshifte_trace_matrix('shftreg.rlon_full_after_final_lon_shift',
     +                          rlon, nlat+1, nlon)
	do j=1,nlon
	  do i=1,nlat
	    goff(j,i) = rlon(i,j)
	  end do
	end do
	call sshifte_trace_matrix('shftreg.goff_final', goff, nlon, nlat)
      end

      subroutine sshifti(ioff,nlon,nlat,lsav,wsav,ier)
      integer ioff,nlat,nlon,nlat2,isav,ier
      real wsav(lsav)
      real pi,dlat,dlon,dp
      ier = 1
      if (ioff*(ioff-1).ne.0) return
      ier = 2
      if (nlon .lt. 4) return
      ier = 3
      if (nlat .lt. 3) return
      ier = 4
      if (lsav .lt. 2*(2*nlat+nlon+16)) return
      ier = 0
      pi = 4.0*atan(1.0)
c
c     set lat,long increments
c
      dlat = pi/nlat
      dlon = (pi+pi)/nlon
c
c     initialize wsav for left or right latitude shifts
c
      if (ioff.eq.0) then
	dp = -0.5*dlat
      else
	dp = 0.5*dlat
      end if
      nlat2 = nlat+nlat
      call shifthi(nlat2,dp,wsav)
c
c     initialize wsav for left or right longitude shifts
c
      if (ioff.eq.0) then
	dp = -0.5*dlon
      else
	dp = 0.5*dlon
      end if
      isav = 4*nlat + 17
      call shifthi(nlon,dp,wsav(isav))
      return
      end

      subroutine shifth(m,n,r,wsav,work)
      implicit none
      integer m,n,n2,k,l,sshifte_trace_ctx
      real r(m,n),wsav(*),work(*),r2km2,r2km1
      common /sshifte_trace_ctx_blk/ sshifte_trace_ctx
      n2 = (n+1)/2
c
c     compute fourier coefficients for r on shifted grid
c
      call hrfftf(m,n,r,m,wsav(n+2),work)
      if (sshifte_trace_ctx .eq. 1)
     +   call sshifte_trace_modes('shftreg.even.hrfftf',r,m,n)
      do l=1,m
	do k=2,n2
	  r2km2 = r(l,k+k-2)
	  r2km1 = r(l,k+k-1)
	  r(l,k+k-2)   =  r2km2*wsav(n2+k) - r2km1*wsav(k)
	  r(l,k+k-1)   =  r2km2*wsav(k)    + r2km1*wsav(n2+k)
	end do
      end do
      if (sshifte_trace_ctx .eq. 1)
     +   call sshifte_trace_modes('shftreg.even.rotate',r,m,n)
c
c     shift r with fourier synthesis and normalization
c
      call hrfftb(m,n,r,m,wsav(n+2),work)
      if (sshifte_trace_ctx .eq. 1)
     +   call sshifte_trace_modes('shftreg.even.hrfftb',r,m,n)
      do l=1,m
	do k=1,n
	  r(l,k) = r(l,k)/n
	end do
      end do
      return
      end

      subroutine shifthi(n,dp,wsav)
c
c     initialize wsav for subroutine shifth
c
      implicit none
      integer n,n2,k
      real wsav(*),dp
      n2 = (n+1)/2
      do k=2,n2
	wsav(k)    =  sin((k-1)*dp)
	wsav(k+n2) =  cos((k-1)*dp)
      end do
      call hrffti(n,wsav(n+2))
      return
      end

      subroutine sshifte_trace_modes(tag,r,m,n)
      implicit none
      character*(*) tag
      integer m,n,unit
      real r(m,n)
      logical sshifte_trace_enabled

      if (.not. sshifte_trace_enabled()) return
      unit = 99
      open(unit=unit,file='sshifte.log',status='unknown',
     +     position='append')
      write(unit,*) '[sshifte trace] ', tag, ' row1=',
     +              r(1,1), r(1,2), r(1,3), r(1,4), r(1,5)
      if (m .ge. 2) then
         write(unit,*) '[sshifte trace] ', tag, ' row2=',
     +                 r(2,1), r(2,2), r(2,3), r(2,4), r(2,5)
      end if
      return
      end

      logical function sshifte_trace_enabled()
      implicit none
      character*80 env
      integer lenv
      sshifte_trace_enabled = .false.
      env = ' '
      call getenv('SSHIFTE_TRACE', env)
      lenv = 80
   10 if (lenv .gt. 0 .and. env(lenv:lenv) .eq. ' ') then
         lenv = lenv - 1
         goto 10
      end if
      if (lenv .gt. 0) then
         if (env(1:1) .ne. '0' .and. env(1:1) .ne. 'f' .and.
     +       env(1:1) .ne. 'F') then
            sshifte_trace_enabled = .true.
         end if
      end if
      return
      end

      subroutine sshifte_trace_matrix(tag,data,nrow,ncol)
      implicit none
      character*(*) tag
      integer nrow,ncol
      real data(nrow,ncol)
      integer unit, i, j, head_rows, tail_rows, head_cols, tail_cols
      logical sshifte_trace_enabled, opened
      save opened, unit
      data opened /.false./
      data unit /99/

      if (.not. sshifte_trace_enabled()) return
      if (.not. opened) then
         open(unit=unit,file='sshifte.log',status='unknown')
         opened = .true.
      end if

      head_rows = 2
      if (head_rows .gt. nrow) head_rows = nrow
      tail_rows = 2
      if (tail_rows .gt. nrow - head_rows) tail_rows = nrow - head_rows
      head_cols = 4
      if (head_cols .gt. ncol) head_cols = ncol
      tail_cols = 4
      if (tail_cols .gt. ncol - head_cols) tail_cols = ncol - head_cols

      write(unit,*) '[sshifte trace] ', tag, ' shape=(', nrow, ',',
     +              ncol, ')'

      do 20 i = 1, head_rows
         if (ncol .le. 8) then
            write(unit,*) '[sshifte trace] ', tag, ' row ', i, ':',
     +                   (data(i,j), j=1,ncol)
         else
            write(unit,*) '[sshifte trace] ', tag, ' row ', i,
     +                   ' head'
            write(unit,*) (data(i,j), j=1,head_cols)
            write(unit,*) '[sshifte trace] ', tag, ' row ', i,
     +                   ' tail'
            write(unit,*) (data(i,j), j=ncol-tail_cols+1,ncol)
         end if
   20 continue

      if (tail_rows .gt. 0) then
         if (nrow .gt. head_rows) then
            write(unit,*) '[sshifte trace] ', tag, ' ...'
         end if
         do 30 i = nrow - tail_rows + 1, nrow
            if (i .le. head_rows) goto 30
            if (ncol .le. 8) then
               write(unit,*) '[sshifte trace] ', tag, ' row ', i, ':',
     +                      (data(i,j), j=1,ncol)
            else
               write(unit,*) '[sshifte trace] ', tag, ' row ', i,
     +                      ' head'
               write(unit,*) (data(i,j), j=1,head_cols)
               write(unit,*) '[sshifte trace] ', tag, ' row ', i,
     +                      ' tail'
               write(unit,*) (data(i,j), j=ncol-tail_cols+1,ncol)
            end if
   30    continue
      end if

      return
      end
