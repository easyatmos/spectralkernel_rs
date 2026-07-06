      subroutine shallow_initial(nlat,nlon,u,v,p,f)
c
c     Callable initialization subset from spherepack shallow.f.
c
      integer nlat,nlon
      real u(nlat,nlon),v(nlat,nlon),p(nlat,nlon),f(nlat,nlon)
      real phlt(361),work(361)
      real lambda,lhat

      pi = 4.*atan(1.)
      hpi = pi/2.
      dtr = pi/180.
      aa = 6.37122e6
      omega = 7.292e-5
      fzero = omega+omega
      uzero = 40.
      alphad = 60.
      alpha = dtr*alphad

      nl = 91
      nlm1 = nl-1
      nlm2 = nl-2
      cfn = 1./nlm1
      dlath = pi/nlm1
      do 10 i=1,nlm2
      theta = i*dlath
      sth = sin(theta)
      cth = cos(theta)
      uhat = shallow_ui(uzero,hpi-theta)
      phlt(i) = cfn*cth*uhat*(uhat/sth+aa*fzero)
   10 continue

      call shallow_sine(nlm2,phlt,work)
      do 12 i=1,nlm2
      phlt(i) = -phlt(i)/i
   12 continue

      ca = cos(alpha)
      sa = sin(alpha)
      dtheta = pi/(nlat-1)
      dlam = (pi+pi)/nlon
      do 50 j=1,nlon
      lambda = (j-1)*dlam
      cl = cos(lambda)
      sl = sin(lambda)
      do 50 i=1,nlat
      theta = (i-1)*dtheta
      st = cos(theta)
      ct = sin(theta)
      sth = ca*st+sa*ct*cl
      cthclh = ca*ct*cl-sa*st
      cthslh = ct*sl
      lhat = shallow_atanxy(cthclh,cthslh)
      clh = cos(lhat)
      slh = sin(lhat)
      cth = clh*cthclh+slh*cthslh
      that = shallow_atanxy(sth,cth)
      uhat = shallow_ui(uzero,hpi-that)
      p(i,j) = shallow_cosine(that,nlm2,phlt)
      u(i,j) = uhat*(ca*sl*slh+cl*clh)
      v(i,j) = uhat*(ca*cl*slh*st-clh*sl*st+sa*slh*ct)
      f(i,j) = fzero*sth
   50 continue
      return
      end

      function shallow_ui(amp,thetad)
      pi=4.*atan(1.)
      thetab=-pi/6.
      thetae= pi/2.
      xe=3.e-1
      x =xe*(thetad-thetab)/(thetae-thetab)
      shallow_ui = 0.
      if(x.le.0. .or. x.ge.xe) return
      shallow_ui=amp*exp(-1./x-1./(xe-x)+4./xe)
      return
      end

      function shallow_atanxy(x,y)
      shallow_atanxy = 0.
      if(x.eq.0. .and. y.eq.0.) return
      shallow_atanxy = atan2(y,x)
      return
      end

      subroutine shallow_sine(n,x,w)
      dimension x(n),w(n)
      arg = 4.*atan(1.)/(n+1)
      do 10 j=1,n
      w(j) = 0.
      do 10 i=1,n
      w(j) = w(j)+x(i)*sin(i*j*arg)
   10 continue
      do 15 i=1,n
      x(i) = 2.*w(i)
   15 continue
      return
      end

      function shallow_cosine(theta,n,cf)
      dimension cf(n)
      shallow_cosine = 0.
      do 10 i=1,n
      shallow_cosine = shallow_cosine+cf(i)*cos(i*theta)
   10 continue
      return
      end
