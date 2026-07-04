"""
spharmt_rs entry
"""
import time

import numpy as np
import math, sys

from ._common import _debug_print

from spectralkernel_rs.spectralkernel_rs import (
    # Scalar Harmonic Analysis/Synthesis - Equally-Spaced (including poles)
    shaesi as shaesi_rs,
    shsesi as shsesi_rs,
    shaeci as shaeci_rs,
    shseci as shseci_rs,
    # Vector Harmonic Analysis/Synthesis - Equally-Spaced (including poles)
    vhaesi as vhaesi_rs,
    vhsesi as vhsesi_rs,
    vhaeci as vhaeci_rs,
    vhseci as vhseci_rs,
    # Scalar Harmonic Analysis/Synthesis - Gaussian (including poles)
    shagsi as shagsi_rs,
    shsgsi as shsgsi_rs,
    shagci as shagci_rs,
    shsgci as shsgci_rs,
    # Vector Harmonic Analysis/Synthesis - Gaussian (including poles)
    vhagsi as vhagsi_rs,
    vhsgsi as vhsgsi_rs,
    vhagci as vhagci_rs,
    vhsgci as vhsgci_rs,
    # Scalar Harmonic Analysis - Initialization routines
    shaes as shaes_rs,
    shaec as shaec_rs,
    shags as shags_rs,
    shagc as shagc_rs,
    # Grid conversion utilities
    twodtooned as twodtooned_rs,
    onedtotwod as onedtotwod_rs,
    # Scalar Harmonic Synthesis - Initialization routines
    shses as shses_rs,
    shsec as shsec_rs,
    shsec_nogil as shsec_nogil_rs,
    shsgs as shsgs_rs,
    shsgc as shsgc_rs,
    # Vector Harmonic Analysis - Initialization routines
    vhaes as vhaes_rs,
    vhaes_nogil as vhaes_nogil_rs,
    vhaes_latpar_nogil as vhaes_latpar_nogil_rs,
    vhaec as vhaec_rs,
    vhaec_nogil as vhaec_nogil_rs,
    vhags as vhags_rs,
    vhags_nogil as vhags_nogil_rs,
    vhags_latpar_nogil as vhags_latpar_nogil_rs,
    vhagc as vhagc_rs,
    # Vector grid conversion utilities
    twodtooned_vrtdiv as twodtooned_vrtdiv_rs,
    onedtotwod_vrtdiv as onedtotwod_vrtdiv_rs,
    # Vector Harmonic Synthesis - Initialization routines
    vhses as vhses_rs,
    vhsec as vhsec_rs,
    vhsgs as vhsgs_rs,
    vhsgc as vhsgc_rs,
    # Laplacian operators
    invlap as invlap_rs,
    invlap_nogil as invlap_nogil_rs,
    lap as lap_rs,
    # Utility functions
    multsmoothfact as multsmoothfact_rs,
    regular_stored_init_nogil as regular_stored_init_nogil_rs,
    gaussian_stored_init_nogil as gaussian_stored_init_nogil_rs,
)


# define a list of instance variables that cannot be rebound
# or unbound.
_private_vars = ['nlon','nlat','gridtype','legfunc','rsphere']


def _as_rs_float32_array(arr):
    """Convert input to a C-contiguous `float32` array for Rust bindings."""
    return np.ascontiguousarray(np.asarray(arr), dtype=np.float32)

class Spharmt_rs:
    """
    spherical harmonic transform class.

    @ivar nlat: number of latitudes (set when class instance is created,
    cannot be changed).

    @ivar nlon: number of longitudes (set when class instance is created,
    cannot be changed).

    @ivar rsphere: The radius of the sphere in meters (set when class
    instance is created, cannot be changed).

    @ivar legfunc: 'stored' or 'computed'.  If 'stored',
    associated legendre functions are precomputed and stored when the
    class instance is created.  If 'computed', associated
    legendre functions are computed on the fly when transforms are
    requested. Set when class instance is created, cannot be changed.

    @ivar gridtype: 'regular' (equally spaced in longitude and latitude)
    or 'gaussian' (equally spaced in longitude, latitudes located at
    roots of ordinary Legendre polynomial of degree nlat). Set when class
    instance is created, cannot be changed.
    """

    def __setattr__(self, key, val):
        """
        prevent modification of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to rebind read-only instance variable '+key)
        else:
            self.__dict__[key] = val

    def __delattr__(self, key):
        """
        prevent deletion of read-only instance variables.
        """
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to unbind read-only instance variable '+key)
        else:
            del self.__dict__[key]

    def __init__(self, nlon, nlat, rsphere=6.3712e6, gridtype='regular', legfunc='stored', transform_backend='standard'):
        """
        create a Spharmt class instance.

        @param nlon: Number of longitudes. The grid must be oriented from
        east to west, with the first point at the Greenwich meridian
        and the last point at 360-delta degrees east
        (where delta = 360/nlon degrees). Must be >= 4. Transforms will
        be faster when nlon is the product of small primes.

        @param nlat: Number of latitudes.  The grid must be oriented from north
        to south. If nlat is odd the equator is included.
        If nlat is even the equator will lie half way between points
        points nlat/2 and (nlat/2)+1. Must be >=3.

        @keyword rsphere: The radius of the sphere in meters.
        Default 6371200 (the value for Earth).

        @keyword legfunc: 'stored' (default) or 'computed'.  If 'stored',
        associated legendre functions are precomputed and stored when the
        class instance is created.  This uses O(nlat**3) memory, but
        speeds up the spectral transforms.  If 'computed', associated
        legendre functions are computed on the fly when transforms are
        requested.  This uses O(nlat**2) memory, but slows down the spectral
        transforms a bit.

        @keyword gridtype: 'regular' (default) or 'gaussian'. Regular grids
        will include the poles and equator if nlat is odd.  Gaussian
        grids never include the poles, but will include the equator if
        nlat is odd.
        """
        # sanity checks.
        if rsphere > 0.0:
            self.rsphere= rsphere
        else:
            msg = 'Spharmt.__init__ illegal value of rsphere (%s) - must be postitive' % (rsphere)
            raise ValueError(msg)
        if nlon > 3:
            self.nlon = nlon
        else:
            msg = 'Spharmt.__init__ illegal value of nlon (%s) - must be at least 4' % (nlon,)
            raise ValueError(msg)
        if nlat > 2:
            self.nlat = nlat
        else:
            msg = 'Spharmt.__init__ illegal value of nlat (%s) - must be at least 3' % (nlat,)
            raise ValueError(msg)
        if gridtype != 'regular' and gridtype != 'gaussian':
            msg = 'Spharmt.__init__ illegal value of gridtype (%s) - must be either "gaussian" or "regular"' % gridtype
            raise ValueError(msg)
        else:
            self.gridtype = gridtype

        if legfunc != 'computed' and legfunc != 'stored':
            msg = 'Spharmt.__init__ illegal value of legfunc (%s) - must be either "computed" or "stored"' % legfunc
            raise ValueError(msg)
        else:
            self.legfunc = legfunc
        self.transform_backend = transform_backend

        if nlon%2:                              # nlon is odd
            n1 = min(nlat, (nlon + 1)//2)
        else:
            n1 = min(nlat, (nlon + 2)//2)
        if nlat%2:                              # nlat is odd
            n2 = (nlat + 1)//2
        else:
            n2 = nlat//2

        if gridtype == 'regular':
            if legfunc == 'stored':
                lshaes = (n1*n2*(nlat + nlat - n1 + 1))//2 + nlon + 15
                lwork = 5*nlat*n2 + 3*((n1 - 2)*(nlat + nlat - n1 -1))//2
                lshaes = int(lshaes)
                lwork = int(lwork)
                lvhaes = n1*n2*(nlat + nlat - n1 + 1) + nlon + 15
                lwork_vector = 3*(max(n1 -2,0)*(nlat + nlat - n1 - 1))//2 + 5*n2*nlat
                lvhaes = int(lvhaes)
                lwork_vector = int(lwork_vector)
                lvhses = n1*n2*(nlat + nlat - n1 + 1) + nlon + 15
                lvhses = int(lvhses)
                if transform_backend in ('nogil', 'latpar'):
                    wshaes, ierror_shaes, wshses, ierror_shses, wvhaes, ierror_vhaes, wvhses, ierror_vhses = regular_stored_init_nogil_rs(
                        nlat, nlon, lshaes, lwork, lvhaes, lwork_vector, nlat+1, 2*(nlat+1)
                    )
                    if ierror_shaes != 0:
                        msg = 'In return from call to regular_stored_init_nogil/shaesi ierror =  %d' % ierror_shaes
                        raise ValueError(msg)
                    if ierror_shses != 0:
                        msg = 'In return from call to regular_stored_init_nogil/shsesi ierror =  %d' % ierror_shses
                        raise ValueError(msg)
                    if ierror_vhaes != 0:
                        msg = 'In return from call to regular_stored_init_nogil/vhaesi ierror =  %d' % ierror_vhaes
                        raise ValueError(msg)
                    if ierror_vhses != 0:
                        msg = 'In return from call to regular_stored_init_nogil/vhsesi ierror =  %d' % ierror_vhses
                        raise ValueError(msg)
                    self.wshaes = wshaes
                    self.wshses = wshses
                    self.wvhaes = wvhaes
                    self.wvhses = wvhses
                else:
                    wshaes, ierror = shaesi_rs(nlat, nlon, lshaes, lwork, nlat+1)
                    if ierror != 0:
                        msg = 'In return from call to shaesi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wshaes = wshaes
                    lshses = lshaes
                    wshses, ierror = shsesi_rs(nlat, nlon, lshses, lwork, nlat+1)
                    if ierror != 0:
                        msg = 'In return from call to shsesi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wshses = wshses
                    wvhaes, ierror = vhaesi_rs(nlat, nlon, lvhaes, lwork_vector, 2*(nlat+1))
                    if ierror != 0:
                        msg = 'In return from call to vhaesi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wvhaes = wvhaes
                    wvhses, ierror = vhsesi_rs(nlat,nlon,lvhses,lwork_vector,2*(nlat+1))
                    if ierror != 0:
                        msg = 'In return from call to vhsesi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wvhses = wvhses
            else:
                lshaec = 2*nlat*n2 + 3*((n1 - 2)*(nlat + nlat - n1 - 1))//2 + nlon + 15
                lshaec = int(lshaec)
                wshaec, ierror = shaeci_rs(nlat, nlon, lshaec, 2*(nlat+1))
                if ierror != 0:
                    msg = 'In return from call to shaeci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wshaec = wshaec
                lshsec = lshaec
                wshsec, ierror = shseci_rs(nlat, nlon, lshsec, 2*(nlat+1))
                if ierror != 0:
                    msg = 'In return from call to shseci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wshsec = wshsec
                lvhaec = 4*nlat*n2 + 3*max(n1 - 2, 0)*(2*nlat - n1 - 1) + nlon + 15
                lvhaec = int(lvhaec)
                wvhaec, ierror = vhaeci_rs(nlat, nlon, lvhaec,  2*(nlat+1))
                if ierror != 0:
                    msg = 'In return from call to vhaeci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wvhaec = wvhaec
                lvhsec = lvhaec
                wvhsec, ierror = vhseci_rs(nlat, nlon, lvhsec,  2*(nlat+1))
                if ierror != 0:
                    msg = 'In return from call to vhseci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wvhsec = wvhsec


        elif gridtype == 'gaussian':
            if legfunc == 'stored':
                lshags = nlat*(3*(n1 + n2) - 2) + (n1 - 1)*(n2*(2*nlat - n1) - 3*n1)//2 + nlon + 15
                lwork = 4*nlat*(nlat + 2) + 2
                ldwork = nlat*(nlat + 4)
                lshags = int(lshags)
                lshsgs = lshags
                lvhags = (nlat + 1)*(nlat + 1)*nlat//2 + nlon + 15
                ldwork = (3*nlat*(nlat + 3) + 2)//2
                lvhags = int(lvhags)
                ldwork = int(ldwork)
                lmn = nlat*(nlat + 1)//2
                lvhsgs = 2*n2*lmn + nlon + 15
                ldwork = (3*nlat*(nlat + 3) + 2)//2
                lvhsgs = int(lvhsgs)
                ldwork = int(ldwork)
                if transform_backend in ('nogil', 'latpar'):
                    wshags, ierror_shags, wshsgs, ierror_shsgs, wvhags, ierror_vhags, wvhsgs, ierror_vhsgs = gaussian_stored_init_nogil_rs(
                        nlat, nlon, lshags, lwork, nlat*(nlat + 4), lvhags, ldwork, lvhsgs, ldwork
                    )
                    if ierror_shags != 0:
                        msg = 'In return from call to gaussian_stored_init_nogil/shagsi ierror =  %d' % ierror_shags
                        raise ValueError(msg)
                    if ierror_shsgs != 0:
                        msg = 'In return from call to gaussian_stored_init_nogil/shsgsi ierror =  %d' % ierror_shsgs
                        raise ValueError(msg)
                    if ierror_vhags != 0:
                        msg = 'In return from call to gaussian_stored_init_nogil/vhagsi ierror =  %d' % ierror_vhags
                        raise ValueError(msg)
                    if ierror_vhsgs != 0:
                        msg = 'In return from call to gaussian_stored_init_nogil/vhsgsi ierror =  %d' % ierror_vhsgs
                        raise ValueError(msg)
                    self.wshags = wshags
                    self.wshsgs = wshsgs
                    self.wvhags = wvhags
                    self.wvhsgs = wvhsgs
                else:
                    wshags, ierror = shagsi_rs(nlat, nlon, lshags, lwork, nlat*(nlat + 4))
                    if ierror != 0:
                        msg = 'In return from call to shagsi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wshags = wshags
                    wshsgs, ierror = shsgsi_rs(nlat, nlon, lshsgs, lwork, nlat*(nlat + 4))
                    if ierror != 0:
                        msg = 'In return from call to shsgsi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wshsgs = wshsgs
                    wvhags, ierror = vhagsi_rs(nlat, nlon, lvhags, ldwork)
                    if ierror != 0:
                        msg = 'In return from call to vhagsi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wvhags = wvhags
                    wvhsgs, ierror = vhsgsi_rs(nlat, nlon, lvhsgs, ldwork)
                    if ierror != 0:
                        msg = 'In return from call to vhsgsi in Spharmt.__init__ ierror =  %d' % ierror
                        raise ValueError(msg)
                    self.wvhsgs = wvhsgs
            else:
                lshagc = nlat*(2*n2 + 3*n1 - 2) + 3*n1*(1 - n1)//2 + nlon + 15
                lshagc = int(lshagc)
                wshagc, ierror = shagci_rs(nlat, nlon, lshagc, nlat*(nlat+4))
                if ierror != 0:
                    msg = 'In return from call to shagci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wshagc = wshagc
                lshsgc = lshagc
                wshsgc, ierror = shsgci_rs(nlat, nlon, lshsgc, nlat*(nlat+4))
                if ierror != 0:
                    msg = 'In return from call to shsgci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wshsgc = wshsgc
                lvhagc = 4*nlat*n2 + 3*max(n1 - 2, 0)*(2*nlat - n1 - 1) + nlon + n2 + 15
                ldwork = 2*nlat*(nlat+1)+1
                lvhagc = int(lvhagc)
                wvhagc, ierror = vhagci_rs(nlat, nlon, lvhagc, ldwork)
                if ierror != 0:
                    msg = 'In return from call to vhagci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wvhagc = wvhagc
                lvhsgc = 4*nlat*n2 + 3*max(n1 - 2, 0)*(2*nlat - n1 - 1) + nlon + 15
                lvhsgc = int(lvhsgc)
                wvhsgc, ierror = vhsgci_rs(nlat, nlon, lvhsgc, ldwork)
                if ierror != 0:
                    msg = 'In return from call to vhsgci in Spharmt.__init__ ierror =  %d' % ierror
                    raise ValueError(msg)
                self.wvhsgc = wvhsgc

    def grdtospec(self, datagrid, ntrunc=None):
        """
        grid to spectral transform (spherical harmonic analysis).

        @param datagrid: rank 2 or 3 numpy float32 array with shape (nlat,nlon) or
        (nlat,nlon,nt), where nt is the number of grids to be transformed.  If
        datagrid is rank 2, nt is assumed to be 1.

        @keyword ntrunc:  optional spectral truncation limit.
        (default self.nlat-1)

        @return: C{B{dataspec}} - rank 1 or 2 numpy complex array with shape
        (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt) containing
        complex spherical harmonic coefficients resulting from the spherical
        harmonic analysis of datagrid.
        """

        # check that datagrid is rank 2 or 3 with size (self.nlat, self.nlon) or
        # (self.nlat, self.nlon, nt) where nt is number of grids to transform.

        idim = datagrid.ndim

        if len(datagrid.shape) > 3:
            msg = 'grdtospec needs a rank two or three array, got %d' % (len(datagrid.shape),)
            raise ValueError(msg)

        if datagrid.shape[0] != self.nlat or datagrid.shape[1] != self.nlon:
            msg = 'grdtospec needs an array of size %d by %d, got %d by %d' % (self.nlat, self.nlon, datagrid.shape[0], datagrid.shape[1],)
            raise ValueError(msg)

        # check ntrunc.

        if ntrunc is None:
            ntrunc = self.nlat-1

        if ntrunc < 0 or ntrunc+1 > datagrid.shape[0]:
            msg = 'ntrunc must be between 0 and %d' % (datagrid.shape[0]-1,)
            raise ValueError(msg)


        nlat = self.nlat
        nlon = self.nlon
        if nlat%2:                              # nlat is odd
            n2 = (nlat + 1)//2
        else:
            n2 = nlat//2

        if len(datagrid.shape) == 2:
            nt = 1
            datagrid = np.reshape(datagrid, (nlat,nlon,1))
        else:
            nt = datagrid.shape[2]

        datagrid = _as_rs_float32_array(datagrid)

        # regular grid.
        if self.gridtype == 'regular':

        # do grid to spectral transform.
            if self.legfunc == 'stored':
                lwork = (nt+1)*nlat*nlon
                a,b,ierror = shaes_rs(datagrid,_as_rs_float32_array(self.wshaes),lwork)
                if ierror != 0:
                    msg = 'In return from call to shaes in Spharmt.grdtospec ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(nt*nlon+max(3*n2,nlon))
                a,b,ierror = shaec_rs(datagrid,_as_rs_float32_array(self.wshaec),lwork)
                if ierror != 0:
                    msg = 'In return from call to shaec in Spharmt.grdtospec ierror =  %d' % ierror

                    raise ValueError(msg)

        # gaussian grid.
        elif self.gridtype == 'gaussian':

        # do grid to spectral transform.
            if self.legfunc == 'stored':
                lwork = nlat*nlon*(nt+1)
                a,b,ierror = shags_rs(datagrid,_as_rs_float32_array(self.wshags),lwork)
                if ierror != 0:
                    msg = 'In return from call to shags in Spharmt.grdtospec ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(nlon*nt+max(3*n2,nlon))
                a,b,ierror = shagc_rs(datagrid,_as_rs_float32_array(self.wshagc),lwork)
                if ierror != 0:
                    msg = 'In return from call to shagc in Spharmt.grdtospec ierror =  %d' % ierror
                    raise ValueError(msg)

        # convert 2d real and imag spectral arrays into 1d complex array.
        dataspec = twodtooned_rs(a,b,ntrunc)

        if idim == 2:
            return np.squeeze(dataspec)
        else:
            return dataspec

    def spectogrd(self, dataspec):

        """
        spectral to grid transform (spherical harmonic synthesis).

        @param dataspec: rank 1 or 2 numpy complex array with shape
        (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt) containing
        complex spherical harmonic coefficients (where ntrunc is the
        triangular truncation limit and nt is the number of spectral arrays
        to be transformed). If dataspec is rank 1, nt is assumed to be 1.

        @return: C{B{datagrid}} - rank 2 or 3 numpy float32 array with shape
        (nlat,nlon) or (nlat,nlon,nt) containing the gridded data resulting from
        the spherical harmonic synthesis of dataspec.
        """

        total_start = time.perf_counter()
        _debug_print(
            "[Spharmt_rs.spectogrd] start | "
            f"dataspec_shape={getattr(dataspec, 'shape', None)} | "
            f"gridtype={self.gridtype} | legfunc={self.legfunc}"
        , flush=True)

        # make sure dataspec is rank 1 or 2.

        idim = dataspec.ndim

        if len(dataspec.shape) > 2:
            msg = 'spectogrd needs a rank one or two array, got %d' % (len(dataspec.shape),)
            raise ValueError(msg)

        nlat = self.nlat
        nlon = self.nlon
        if nlat%2:                              # nlat is odd
            n2 = (nlat + 1)//2
        else:
            n2 = nlat//2

        if len(dataspec.shape) == 1:
            nt = 1
            dataspec = np.reshape(dataspec, (dataspec.shape[0],1))
        else:
            nt = dataspec.shape[1]

        ntrunc = int(-1.5 + 0.5*math.sqrt(9.-8.*(1.-dataspec.shape[0])))
        if ntrunc > nlat-1:
            msg = 'ntrunc too large - can be max of %d, got %d' % (nlat-1,ntrunc)
            raise ValueError(msg)

        step_start = time.perf_counter()
        _debug_print("[Spharmt_rs.spectogrd] entering onedtotwod_rs(...)", flush=True)
        a, b = onedtotwod_rs(dataspec,nlat)
        _debug_print(
            "[Spharmt_rs.spectogrd] finished onedtotwod_rs(...) | "
            f"a_shape={getattr(a, 'shape', None)} | "
            f"b_shape={getattr(b, 'shape', None)} | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)

        # regular grid.
        if self.gridtype == 'regular':

            # do spectral to grid transform.
            if self.legfunc == 'stored':
                lwork = (nt+1)*nlat*nlon
                step_start = time.perf_counter()
                _debug_print("[Spharmt_rs.spectogrd] entering shses_rs(...)", flush=True)
                datagrid, ierror = shses_rs(a,b,self.wshses,lwork)
                _debug_print(
                    "[Spharmt_rs.spectogrd] finished shses_rs(...) | "
                    f"elapsed={time.perf_counter() - step_start:.3f}s"
                , flush=True)
                if ierror != 0:
                    msg = 'In return from call to shses in Spharmt.spectogrd ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(nt*nlon+max(3*n2,nlon))
                step_start = time.perf_counter()
                if self.transform_backend in ('nogil', 'latpar'):
                    _debug_print("[Spharmt_rs.spectogrd] entering shsec_nogil_rs(...)", flush=True)
                    datagrid, ierror = shsec_nogil_rs(a,b,self.wshsec,lwork)
                    _debug_print(
                        "[Spharmt_rs.spectogrd] finished shsec_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                else:
                    _debug_print("[Spharmt_rs.spectogrd] entering shsec_rs(...)", flush=True)
                    datagrid, ierror = shsec_rs(a,b,self.wshsec,lwork)
                    _debug_print(
                        "[Spharmt_rs.spectogrd] finished shsec_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                if ierror != 0:
                    msg = 'In return from call to shsec in Spharmt.spectogrd ierror =  %d' % ierror
                    raise ValueError(msg)

        # gaussian grid.
        elif self.gridtype == 'gaussian':

            # do spectral to grid transform.
            if self.legfunc == 'stored':
                lwork = nlat*nlon*(nt+1)
                step_start = time.perf_counter()
                _debug_print("[Spharmt_rs.spectogrd] entering shsgs_rs(...)", flush=True)
                datagrid, ierror = shsgs_rs(a,b,self.wshsgs,lwork)
                _debug_print(
                    "[Spharmt_rs.spectogrd] finished shsgs_rs(...) | "
                    f"elapsed={time.perf_counter() - step_start:.3f}s"
                , flush=True)
                if ierror != 0:
                    msg = 'In return from call to shsgs in Spharmt.spectogrd ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(nlon*nt+max(3*n2,nlon))
                step_start = time.perf_counter()
                _debug_print("[Spharmt_rs.spectogrd] entering shsgc_rs(...)", flush=True)
                datagrid, ierror = shsgc_rs(a,b,self.wshsgc,lwork)
                _debug_print(
                    "[Spharmt_rs.spectogrd] finished shsgc_rs(...) | "
                    f"elapsed={time.perf_counter() - step_start:.3f}s"
                , flush=True)
                if ierror != 0:
                    msg = 'In return from call to shsgc in Spharmt.spectogrd ierror =  %d' % ierror
                    raise ValueError(msg)

        _debug_print(
            "[Spharmt_rs.spectogrd] finished | "
            f"datagrid_shape={getattr(datagrid, 'shape', None)} | "
            f"total_elapsed={time.perf_counter() - total_start:.3f}s"
        , flush=True)
        if idim == 1:
            return np.squeeze(datagrid)
        else:
            return datagrid

    def getvrtdivspec(self, ugrid, vgrid, ntrunc=None):
        """
        compute spectral coefficients of vorticity and divergence given vector wind.

        @param ugrid: rank 2 or 3 numpy float32 array containing grid of zonal
        winds.  Must have shape (nlat,nlon) or (nlat,nlon,nt), where nt is the number
        of grids to be transformed.  If ugrid is rank 2, nt is assumed to be 1.

        @param vgrid: rank 2 or 3 numpy float32 array containing grid of meridional
        winds.  Must have shape (nlat,nlon) or (nlat,nlon,nt), where nt is the number
        of grids to be transformed.  Both ugrid and vgrid must have the same shape.

        @keyword ntrunc:  optional spectral truncation limit.
        (default self.nlat-1)

        @return: C{B{vrtspec, divspec}} - rank 1 or 2 numpy complex arrays
        of vorticity and divergence spherical harmonic coefficients with shape
        shape (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt).
        """

        total_start = time.perf_counter()
        _debug_print(
            "[Spharmt_rs.getvrtdivspec] start | "
            f"ugrid_shape={getattr(ugrid, 'shape', None)} | "
            f"vgrid_shape={getattr(vgrid, 'shape', None)} | "
            f"ntrunc={ntrunc} | gridtype={self.gridtype} | "
            f"legfunc={self.legfunc} | transform_backend={self.transform_backend}"
        , flush=True)

        # make sure ugrid,vgrid are rank 2 or 3 and same shape.

        idim = ugrid.ndim

        shapeu = ugrid.shape
        shapev = vgrid.shape

        if ntrunc is None:
            ntrunc = self.nlat-1

        if shapeu != shapev:
            msg = 'getvrtdivspec input arrays must be same shape!'
            raise ValueError(msg)


        if len(shapeu) !=2 and len(shapeu) !=3:
            msg = 'getvrtdivspec needs rank two or three arrays!'
            raise ValueError(msg)

        if shapeu[0] != self.nlat or shapeu[1] != self.nlon:
            msg = 'getvrtdivspec needs input arrays whose first two dimensions are si%d and %d, got %d and %d' % (self.nlat, self.nlon, ugrid.shape[0], ugrid.shape[1],)
            raise ValueError(msg)


        # check ntrunc.

        if ntrunc < 0 or ntrunc+1 > shapeu[0]:
            msg = 'ntrunc must be between 0 and %d' % (ugrid.shape[0]-1,)
            raise ValueError(msg)

        nlat = self.nlat
        nlon = self.nlon
        if nlat%2:                              # nlat is odd
            n2 = (nlat + 1)//2
        else:
            n2 = nlat//2
        rsphere= self.rsphere

        # convert from geographical to math coordinates, add extra dimension
        # if necessary.

        if len(shapeu) == 2:
            nt = 1
            w = np.reshape(ugrid, (nlat,nlon,1))
            v = -np.reshape(vgrid, (nlat,nlon,1))
        else:
            nt = shapeu[2]
            w = ugrid
            v = -vgrid

        _debug_print(
            "[Spharmt_rs.getvrtdivspec] prepared math-coordinate arrays | "
            f"w_shape={getattr(w, 'shape', None)} | "
            f"v_shape={getattr(v, 'shape', None)} | nt={nt}"
        , flush=True)

        # regular grid.
        if self.gridtype == 'regular':
            _debug_print("I am Here0!")

            # vector harmonic analysis.
            if self.legfunc == 'stored':
                lwork = (2*nt+1)*nlat*nlon

                step_start = time.perf_counter()
                _debug_print("[Spharmt_rs.getvrtdivspec] making regular/stored contiguous arrays", flush=True)
                v = np.ascontiguousarray(v, dtype=np.float32)
                w = np.ascontiguousarray(w, dtype=np.float32)
                wvhaes = np.ascontiguousarray(self.wvhaes, dtype=np.float32)
                _debug_print(
                    "[Spharmt_rs.getvrtdivspec] finished regular/stored contiguous arrays | "
                    f"elapsed={time.perf_counter() - step_start:.3f}s"
                , flush=True)

                if self.transform_backend == 'nogil':
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhaes_nogil_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhaes_nogil_rs(v,w,self.wvhaes,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhaes_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                elif self.transform_backend == 'latpar':
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhaes_latpar_nogil_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhaes_latpar_nogil_rs(v,w,self.wvhaes,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhaes_latpar_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                else:
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhaes_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhaes_rs(v,w,self.wvhaes,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhaes_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                if ierror != 0:
                    msg = 'In return from call to vhaes in Spharmt.getvrtdivspec ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(2*nt*nlon+max(6*n2,nlon))
                step_start = time.perf_counter()
                if self.transform_backend in ('nogil', 'latpar'):
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhaec_nogil_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhaec_nogil_rs(v,w,self.wvhaec,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhaec_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                else:
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhaec_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhaec_rs(v,w,self.wvhaec,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhaec_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                if ierror != 0:
                    msg = 'In return from call to vhaec in Spharmt.getvrtdivspec ierror =  %d' % ierror
                    raise ValueError(msg)

        # gaussian grid.
        elif self.gridtype == 'gaussian':

            # vector harmonic analysis.
            if self.legfunc == 'stored':
                lwork = (2*nt+1)*nlat*nlon
                if self.transform_backend == 'nogil':
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhags_nogil_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhags_nogil_rs(v,w,self.wvhags,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhags_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                elif self.transform_backend == 'latpar':
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhags_latpar_nogil_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhags_latpar_nogil_rs(v,w,self.wvhags,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhags_latpar_nogil_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                else:
                    step_start = time.perf_counter()
                    _debug_print("[Spharmt_rs.getvrtdivspec] entering vhags_rs(...)", flush=True)
                    br,bi,cr,ci,ierror = vhags_rs(v,w,self.wvhags,lwork)
                    _debug_print(
                        "[Spharmt_rs.getvrtdivspec] finished vhags_rs(...) | "
                        f"elapsed={time.perf_counter() - step_start:.3f}s"
                    , flush=True)
                if ierror != 0:
                    msg = 'In return from call to vhags in Spharmt.getvrtdivspec ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = 2*nlat*(2*nlon*nt+3*n2)
                step_start = time.perf_counter()
                _debug_print("[Spharmt_rs.getvrtdivspec] entering vhagc_rs(...)", flush=True)
                br,bi,cr,ci,ierror = vhagc_rs(v,w,self.wvhagc,lwork)
                _debug_print(
                    "[Spharmt_rs.getvrtdivspec] finished vhagc_rs(...) | "
                    f"elapsed={time.perf_counter() - step_start:.3f}s"
                , flush=True)
                if ierror != 0:
                    msg = 'In return from call to vhagc in Spharmt.getvrtdivspec ierror =  %d' % ierror
                    raise ValueError(msg)

        # convert vector harmonic coeffs to 1d complex coefficients
        # of vorticity and divergence.
        step_start = time.perf_counter()
        _debug_print("[Spharmt_rs.getvrtdivspec] entering twodtooned_vrtdiv_rs(...)", flush=True)
        vrtspec, divspec = twodtooned_vrtdiv_rs(br,bi,cr,ci,ntrunc,rsphere)
        _debug_print(
            "[Spharmt_rs.getvrtdivspec] finished twodtooned_vrtdiv_rs(...) | "
            f"vrtspec_shape={getattr(vrtspec, 'shape', None)} | "
            f"divspec_shape={getattr(divspec, 'shape', None)} | "
            f"elapsed={time.perf_counter() - step_start:.3f}s | "
            f"total_elapsed={time.perf_counter() - total_start:.3f}s"
        , flush=True)

        if idim == 2:
            return np.squeeze(vrtspec), np.squeeze(divspec)
        else:
            return vrtspec, divspec

    def getuv(self, vrtspec, divspec):
        """
        compute vector wind on grid given complex spectral coefficients
        of vorticity and divergence.

        @param vrtspec: rank 1 or 2 numpy complex array of vorticity spectral
        coefficients, with shape (ntrunc+1)*(ntrunc+2)/2 or
        ((ntrunc+1)*(ntrunc+2)/2,nt) (where ntrunc is the triangular truncation
        and nt is the number of spectral arrays to be transformed).
        If vrtspec is rank 1, nt is assumed to be 1.

        @param divspec: rank 1 or 2 numpy complex array of divergence spectral
        coefficients, with shape (ntrunc+1)*(ntrunc+2)/2 or
        ((ntrunc+1)*(ntrunc+2)/2,nt) (where ntrunc is the triangular truncation
        and nt is the number of spectral arrays to be transformed).
        Both vrtspec and divspec must have the same shape.

        @return: C{B{ugrid, vgrid}} - rank 2 or 3 numpy float32 arrays containing
        gridded zonal and meridional winds. Shapes are either (nlat,nlon) or
        (nlat,nlon,nt).
        """

        idim = vrtspec.ndim

        shapevrt = vrtspec.shape
        shapediv = divspec.shape

        # make sure vrtspec, divspec are rank 1 or 2, and have the same shape.
        if shapevrt != shapediv:
            msg = 'vrtspec, divspec must be same size in getuv!'
            raise ValueError(msg)

        if len(shapevrt) !=1 and len(shapevrt) !=2:
            msg = 'getuv needs rank one or two input arrays!'
            raise ValueError(msg)

        # infer ntrunc from size of dataspec (dataspec must be rank 1!)
        # dataspec is assumed to have size (ntrunc+1)*(ntrunc+2)/2
        ntrunc = int(-1.5 + 0.5*math.sqrt(9.-8.*(1.-vrtspec.shape[0])))

        nlat = self.nlat
        nlon = self.nlon
        if nlat%2:                              # nlat is odd
            n2 = (nlat + 1)//2
        else:
            n2 = nlat//2
        rsphere= self.rsphere

        if len(vrtspec.shape) == 1:
            nt = 1
            vrtspec = np.reshape(vrtspec, (vrtspec.shape[0],1))
            divspec = np.reshape(divspec, (divspec.shape[0],1))
        else:
            nt = vrtspec.shape[1]

        # convert 1d complex arrays of vort, div to 2d vector harmonic arrays.
        br,bi,cr,ci = onedtotwod_vrtdiv_rs(vrtspec,divspec,nlat,rsphere)

        # regular grid.
        if self.gridtype == 'regular':

            # vector harmonic synthesis.
            if self.legfunc == 'stored':
                lwork = (2*nt+1)*nlat*nlon
                # v, w, ierror = vhses_rs(nlon, br,bi,cr,ci,self.wvhses,lwork)
                v, w, ierror = vhses_rs(br,bi,cr,ci,self.wvhses,lwork)
                if ierror != 0:
                    msg = 'In return from call to vhses in Spharmt.getuv ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(2*nt*nlon+max(6*n2,nlon))
                v, w, ierror = vhsec_rs(br,bi,cr,ci,self.wvhsec,lwork)
                if ierror != 0:
                    msg = 'In return from call to vhsec in Spharmt.getuv ierror =  %d' % ierror
                    raise ValueError(msg)

        # gaussian grid.
        elif self.gridtype == 'gaussian':

            # vector harmonic synthesis.
            if self.legfunc == 'stored':
                lwork = (2*nt+1)*nlat*nlon
                v, w, ierror = vhsgs_rs(br,bi,cr,ci,self.wvhsgs,lwork)
                if ierror != 0:
                    msg = 'In return from call to vhsgs in Spharmt.getuv ierror =  %d' % ierror
                    raise ValueError(msg)
            else:
                lwork = nlat*(2*nt*nlon+max(6*n2,nlon))
                v, w, ierror = vhsgc_rs(br,bi,cr,ci,self.wvhsgc,lwork)
                if ierror != 0:
                    msg = 'In return from call to vhsgc in Spharmt.getuv ierror =  %d' % ierror
                    raise ValueError(msg)

        # convert to u and v in geographical coordinates.
        if idim == 1:
            return np.reshape(w, (nlat,nlon)), -np.reshape(v, (nlat,nlon))
        else:
            return w,-v

    def getpsichi(self, ugrid, vgrid, ntrunc=None, return_spectra=False):
        """
        compute streamfunction and velocity potential on grid given vector wind.

        @param ugrid: rank 2 or 3 numpy float32 array containing grid of zonal
        winds.  Must have shape (nlat,nlon) or (nlat,nlon,nt), where nt is the number
        of grids to be transformed.  If ugrid is rank 2, nt is assumed to be 1.

        @param vgrid: rank 2 or 3 numpy float32 array containing grid of meridional
        winds.  Must have shape (nlat,nlon) or (nlat,nlon,nt), where nt is the number
        of grids to be transformed.  Both ugrid and vgrid must have the same shape.

        @keyword ntrunc:  optional spectral truncation limit.
        (default self.nlat-1)

        @return: C{B{psigrid, chigrid}} - rank 2 or 3 numpy float32 arrays
        of gridded streamfunction and velocity potential. Shapes are either
        (nlat,nlon) or (nlat,nlon,nt).

        If C{return_spectra=True}, also return spectral coefficients
        C{psispec, chispec}. This avoids an additional grid-to-spectral
        transform in higher-level APIs that immediately call C{grdtospec}
        on the returned grids.
        """

        # make sure ugrid,vgrid are rank 2 or 3 and same shape.
        idim = ugrid.ndim

        shapeu = ugrid.shape
        shapev = vgrid.shape

        if ntrunc is None:
            ntrunc = self.nlat-1

        if shapeu != shapev:
            msg = 'getvrtdivspec input arrays must be same shape!'
            raise ValueError(msg)


        if len(shapeu) !=2 and len(shapeu) !=3:
            msg = 'getvrtdivspec needs rank two or three arrays!'
            raise ValueError(msg)

        if shapeu[0] != self.nlat or shapeu[1] != self.nlon:
            msg = 'getpsichi needs input arrays whose first two dimensions are si%d and %d, got %d and %d' % (self.nlat, self.nlon, ugrid.shape[0], ugrid.shape[1],)
            raise ValueError(msg)

        # check ntrunc.
        if ntrunc < 0 or ntrunc+1 > ugrid.shape[0]:
            msg = 'ntrunc must be between 0 and %d' % (ugrid.shape[0]-1,)
            raise ValueError(msg)

        # compute spectral coeffs of vort, div.
        vrtspec, divspec = self.getvrtdivspec(ugrid, vgrid, ntrunc)

        # number of grids to compute.
        if len(vrtspec.shape) == 1:
            nt = 1
            vrtspec = np.reshape(vrtspec, ((ntrunc+1)*(ntrunc+2)//2,1))
            divspec = np.reshape(divspec, ((ntrunc+1)*(ntrunc+2)//2,1))
        else:
            nt = vrtspec.shape[1]

        # convert to spectral coeffs of psi, chi.
        if self.transform_backend in ('nogil', 'latpar'):
            psispec = invlap_nogil_rs(vrtspec, self.rsphere)
            chispec = invlap_nogil_rs(divspec, self.rsphere)
        else:
            psispec = invlap_rs(vrtspec, self.rsphere)
            chispec = invlap_rs(divspec, self.rsphere)

        # inverse transform to grid.
        psigrid =  self.spectogrd(psispec)
        chigrid =  self.spectogrd(chispec)

        if idim == 2:
            psigrid = np.squeeze(psigrid)
            chigrid = np.squeeze(chigrid)
            psispec = np.squeeze(psispec)
            chispec = np.squeeze(chispec)

        if return_spectra:
            return psigrid, chigrid, psispec, chispec
        else:
            return psigrid, chigrid

    def getgrad(self, chispec):
        """
        compute vector gradient on grid given complex spectral coefficients.

        @param chispec: rank 1 or 2 numpy complex array with shape
        (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt) containing
        complex spherical harmonic coefficients (where ntrunc is the
        triangular truncation limit and nt is the number of spectral arrays
        to be transformed). If chispec is rank 1, nt is assumed to be 1.

        @return: C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
        gridded zonal and meridional components of the vector gradient.
        Shapes are either (nlat,nlon) or (nlat,nlon,nt).
        """

        # make sure chispec is rank 1 or 2.
        idim = chispec.ndim

        if len(chispec.shape) !=1 and len(chispec.shape) !=2:
            msg = 'getgrad needs rank one or two arrays!'
            raise ValueError(msg)

        # infer ntrunc from size of chispec (chispec must be rank 1!)
        # chispec is assumed to have size (ntrunc+1)*(ntrunc+2)/2
        ntrunc = int(-1.5 + 0.5*math.sqrt(9.-8.*(1.-chispec.shape[0])))

        # number of grids to compute.
        if len(chispec.shape) == 1:
            nt = 1
            chispec = np.reshape(chispec, ((ntrunc+1)*(ntrunc+2)//2,1))
        else:
            nt = chispec.shape[1]

        # convert chispec to divspec.
        divspec = lap_rs(chispec,self.rsphere)

        # call getuv, with vrtspec=0, to get uchi,vchi.
        uchi, vchi = self.getuv(np.zeros(chispec.shape, chispec.dtype), divspec)

        if idim == 1:
            return np.squeeze(uchi), np.squeeze(vchi)
        else:
            return uchi, vchi

    def specsmooth(self, datagrid, smooth):
        """
        isotropic spectral smoothing on a sphere.

        @param datagrid: rank 2 or 3 numpy float32 array with shape (nlat,nlon) or
        (nlat,nlon,nt), where nt is the number of grids to be smoothed.  If
        datagrid is rank 2, nt is assumed to be 1.

        @param smooth: rank 1 array of length nlat containing smoothing factors
        as a function of total wavenumber.

        @return: C{B{datagrid}} - rank 2 or 3 numpy float32 array with shape
        (nlat,nlon) or (nlat,nlon,nt) containing the smoothed grids.
        """

        # check that datagrid is rank 2 or 3 with size (self.nlat, self.nlon) or
        # (self.nlat, self.nlon, nt) where nt is number of grids to transform.
        if len(datagrid.shape) > 3:
            msg = 'specsmooth needs a rank two or three array, got %d' % (len(datagrid.shape),)
            raise ValueError(msg)

        if datagrid.shape[0] != self.nlat or datagrid.shape[1] != self.nlon:
            msg = 'specsmooth needs an array of size %d by %d, got %d by %d' % (self.nlat, self.nlon, datagrid.shape[0], datagrid.shape[1],)
            raise ValueError(msg)


        # make sure smooth is rank 1, same size as datagrid.shape[0]
        if len(smooth.shape) !=1 or smooth.shape[0] != datagrid.shape[0]:
            msg = 'smooth must be rank 1 and same size as datagrid.shape[0] in specsmooth!'
            raise ValueError(msg)

        # grid to spectral transform.
        nlat = self.nlat
        dataspec = self.grdtospec(datagrid, nlat-1)

        # multiply spectral coeffs. by smoothing factor.
        dataspec = multsmoothfact_rs(dataspec, smooth)

        # spectral to grid transform.
        datagrid = self.spectogrd(dataspec)

        return datagrid
