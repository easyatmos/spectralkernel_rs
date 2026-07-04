"""

"""
import numpy as np
from spectralkernel_rs.spectralkernel_rs import (
    gaqd as gaqd_rs,
    getlegfunc as getlegfunc_rs,
    ihgeod as ihgeod_rs,
    specintrp as _specintrp_rs,
    multsmoothfact as multsmoothfact_rs,
)


__all__ = [
    "gaussian_lats_wts_rs",
    "legendre_rs",
    "getgeodesicpts_rs",
    "specintrp_rs",
    "getspecindx"
    "regrid_rs",
]

def gaussian_lats_wts_rs(nlat):
    """
    compute the gaussian latitudes (in degrees) and quadrature weights.

    @param nlat: number of gaussian latitudes desired.

    @return: C{B{lats, wts}} - rank 1 numpy float64 arrays containing
    gaussian latitudes (in degrees north) and gaussian quadrature weights.
    """

    # get the gaussian colatitudes and weights using gaqd.
    colats, wts, ierror = gaqd_rs(nlat)

    if ierror != 0:
        msg = 'In return from call to gaqd ierror =  %d' % ierror
        raise ValueError(msg)

    # convert to degrees north latitude.
    lats = 90.0 - colats*180.0/np.pi

    return lats, wts


def legendre_rs(lat,ntrunc):
    """
    calculate associated legendre functions for triangular truncation T(ntrunc),
    at a given latitude.

    @param lat:  the latitude (in degrees) to compute the associate legendre
    functions.

    @param ntrunc:  the triangular truncation limit.

    @return: C{B{pnm}} - rank 1 numpy float32 array containing the
    (C{B{ntrunc}}+1)*(C{B{ntrunc}}+2)/2 associated legendre functions at
    latitude C{B{lat}}.
    """
    return getlegfunc_rs(lat,ntrunc)


def getgeodesicpts_rs(m):
    """
    computes the lat/lon values of the points on the surface of the sphere
    corresponding to a twenty-sided (icosahedral) geodesic.

    @param m: the number of points on the edge of a single geodesic triangle.
    There are 10*(m-1)**2+2 total geodesic points, including the poles.

    @return: C{B{lats, lons}} - rank 1 numpy float32 arrays containing
    the latitudes and longitudes of the geodesic points (in degrees). These
    points are nearly evenly distributed on the surface of the sphere.
    """
    x,y,z = ihgeod_rs(m)

    # convert cartesian coords to lat/lon.
    rad2dg = 180./np.pi
    r1 = x*x+y*y
    r = np.sqrt(r1+z*z)
    r1 = np.sqrt(r1)
    xtmp = np.where(np.logical_or(x,y),x,np.ones(x.shape,np.float32))
    ztmp = np.where(np.logical_or(r1,z),z,np.ones(z.shape,np.float32))
    lons = rad2dg*np.arctan2(y,xtmp)+180.
    lats = rad2dg*np.arctan2(r1,ztmp)-90.
    lat = np.zeros(10*(m-1)**2+2,np.float32)
    lon = np.zeros(10*(m-1)**2+2,np.float32)

    # first two points are poles.
    lat[0] = 90; lat[1] = -90.
    lon[0] = 0.; lon[1] = 0.
    lat[2:] = lats[0:2*(m-1),0:m-1,:].flatten()
    lon[2:] = lons[0:2*(m-1),0:m-1,:].flatten()
    return lat,lon


def specintrp_rs(lon,dataspec,legfuncs):
    """
    spectral interpolation given spherical harmonic coefficients.

    @param lon: longitude (in degrees) of point on a sphere to interpolate to.

    @param dataspec:  spectral coefficients of function to interpolate.

    @param legfuncs: associated legendre functions with same triangular
    truncation as C{B{dataspec}} (computed using L{legendre}), computed
    at latitude of interpolation point.

    @return: C{B{ob}} - interpolated value.
    """
    ntrunc1 = int(-1.5 + 0.5*np.sqrt(9.-8.*(1.-dataspec.shape[0])))
    ntrunc2 = int(-1.5 + 0.5*np.sqrt(9.-8.*(1.-legfuncs.shape[0])))
    if ntrunc1 != ntrunc2:
        raise ValueError('first dimensions of dataspec and legfuncs in Spharmt.specintrp imply inconsistent spectral truncations - they must be the same!')
    return _specintrp_rs((np.pi/180.)*lon,ntrunc1,dataspec,legfuncs)


def getspecindx(ntrunc):
    """
    compute indices of zonal wavenumber (indxm) and degree (indxn)
    for complex spherical harmonic coefficients.

    @param ntrunc: spherical harmonic triangular truncation limit.

    @return: C{B{indxm, indxn}} - rank 1 numpy Int32 arrays
    containing zonal wavenumber (indxm) and degree (indxn) of
    spherical harmonic coefficients.
    """

    indexn = np.indices((ntrunc+1,ntrunc+1))[1,:,:]
    indexm = np.indices((ntrunc+1,ntrunc+1))[0,:,:]
    indices = np.nonzero(np.greater(indexn, indexm-1).flatten())
    indxn = np.take(indexn.flatten(),indices)
    indxm = np.take(indexm.flatten(),indices)

    return np.squeeze(indxm), np.squeeze(indxn)


def regrid_rs(grdin, grdout, datagrid, ntrunc=None, smooth=None):
    """
    regrid data using spectral interpolation, while performing
    optional spectral smoothing and/or truncation.

    @param grdin: Spharmt class instance describing input grid.

    @param grdout: Spharmt class instance describing output grid.

    @param datagrid: data on input grid (grdin.nlat x grdin.nlon). If
    datagrid is rank 3, last dimension is the number of grids to interpolate.

    @keyword ntrunc:  optional spectral truncation limit for datagrid
    (default min(grdin.nlat-1,grdout.nlat-1)).

    @keyword smooth: rank 1 array of length grdout.nlat containing smoothing
    factors as a function of total wavenumber (default is no smoothing).

    @return: C{B{datagrid}} - interpolated (and optionally smoothed) array(s)
    on grdout.nlon x grdout.nlat grid.
    """

    # check that datagrid is rank 2 or 3 with size (grdin.nlat, grdin.nlon) or
    # (grdin.nlat, grdin.nlon, nt) where nt is number of grids to transform.

    if len(datagrid.shape) > 3:
        msg = 'regrid needs a rank two or three array, got %d' % (len(datagrid.shape),)
        raise ValueError(msg)

    if datagrid.shape[0] != grdin.nlat or datagrid.shape[1] != grdin.nlon:
        msg = 'grdtospec needs an array of size %d by %d, got %d by %d' % (grdin.nlat, grdin.nlon, datagrid.shape[0], datagrid.shape[1],)
        raise ValueError(msg)

    if smooth is not None and (len(smooth.shape) !=1 or smooth.shape[0] != grdout.nlat):
        msg = 'smooth must be rank 1 size grdout.nlat in regrid!'
        raise ValueError(msg)

    if ntrunc is None:
        ntrunc = min(grdout.nlat-1,grdin.nlat-1)

    dataspec = grdin.grdtospec(datagrid,ntrunc)

    if smooth is not None:
        dataspec = multsmoothfact_rs(dataspec, smooth)

    return grdout.spectogrd(dataspec)