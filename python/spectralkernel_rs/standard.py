"""Spherical harmonic vector wind computations."""
# Copyright (c) 2012-2018 Andrew Dawson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import time

import numpy as np
from ._common import _debug_print
from .spharmt_rs import Spharmt_rs as Spharmt
from .tool import gaussian_lats_wts_rs as gaussian_lats_wts


_SPHARMT_CACHE = {}


def _get_cached_spharmt(nlon, nlat, gridtype, rsphere, legfunc, transform_backend):
    key = (int(nlon), int(nlat), str(gridtype), float(rsphere), str(legfunc), str(transform_backend))
    spharmt = _SPHARMT_CACHE.get(key)
    if spharmt is None:
        spharmt = Spharmt(nlon, nlat, gridtype=gridtype, rsphere=rsphere,
                         legfunc=legfunc, transform_backend=transform_backend)
        _SPHARMT_CACHE[key] = spharmt
    return spharmt


class VectorWind(object):
    """Vector Wind computations (standard `numpy` interface)."""

    def __init__(self, u, v, gridtype='regular', rsphere=6.3712e6,
                 legfunc='stored', transform_backend='standard'):
        """Initialize a VectorWind instance.

        **Arguments:**

        *u*, *v*
            Zonal and meridional wind components respectively. Their
            types should be either `numpy.ndarray` or
            `numpy.ma.MaskedArray`. *u* and *v* must have matching
            shapes and contain no missing values. *u* and *v* may be 2
            or 3-dimensional with shape (nlat, nlon) or
            (nlat, nlon, nt), where nlat and nlon are the number of
            latitudes and longitudes respectively and nt is the number
            of fields. The latitude dimension must be oriented
            north-to-south. The longitude dimension should be
            oriented west-to-east.

        **Optional arguments:**

        *gridtype*
            Type of the input grid, either 'regular' for evenly-spaced
            grids, or 'gaussian' for Gaussian grids. Defaults to
            'regular'.

        *rsphere*
            The radius in metres of the sphere used in the spherical
            harmonic computations. Default is 6371200 m, the approximate
            mean spherical Earth radius.

        *legfunc*
            'stored' (default) or 'computed'.  If 'stored', associated legendre
            functions are precomputed and stored when the class instance is
            created.  This uses O(nlat**3) memory, but speeds up the spectral
            transforms.  If 'computed', associated legendre functions are
            computed on the fly when transforms are requested.  This uses
            O(nlat**2) memory, but slows down the spectral transforms a bit.

        **See also:**

        `~windspharm.tools.prep_data`,
        `~windspharm.tools.recover_data`,
        `~windspharm.tools.get_recovery`,
        `~windspharm.tools.reverse_latdim`,
        `~windspharm.tools.order_latdim`.

        **Examples:**

        Initialize a `VectorWind` instance with zonal and meridional
        components of the vector wind on the default regular
        (evenly-spaced) grid:

            from windspharm.standard import VectorWind
            w = VectorWind(u, v)

        Initialize a `VectorWind` instance with zonal and meridional
        components of the vector wind specified on a Gaussian grid:

            from windspharm.standard import VectorWind
            w = VectorWind(u, v, gridtype='gaussian')

        """
        # For both the input components check if there are missing values by
        # attempting to fill missing values with NaN and detect them. If the
        # inputs are not masked arrays then take copies and check for NaN.
        total_start = time.perf_counter()
        _debug_print(
            "[standard.VectorWind.__init__] start | "
            f"u_shape={getattr(u, 'shape', None)} | "
            f"v_shape={getattr(v, 'shape', None)} | "
            f"gridtype={gridtype} | rsphere={rsphere} | "
            f"legfunc={legfunc} | transform_backend={transform_backend}"
        , flush=True)

        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.__init__] converting/filling u input", flush=True)
        try:
            self.u = u.filled(fill_value=np.nan)
        except AttributeError:
            self.u = np.asarray(u)
        _debug_print(
            "[standard.VectorWind.__init__] finished converting/filling u input | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)

        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.__init__] converting/filling v input", flush=True)
        try:
            self.v = v.filled(fill_value=np.nan)
        except AttributeError:
            self.v = np.asarray(v)
        _debug_print(
            "[standard.VectorWind.__init__] finished converting/filling v input | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)

        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.__init__] checking NaN values", flush=True)
        if np.isnan(self.u).any() or np.isnan(self.v).any():
            raise ValueError('u and v cannot contain missing values')
        _debug_print(
            "[standard.VectorWind.__init__] finished checking NaN values | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)
        # Make sure the shapes of the two components match.
        _debug_print("[standard.VectorWind.__init__] checking input shapes", flush=True)
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')
        _debug_print("[standard.VectorWind.__init__] checking input ranks", flush=True)
        if len(u.shape) not in (2, 3):
            raise ValueError('u and v must be rank 2 or 3 arrays')
        nlat = u.shape[0]
        nlon = u.shape[1]
        try:
            # Create a Spharmt object to do the computations.
            self.gridtype = gridtype.lower()
            step_start = time.perf_counter()
            _debug_print("[standard.VectorWind.__init__] entering _get_cached_spharmt(...)", flush=True)
            self.s = _get_cached_spharmt(nlon, nlat, self.gridtype,
                                         rsphere, legfunc, transform_backend)
            _debug_print(
                "[standard.VectorWind.__init__] finished _get_cached_spharmt(...) | "
                f"elapsed={time.perf_counter() - step_start:.3f}s"
            , flush=True)
        except ValueError:
            if self.gridtype not in ('regular', 'gaussian'):
                err = 'invalid grid type: {0:s}'.format(repr(gridtype))
            else:
                err = 'invalid input dimensions'
            raise ValueError(err)
        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.__init__] making contiguous float32 arrays", flush=True)
        self.u = np.ascontiguousarray(self.u, dtype=np.float32)
        self.v = np.ascontiguousarray(self.v, dtype=np.float32)
        _debug_print(
            "[standard.VectorWind.__init__] finished making contiguous float32 arrays | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)
        # Method aliases.
        self.rotationalcomponent = self.nondivergentcomponent
        self.divergentcomponent = self.irrotationalcomponent
        _debug_print(
            "[standard.VectorWind.__init__] finished | "
            f"total_elapsed={time.perf_counter() - total_start:.3f}s"
        , flush=True)

    def magnitude(self):
        """Wind speed (magnitude of vector wind).

        **Returns:**

        *speed*
            The wind speed.

        **Example:**

        Magnitude of the vector wind::

            spd = w.magnitude()

        """
        return (self.u ** 2 + self.v ** 2) ** 0.5

    def vrtdiv(self, truncation=None):
        """Relative vorticity and horizontal divergence.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *vrt*, *div*
            The relative vorticity and divergence respectively.

        **See also:**

        `~VectorWind.vorticity`, `~VectorWind.divergence`.

        **Examples:**

        Compute the relative vorticity and divergence::

            vrt, div = w.vrtdiv()

        Compute the relative vorticity and divergence and apply spectral
        truncation at triangular T13::

            vrtT13, divT13 = w.vrtdiv(truncation=13)

        """
        vrtspec, divspec = self.s.getvrtdivspec(self.u,
                                                self.v,
                                                ntrunc=truncation)
        vrtgrid = self.s.spectogrd(vrtspec)
        divgrid = self.s.spectogrd(divspec)
        return vrtgrid, divgrid

    def vorticity(self, truncation=None):
        """Relative vorticity.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *vrt*
            The relative vorticity.

        **See also:**

        `~VectorWind.vrtdiv`, `~VectorWind.absolutevorticity`.

        **Examples:**

        Compute the relative vorticity::

            vrt = w.vorticity()

        Compute the relative vorticity and apply spectral truncation at
        triangular T13::

            vrtT13 = w.vorticity(truncation=13)

        """
        total_start = time.perf_counter()
        _debug_print(
            "[standard.VectorWind.vorticity] start | "
            f"u_shape={getattr(self.u, 'shape', None)} | "
            f"v_shape={getattr(self.v, 'shape', None)} | "
            f"truncation={truncation}"
        , flush=True)

        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.vorticity] entering self.s.getvrtdivspec(...)", flush=True)
        vrtspec, divspec = self.s.getvrtdivspec(self.u,
                                                self.v,
                                                ntrunc=truncation)
        _debug_print(
            "[standard.VectorWind.vorticity] finished self.s.getvrtdivspec(...) | "
            f"vrtspec_shape={getattr(vrtspec, 'shape', None)} | "
            f"divspec_shape={getattr(divspec, 'shape', None)} | "
            f"elapsed={time.perf_counter() - step_start:.3f}s"
        , flush=True)

        step_start = time.perf_counter()
        _debug_print("[standard.VectorWind.vorticity] entering self.s.spectogrd(vrtspec)", flush=True)
        vrtgrid = self.s.spectogrd(vrtspec)
        _debug_print(
            "[standard.VectorWind.vorticity] finished self.s.spectogrd(vrtspec) | "
            f"vrtgrid_shape={getattr(vrtgrid, 'shape', None)} | "
            f"elapsed={time.perf_counter() - step_start:.3f}s | "
            f"total_elapsed={time.perf_counter() - total_start:.3f}s"
        , flush=True)
        return vrtgrid

    def divergence(self, truncation=None):
        """Horizontal divergence.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *div*
            The divergence.

        **See also:**

        `~VectorWind.vrtdiv`.

        **Examples:**

        Compute the divergence::

            div = w.divergence()

        Compute the divergence and apply spectral truncation at
        triangular T13::

            divT13 = w.divergence(truncation=13)

        """
        vrtspec, divspec = self.s.getvrtdivspec(self.u,
                                                self.v,
                                                ntrunc=truncation)
        divgrid = self.s.spectogrd(divspec)
        return divgrid

    def planetaryvorticity(self, omega=None):
        """Planetary vorticity (Coriolis parameter).

        **Optional argument:**

        *omega*
            Earth's angular velocity. The default value if not specified
            is 7.292x10**-5 s**-1.

        **Returns:**

        *pvorticity*
            The planetary vorticity.

        **See also:**

        `~VectorWind.absolutevorticity`.

        **Example:**

        Compute planetary vorticity using default values::

            pvrt = w.planetaryvorticity()

        Override the default value for Earth's angular velocity::

            pvrt = w.planetaryvorticity(omega=7.2921150)

        """
        if omega is None:
            # Define the Earth's angular velocity.
            omega = 7.292e-05
        nlat = self.s.nlat
        if self.gridtype == 'gaussian':
            lat, wts = gaussian_lats_wts(nlat)
        else:
            if nlat % 2:
                lat = np.linspace(90, -90, nlat)
            else:
                dlat = 180. / nlat
                lat = np.arange(90 - dlat / 2., -90, -dlat)
        try:
            cp = 2. * omega * np.sin(np.deg2rad(lat))
        except (TypeError, ValueError):
            raise ValueError('invalid value for omega: {!r}'.format(omega))
        indices = [slice(0, None)] + [np.newaxis] * (len(self.u.shape) - 1)
        f = cp[tuple(indices)] * np.ones(self.u.shape, dtype=np.float32)
        return f

    def absolutevorticity(self, omega=None, truncation=None):
        """Absolute vorticity (sum of relative and planetary vorticity).

        **Optional arguments:**

        *omega*
            Earth's angular velocity. The default value if not specified
            is 7.292x10**-5 s**-1.

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *avorticity*
            The absolute (relative + planetary) vorticity.

        **See also:**

        `~VectorWind.vorticity`, `~VectorWind.planetaryvorticity`.

        **Examples:**

        Compute absolute vorticity::

            avrt = w.absolutevorticity()

        Compute absolute vorticity and apply spectral truncation at
        triangular T13, also override the default value for Earth's
        angular velocity::

            avrt = w.absolutevorticity(omega=7.2921150, truncation=13)

        """
        pvrt = self.planetaryvorticity(omega=omega)
        rvrt = self.vorticity(truncation=truncation)
        return pvrt + rvrt

    def sfvp(self, truncation=None):
        """Streamfunction and velocity potential.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *sf*, *vp*
            The streamfunction and velocity potential respectively.

        **See also:**

        `~VectorWind.streamfunction`, `~VectorWind.velocitypotential`.

        **Examples:**

        Compute streamfunction and velocity potential::

            sf, vp = w.sfvp()

        Compute streamfunction and velocity potential and apply spectral
        truncation at triangular T13::

            sfT13, vpT13 = w.sfvp(truncation=13)

        """
        psigrid, chigrid = self.s.getpsichi(self.u, self.v, ntrunc=truncation)
        return psigrid, chigrid

    def streamfunction(self, truncation=None):
        """Streamfunction.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *sf*
            The streamfunction.

        **See also:**

        `~VectorWind.sfvp`.

        **Examples:**

        Compute streamfunction::

            sf = w.streamfunction()

        Compute streamfunction and apply spectral truncation at
        triangular T13::

            sfT13 = w.streamfunction(truncation=13)

        """
        psigrid, chigrid = self.sfvp(truncation=truncation)
        return psigrid

    def velocitypotential(self, truncation=None):
        """Velocity potential.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *vp*
            The velocity potential.

        **See also:**

        `~VectorWind.sfvp`.

        **Examples:**

        Compute velocity potential::

            vp = w.velocity potential()

        Compute velocity potential and apply spectral truncation at
        triangular T13::

            vpT13 = w.velocity potential(truncation=13)

        """
        psigrid, chigrid = self.sfvp(truncation=truncation)
        return chigrid

    def helmholtz(self, truncation=None):
        """Irrotational and non-divergent components of the vector wind.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *uchi*, *vchi*, *upsi*, *vpsi*
            Zonal and meridional components of irrotational and
            non-divergent wind components respectively.

        **See also:**

        `~VectorWind.irrotationalcomponent`,
        `~VectorWind.nondivergentcomponent`.

        **Examples:**

        Compute the irrotational and non-divergent components of the
        vector wind::

            uchi, vchi, upsi, vpsi = w.helmholtz()

        Compute the irrotational and non-divergent components of the
        vector wind and apply spectral truncation at triangular T13::

            uchiT13, vchiT13, upsiT13, vpsiT13 = w.helmholtz(truncation=13)

        """
        psigrid, chigrid, psispec, chispec = self.s.getpsichi(
            self.u, self.v, ntrunc=truncation, return_spectra=True
        )
        vpsi, upsi = self.s.getgrad(psispec)
        uchi, vchi = self.s.getgrad(chispec)
        return uchi, vchi, -upsi, vpsi

    def irrotationalcomponent(self, truncation=None):
        """Irrotational (divergent) component of the vector wind.

        .. note::

           If both the irrotational and non-divergent components are
           required then `~VectorWind.helmholtz` should be used instead.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *uchi*, *vchi*
            The zonal and meridional components of the irrotational wind
            respectively.

        **See also:**

        `~VectorWind.helmholtz`.

        **Examples:**

        Compute the irrotational component of the vector wind::

            uchi, vchi = w.irrotationalcomponent()

        Compute the irrotational component of the vector wind and apply
        spectral truncation at triangular T13::

            uchiT13, vchiT13 = w.irrotationalcomponent(truncation=13)

        """
        psigrid, chigrid, psispec, chispec = self.s.getpsichi(
            self.u, self.v, ntrunc=truncation, return_spectra=True
        )
        uchi, vchi = self.s.getgrad(chispec)
        return uchi, vchi

    def nondivergentcomponent(self, truncation=None):
        """Non-divergent (rotational) component of the vector wind.

        .. note::

           If both the non-divergent and irrotational components are
           required then `~VectorWind.helmholtz` should be used instead.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *upsi*, *vpsi*
            The zonal and meridional components of the non-divergent
            wind respectively.

        **See also:**

        `~VectorWind.helmholtz`.

        **Examples:**

        Compute the non-divergent component of the vector wind::

            upsi, vpsi = w.nondivergentcomponent()

        Compute the non-divergent component of the vector wind and apply
        spectral truncation at triangular T13::

            upsiT13, vpsiT13 = w.nondivergentcomponent(truncation=13)

        """
        psigrid, chigrid, psispec, chispec = self.s.getpsichi(
            self.u, self.v, ntrunc=truncation, return_spectra=True
        )
        vpsi, upsi = self.s.getgrad(psispec)
        return -upsi, vpsi

    def gradient(self, chi, truncation=None):
        """Computes the vector gradient of a scalar field on the sphere.

        **Argument:**

        *chi*
            A scalar field. Its shape must be either (nlat, nlon) or
            (nlat, nlon, nfields) where nlat and nlon are the same
            as those for the vector wind components that initialized the
            `VectorWind` instance.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation.

        **Returns:**

        *uchi*, *vchi*
            The zonal and meridional components of the vector gradient
            respectively.

        **Examples:**

        Compute the vector gradient of absolute vorticity::

            avrt = w.absolutevorticity()
            avrt_zonal, avrt_meridional = w.gradient(avrt)

        Compute the vector gradient of absolute vorticity and apply
        spectral truncation at triangular T13::

            avrt = w.absolutevorticity()
            avrt_zonalT13, avrt_meridionalT13 = w.gradient(avrt, truncation=13)

        """
        try:
            chi = chi.filled(fill_value=np.nan)
        except AttributeError:
            pass
        if np.isnan(chi).any():
            raise ValueError('chi cannot contain missing values')
        try:
            chispec = self.s.grdtospec(chi, ntrunc=truncation)
        except ValueError:
            raise ValueError('input field is not compatitble')
        uchi, vchi = self.s.getgrad(chispec)
        return uchi, vchi

    def truncate(self, field, truncation=None):
        """Apply spectral truncation to a scalar field.

        This is useful to represent other fields in a way consistent
        with the output of other `VectorWind` methods.

        **Argument:**

        *field*
            A scalar field. Its shape must be either (nlat, nlon) or
            (nlat, nlon, nfields) where nlat and nlon are the same
            as those for the vector wind components that initialized the
            `VectorWind` instance.

        **Optional argument:**

        *truncation*
            Truncation limit (triangular truncation) for the spherical
            harmonic computation. If not specified it will default to
            *nlats - 1* where *nlats* is the number of latitudes.

        **Returns:**

        *truncated_field*
            The field with spectral truncation applied.

        **Examples:**

        Truncate a scalar field to the computational resolution of the
        `VectorWind` instance::

            scalar_field_truncated = w.truncate(scalar_field)

        Truncate a scalar field to T21::

            scalar_field_T21 = w.truncate(scalar_field, truncation=21)

        """
        try:
            field = field.filled(fill_value=np.nan)
        except AttributeError:
            pass
        if np.isnan(field).any():
            raise ValueError('field cannot contain missing values')
        try:
            fieldspec = self.s.grdtospec(field, ntrunc=truncation)
        except ValueError:
            raise ValueError('field is not compatible')
        fieldtrunc = self.s.spectogrd(fieldspec)
        return fieldtrunc
