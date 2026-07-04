import numpy as _np

from . import spectralkernel_rs as _native
from .spectralkernel_rs import *


class _SphereOpsProxy:
    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)

    @property
    def __class__(self):
        return self._native.__class__

    def grad(self, f):
        return self._native.gradient_grid(_as_float32(f))

    def gradient(self, f):
        return self._native.gradient_grid(_as_float32(f))


class SphereOps:
    @staticmethod
    def regular(nlat, nlon, radius=6.3712e6, legfunc="stored"):
        return _SphereOpsProxy(_native.SphereOps.regular(nlat, nlon, radius=radius, legfunc=legfunc))

    @staticmethod
    def gaussian(nlat, nlon, radius=6.3712e6, legfunc="stored"):
        return _SphereOpsProxy(_native.SphereOps.gaussian(nlat, nlon, radius=radius, legfunc=legfunc))


def _unwrap_ops(ops):
    return getattr(ops, "_native", ops)


def _as_float32(array):
    return _np.asarray(array, dtype=_np.float32)


def scalar_advection_rhs(ops, q, u, v, form="advective"):
    return _native.scalar_advection_rhs(
        _unwrap_ops(ops),
        _as_float32(q),
        _as_float32(u),
        _as_float32(v),
        form=form,
    )


def shallow_water_rhs(
    ops,
    h,
    u,
    v,
    g=9.81,
    omega=7.292115e-5,
    form="vector_invariant",
    diffusion=None,
    mass_correction=False,
):
    dhdt, dudt, dvdt = _native.shallow_water_rhs(
        _unwrap_ops(ops),
        _as_float32(h),
        _as_float32(u),
        _as_float32(v),
        g=g,
        omega=omega,
        form=form,
    )
    if diffusion is not None:
        order = diffusion.get("order", 4)
        tau = diffusion.get("tau")
        nu = diffusion.get("nu")
        native_ops = _unwrap_ops(ops)
        dhdt = dhdt + native_ops.hyperdiffusion(dhdt, order=order, tau=tau, nu=nu)
        du_diff, dv_diff = native_ops.vector_hyperdiffusion(dudt, dvdt, order=order, tau=tau, nu=nu)
        dudt = dudt + du_diff
        dvdt = dvdt + dv_diff
    if mass_correction:
        dhdt = dhdt - _np.mean(dhdt, axis=(0, 1), keepdims=True)
    return dhdt, dudt, dvdt
