from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import spectralkernel_rs as rust_sp


ASSET = Path(__file__).with_name("assets") / "fortran_reference.nc"


def arr(ds: xr.Dataset, name: str) -> np.ndarray:
    return np.asarray(ds[name].values, dtype=np.float32)


def assert_close(actual, expected, *, atol=2e-5, rtol=2e-5):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=atol,
        rtol=rtol,
    )


def test_regular_scalar_analysis_matches_fortran_asset():
    with xr.open_dataset(ASSET) as ds:
        a, b, ierr = rust_sp.shaec(arr(ds, "shaec_g"), arr(ds, "shaec_w"), int(ds.attrs["lwork_shaec"]))

        assert ierr == 0
        assert_close(a, arr(ds, "shaec_a"))
        assert_close(b, arr(ds, "shaec_b"))


def test_regular_scalar_synthesis_matches_fortran_asset():
    with xr.open_dataset(ASSET) as ds:
        g, ierr = rust_sp.shsec(
            arr(ds, "shsec_a"),
            arr(ds, "shsec_b"),
            arr(ds, "shsec_w"),
            int(ds.attrs["lwork_shsec"]),
        )

        assert ierr == 0
        assert_close(g, arr(ds, "shsec_g"))


def test_gaussian_gradient_divergence_and_vorticity_match_fortran_asset():
    with xr.open_dataset(ASSET) as ds:
        grad_v, grad_w, ierr = rust_sp.gradgc(
            arr(ds, "grad_a"),
            arr(ds, "grad_b"),
            arr(ds, "grad_w"),
            int(ds.attrs["lwork_gradgc"]),
        )
        div, div_ierr = rust_sp.divgc(
            int(ds.attrs["nlon"]),
            arr(ds, "div_br"),
            arr(ds, "div_bi"),
            arr(ds, "divvrt_w"),
            int(ds.attrs["lwork_divvrt"]),
            isym=0,
        )
        vort, vort_ierr = rust_sp.vrtgc(
            int(ds.attrs["nlon"]),
            arr(ds, "vrt_cr"),
            arr(ds, "vrt_ci"),
            arr(ds, "divvrt_w"),
            int(ds.attrs["lwork_divvrt"]),
            isym=0,
        )

        assert ierr == div_ierr == vort_ierr == 0
        assert_close(grad_v, arr(ds, "grad_v"), atol=5e-5, rtol=5e-5)
        assert_close(grad_w, arr(ds, "grad_wind"), atol=5e-5, rtol=5e-5)
        assert_close(div, arr(ds, "div"), atol=5e-5, rtol=5e-5)
        assert_close(vort, arr(ds, "vort"), atol=5e-5, rtol=5e-5)


def test_scalar_and_vector_shift_match_fortran_asset():
    with xr.open_dataset(ASSET) as ds:
        greg, ierr = rust_sp.sshifte(
            arr(ds, "sshifte_goff"),
            arr(ds, "sshifte_w"),
            int(ds.attrs["shift_lwork"]),
            ioff=0,
        )
        ureg, vreg, vierr = rust_sp.vshifte(
            arr(ds, "vshifte_uoff"),
            arr(ds, "vshifte_voff"),
            arr(ds, "vshifte_w"),
            int(ds.attrs["shift_lwork"]),
            ioff=0,
        )

        assert ierr == vierr == 0
        assert_close(greg, arr(ds, "sshifte_greg"), atol=2e-5, rtol=2e-5)
        assert_close(ureg, arr(ds, "vshifte_ureg"), atol=2e-5, rtol=2e-5)
        assert_close(vreg, arr(ds, "vshifte_vreg"), atol=2e-5, rtol=2e-5)


def test_gpot_matches_fortran_asset_with_relaxed_subtraction_tolerance():
    with xr.open_dataset(ASSET) as ds:
        gpot, ierr = rust_sp.advec_gpot(
            int(ds.attrs["nlat"]),
            int(ds.attrs["nlon"]),
            np.float32(ds.attrs["gpot_time"]),
            np.float32(ds.attrs["gpot_alpha"]),
            np.float32(ds.attrs["gpot_beta"]),
            np.float32(ds.attrs["gpot_omega"]),
            np.float32(ds.attrs["gpot_hzero"]),
            np.float32(ds.attrs["gpot_re"]),
        )

        assert ierr == 0
        assert_close(gpot, arr(ds, "gpot"), atol=2e-3, rtol=2e-6)
