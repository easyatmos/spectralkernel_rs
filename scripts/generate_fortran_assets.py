from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "assets" / "fortran_reference.nc"


def calc_shaec_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    return 2 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2 + nlon + 15, nlat + 1


def calc_shsec_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    return calc_shaec_sizes(nlat, nlon)


def calc_shagc_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    return nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15, nlat * (nlat + 4)


def calc_shsgc_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    return calc_shagc_sizes(nlat, nlon)


def calc_vhagc_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    return 2 * (lzz1 + labc) + nlon + imid + 15, 2 * nlat * (nlat + 1) + 1


def calc_vhsgc_sizes(nlat: int, nlon: int) -> tuple[int, int]:
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    return 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15, 2 * nlat * (nlat + 1) + 1


def calc_shift_lsav(nlon: int, nlat: int) -> int:
    return 2 * (2 * nlat + nlon + 16)


def calc_shift_lwork(nlon: int, nlat: int) -> int:
    if nlon % 2 == 0:
        return 2 * nlon * (nlat + 1)
    return nlon * (5 * nlat + 1)


def scalar_grid(nlat: int, nlon: int, nt: int) -> np.ndarray:
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    out = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        out[:, :, k] = (
            np.cos((k + 1) * lon)[None, :]
            + np.sin((k + 2) * lon)[None, :] * lat[:, None]
            + (k + 1) * 0.1 * lat[:, None] ** 2
        )
    return out


def coeffs(nlat: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((nlat, nlat, nt), dtype=np.float32)
    b = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(nlat):
            for n in range(m, nlat):
                a[m, n, k] = (m + 1) * 0.1 + (n + 1) * 0.03 + k * 0.07
                b[m, n, k] = (m + 1) * 0.05 - (n + 1) * 0.02 + k * 0.04
    return a, b


def gaussian_grid(nlat: int, nlon: int, nt: int) -> np.ndarray:
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0
    lat = np.sin(0.5 * np.pi - np.asarray(theta, dtype=np.float32))
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    out = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        out[:, :, k] = (
            np.cos((k + 1) * lon)[None, :]
            + np.sin((k + 2) * lon)[None, :] * lat[:, None]
            + 0.25 * (k + 1) * lat[:, None] ** 2
        )
    return out


def vector_grid(nlat: int, nlon: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :] + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
        w[:, :, k] = np.sin(theta)[:, None] * np.sin((k + 1) * lon)[None, :] + 0.15 * np.cos(theta)[:, None] * np.cos((k + 3) * lon)[None, :]
    return v, w


def offset_scalar(nlon: int, nlat: int) -> np.ndarray:
    dlat = np.pi / nlat
    dlon = 2.0 * np.pi / nlon
    goff = np.zeros((nlon, nlat), dtype=np.float32)
    for j in range(nlon):
        lon = 0.5 * dlon + j * dlon
        for i in range(nlat):
            lat = -0.5 * np.pi + 0.5 * dlat + i * dlat
            goff[j, i] = np.exp(np.cos(lat) * np.cos(lon) + np.cos(lat) * np.sin(lon) + np.sin(lat))
    return goff


def offset_vector(nlon: int, nlat: int) -> tuple[np.ndarray, np.ndarray]:
    dlat = np.pi / nlat
    dlon = 2.0 * np.pi / nlon
    u = np.zeros((nlon, nlat), dtype=np.float32)
    v = np.zeros((nlon, nlat), dtype=np.float32)
    for j in range(nlon):
        lon = 0.5 * dlon + j * dlon
        for i in range(nlat):
            lat = -0.5 * np.pi + 0.5 * dlat + i * dlat
            u[j, i] = np.cos(lat) * np.sin(lon) + 0.25 * np.sin(2.0 * lat)
            v[j, i] = np.sin(lat) * np.cos(lon) - 0.5 * np.cos(3.0 * lon)
    return u, v


def constants() -> tuple[np.float32, ...]:
    pi = np.float32(4.0 * np.arctan(np.float32(1.0)))
    omega = np.float32((pi + pi) / np.float32(12.0 * 24.0 * 3600.0))
    return np.float32(pi * 60.0 / 180.0), np.float32(pi / 6.0), omega, np.float32(1000.0), np.float32(1.0 / 3.0)


def build_dataset() -> xr.Dataset:
    nlat, nlon, nt = 5, 8, 2
    ds = xr.Dataset(attrs={"source": "spharm_fortran_reference.pyspharm._spherepack", "nlat": nlat, "nlon": nlon, "nt": nt})

    lshaec, ldwork = calc_shaec_sizes(nlat, nlon)
    wshaec, ierr = fort_sp.shaeci(nlat, nlon, lshaec, ldwork)
    assert ierr == 0
    g = scalar_grid(nlat, nlon, nt)
    lwork_shaec = nlat * (nt * nlon + max(3 * ((nlat + 1) // 2), nlon))
    shaec_a, shaec_b, ierr = fort_sp.shaec(g, np.asarray(wshaec, dtype=np.float32), lwork_shaec)
    assert ierr == 0

    lshsec, ldwork = calc_shsec_sizes(nlat, nlon)
    wshsec, ierr = fort_sp.shseci(nlat, nlon, lshsec, ldwork)
    assert ierr == 0
    shsec_a, shsec_b = coeffs(nlat, nt)
    lwork_shsec = nt * nlat * nlon + max(nlat * nlon, 3 * nlat * ((nlat + 1) // 2))
    shsec_g, ierr = fort_sp.shsec(nlon, shsec_a, shsec_b, np.asarray(wshsec, dtype=np.float32), lwork_shsec)
    assert ierr == 0

    lshagc, ldwork = calc_shagc_sizes(nlat, nlon)
    wshagc, ierr = fort_sp.shagci(nlat, nlon, lshagc, ldwork)
    assert ierr == 0
    ggc = gaussian_grid(nlat, nlon, nt)
    lwork_shagc = nlat * (nt * nlon + max(3 * ((nlat + 1) // 2), nlon))
    grad_a, grad_b, ierr = fort_sp.shagc(ggc, np.asarray(wshagc, dtype=np.float32), lwork_shagc)
    assert ierr == 0
    lvhsgc, ldwork = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork)
    assert ierr == 0
    lwork_gradgc = nlat * (2 * nt * nlon + max(6 * ((nlat + 1) // 2), nlon) + 2 * nlat * nt + 1)
    grad_v, grad_w, ierr = fort_sp.gradgc(nlon, grad_a, grad_b, np.asarray(wvhsgc, dtype=np.float32), lwork_gradgc)
    assert ierr == 0

    vnt = 1
    vg, wg = vector_grid(nlat, nlon, vnt)
    lvhagc, ldwork = calc_vhagc_sizes(nlat, nlon)
    wvhagc, ierr = fort_sp.vhagci(nlat, nlon, lvhagc, ldwork)
    assert ierr == 0
    lwork_vhagc = nlat * (4 * vnt * nlon + max(6 * ((nlat + 1) // 2), nlon))
    div_br, div_bi, vrt_cr, vrt_ci, ierr = fort_sp.vhagc(vg, wg, np.asarray(wvhagc, dtype=np.float32), lwork_vhagc)
    assert ierr == 0
    lshsgc, ldwork = calc_shsgc_sizes(nlat, nlon)
    wshsgc, ierr = fort_sp.shsgci(nlat, nlon, lshsgc, ldwork)
    assert ierr == 0
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lwork_divvrt = nlat * (vnt * nlon + max(3 * l2, nlon) + 2 * vnt * l1 + 1)
    div, ierr = fort_sp.divgc(nlon, div_br, div_bi, np.asarray(wshsgc, dtype=np.float32), lwork_divvrt, isym=0)
    assert ierr == 0
    vort, ierr = fort_sp.vrtgc(nlon, vrt_cr, vrt_ci, np.asarray(wshsgc, dtype=np.float32), lwork_divvrt, isym=0)
    assert ierr == 0

    snlon, snlat = 10, 6
    lsav = calc_shift_lsav(snlon, snlat)
    lwork_shift = calc_shift_lwork(snlon, snlat)
    goff = offset_scalar(snlon, snlat)
    swsav0, ierr = fort_sp.sshifti(0, snlon, snlat, lsav)
    assert ierr == 0
    greg = np.zeros((snlon, snlat + 1), dtype=np.float32)
    _, greg, ierr = fort_sp.sshifte(0, snlon, snlat, goff.copy(), greg, np.asarray(swsav0, dtype=np.float32), lsav, lwork_shift)
    assert ierr == 0
    uoff, voff = offset_vector(snlon, snlat)
    vwsav0, ierr = fort_sp.vshifti(0, snlon, snlat, lsav)
    assert ierr == 0
    ureg = np.zeros((snlon, snlat + 1), dtype=np.float32)
    vreg = np.zeros((snlon, snlat + 1), dtype=np.float32)
    _, _, ureg, vreg, ierr = fort_sp.vshifte(0, snlon, snlat, uoff.copy(), voff.copy(), ureg, vreg, np.asarray(vwsav0, dtype=np.float32), lsav, lwork_shift)
    assert ierr == 0

    alpha, beta, omega, hzero, re = constants()
    gpot_theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0
    gpot = fort_sp.gpot(np.float32(600.0), alpha, beta, omega, hzero, re, nlon, np.asarray(gpot_theta, dtype=np.float32))

    vars_ = {
        "shaec_g": (("nlat", "nlon", "nt"), g),
        "shaec_w": (("wshaec",), np.asarray(wshaec, dtype=np.float32)),
        "shaec_a": (("nlat", "nlat_coeff", "nt"), np.asarray(shaec_a, dtype=np.float32)),
        "shaec_b": (("nlat", "nlat_coeff", "nt"), np.asarray(shaec_b, dtype=np.float32)),
        "shsec_a": (("nlat", "nlat_coeff", "nt"), shsec_a),
        "shsec_b": (("nlat", "nlat_coeff", "nt"), shsec_b),
        "shsec_w": (("wshsec",), np.asarray(wshsec, dtype=np.float32)),
        "shsec_g": (("nlat", "nlon", "nt"), np.asarray(shsec_g, dtype=np.float32)),
        "grad_a": (("nlat", "nlat_coeff", "nt"), np.asarray(grad_a, dtype=np.float32)),
        "grad_b": (("nlat", "nlat_coeff", "nt"), np.asarray(grad_b, dtype=np.float32)),
        "grad_w": (("wvhsgc",), np.asarray(wvhsgc, dtype=np.float32)),
        "grad_v": (("nlat", "nlon", "nt"), np.asarray(grad_v, dtype=np.float32)),
        "grad_wind": (("nlat", "nlon", "nt"), np.asarray(grad_w, dtype=np.float32)),
        "div_br": (("nlat", "nlat_coeff", "vnt"), np.asarray(div_br, dtype=np.float32)),
        "div_bi": (("nlat", "nlat_coeff", "vnt"), np.asarray(div_bi, dtype=np.float32)),
        "vrt_cr": (("nlat", "nlat_coeff", "vnt"), np.asarray(vrt_cr, dtype=np.float32)),
        "vrt_ci": (("nlat", "nlat_coeff", "vnt"), np.asarray(vrt_ci, dtype=np.float32)),
        "divvrt_w": (("wshsgc",), np.asarray(wshsgc, dtype=np.float32)),
        "div": (("nlat", "nlon", "vnt"), np.asarray(div, dtype=np.float32)),
        "vort": (("nlat", "nlon", "vnt"), np.asarray(vort, dtype=np.float32)),
        "sshifte_goff": (("snlon", "snlat"), goff),
        "sshifte_w": (("shift_w",), np.asarray(swsav0, dtype=np.float32)),
        "sshifte_greg": (("snlon", "snlat_reg"), greg),
        "vshifte_uoff": (("snlon", "snlat"), uoff),
        "vshifte_voff": (("snlon", "snlat"), voff),
        "vshifte_w": (("shift_w",), np.asarray(vwsav0, dtype=np.float32)),
        "vshifte_ureg": (("snlon", "snlat_reg"), ureg),
        "vshifte_vreg": (("snlon", "snlat_reg"), vreg),
        "gpot": (("nlat", "nlon"), np.asarray(gpot, dtype=np.float32)),
    }
    ds = ds.assign(vars_)
    ds.attrs.update(
        lwork_shaec=lwork_shaec,
        lwork_shsec=lwork_shsec,
        lwork_gradgc=lwork_gradgc,
        lwork_divvrt=lwork_divvrt,
        shift_nlon=snlon,
        shift_nlat=snlat,
        shift_lwork=lwork_shift,
        gpot_time=600.0,
        gpot_alpha=float(alpha),
        gpot_beta=float(beta),
        gpot_omega=float(omega),
        gpot_hzero=float(hzero),
        gpot_re=float(re),
    )
    return ds


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_dataset().to_netcdf(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
