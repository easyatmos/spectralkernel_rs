import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


# def summarize_diff(name, a, b, eps=1e-12):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     abs_diff = np.abs(diff)
#     denom = np.maximum(np.abs(b), eps)
#     rel = abs_diff / denom
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff| :", np.mean(abs_diff))
#     print("rms diff   :", np.sqrt(np.mean(abs_diff**2)))
#     print("max rel    :", np.max(rel))
#     idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])
def summarize_diff(name, a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    abs_diff = np.abs(diff)

    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)

    # --- 原有误差 ---
    print("max |diff| :", np.max(abs_diff))
    print("mean|diff| :", np.mean(abs_diff))
    print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("fortran    :", a[idx])
    print("rust       :", b[idx])
    print("diff       :", diff[idx])

    # --- 新增：量级 ---
    # print("\n-- magnitude of a --")
    # print("max |a| :", np.max(abs_a))
    # print("mean|a| :", np.mean(abs_a))
    # print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    # print("\n-- magnitude of b --")
    # print("max |b| :", np.max(abs_b))
    # print("mean|b| :", np.mean(abs_b))
    # print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

    # --- 相对误差（整体）---
    denom = np.maximum(abs_b, eps)
    rel = abs_diff / denom

    print("\n-- relative error (vs b) --")
    print("max rel  :", np.max(rel))
    print("mean rel :", np.mean(rel))
    print("rms rel  :", np.sqrt(np.mean(rel ** 2)))

    # --- 一个快速判断 ---
    scale = np.max(abs_b)
    if scale > 0:
        ratio = np.max(abs_diff) / scale
        print("\n-- quick scale check --")
        print("max|diff| / max|b| =", ratio)
        if ratio < 1e-6:
            print("→ likely pure floating-point noise level")
        elif ratio < 1e-3:
            print("→ small but noticeable numerical deviation")
        else:
            print("→ significant difference (check algorithm)")


def constants():
    pi = np.float32(4.0 * np.arctan(np.float32(1.0)))
    omega = np.float32((pi + pi) / np.float32(12.0 * 24.0 * 3600.0))
    hzero = np.float32(1000.0)
    re = np.float32(1.0 / 3.0)
    alpha = np.float32(pi * np.float32(60.0) / np.float32(180.0))
    beta = np.float32(pi / np.float32(6.0))
    return alpha, beta, omega, hzero, re


def calc_shagc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lsave = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lsave, ldwork


def calc_vhsgc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lsave = 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lsave, ldwork


def low_level_lwork(nlat: int, nlon: int):
    return 4 * nlat * nlon + 2 * nlat * (nlat + 1)


def atanxy(x, y):
    if x == 0.0 and y == 0.0:
        return np.float32(0.0)
    return np.float32(np.arctan2(y, x))


def run_gpot_case(nlat: int, nlon: int, t: float):
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0
    colat = np.asarray(theta, dtype=np.float32)
    alpha, beta, omega, hzero, re = constants()

    if not hasattr(fort_sp, "gpot"):
        print("[warn] installed Fortran _spherepack has no gpot yet; rebuild Fortran backend and rerun")
        return

    h_f = fort_sp.gpot(
        np.float32(t),
        alpha,
        beta,
        omega,
        hzero,
        re,
        nlon,
        colat,
    )
    h_r, ierr_r = rust_sp.advec_gpot(nlat, nlon, np.float32(t), alpha, beta, omega, hzero, re)
    assert ierr_r == 0

    print(f"\n{'=' * 80}\nadvec gpot: nlat={nlat}, nlon={nlon}, t={t}\n{'=' * 80}")
    summarize_diff("gpot fortran vs rust", h_f, h_r)


def run_advec_smoke(nlat: int, nlon: int, dt: float, ntime: int):
    phi, pexact, errm, err2, pmax, p2, ierr = rust_sp.advec(nlat, nlon, np.float32(dt), ntime)
    assert ierr == 0
    phi = np.asarray(phi)
    pexact = np.asarray(pexact)
    print(f"\n{'=' * 80}\nadvec rust smoke: nlat={nlat}, nlon={nlon}, dt={dt}, ntime={ntime}\n{'=' * 80}")
    print("phi shape:", phi.shape)
    print("pexact shape:", pexact.shape)
    print("errm:", errm)
    print("err2:", err2)
    print("pmax:", pmax)
    print("p2:", p2)
    assert phi.shape == (nlat, nlon)
    assert pexact.shape == (nlat, nlon)
    assert np.isfinite(phi).all()
    assert np.isfinite(pexact).all()
    assert np.isfinite([errm, err2, pmax, p2]).all()


def smooth_fortran(field, nlon, wshagc, wshsgc, lwork):
    a, b, ierr_a = fort_sp.shagc(np.asarray(field, dtype=np.float32), np.asarray(wshagc, dtype=np.float32), lwork)
    assert ierr_a == 0
    out, ierr_s = fort_sp.shsgc(nlon, np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), np.asarray(wshsgc, dtype=np.float32), lwork)
    assert ierr_s == 0
    return np.asarray(out, dtype=np.float32).reshape(field.shape)


def advec_fortran_reference(nlat: int, nlon: int, dt: float, ntime: int):
    alpha, beta, omega, hzero, re = constants()
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0
    colat = np.asarray(theta, dtype=np.float32)

    lshagc, ldwork_shagc = calc_shagc_sizes(nlat, nlon)
    wshagc, ierr = fort_sp.shagci(nlat, nlon, lshagc, ldwork_shagc)
    assert ierr == 0
    wshsgc, ierr = fort_sp.shsgci(nlat, nlon, lshagc, ldwork_shagc)
    assert ierr == 0
    lvhsgc, ldwork_vhsgc = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork_vhsgc)
    assert ierr == 0
    lwork = low_level_lwork(nlat, nlon)

    pi = np.float32(4.0 * np.arctan(np.float32(1.0)))
    dlon = np.float32((pi + pi) / np.float32(nlon))
    ca = np.float32(np.cos(alpha))
    sa = np.float32(np.sin(alpha))
    u = np.zeros((nlat, nlon), dtype=np.float32)
    v = np.zeros((nlat, nlon), dtype=np.float32)
    for j in range(nlon):
        xlm = np.float32(j) * dlon
        sl = np.float32(np.sin(xlm))
        cl = np.float32(np.cos(xlm))
        for i in range(nlat):
            st = np.float32(np.cos(colat[i]))
            ct = np.float32(np.sin(colat[i]))
            cthclh = np.float32(ca * ct * cl - sa * st)
            cthslh = np.float32(ct * sl)
            xlhat = atanxy(cthclh, cthslh)
            clh = np.float32(np.cos(xlhat))
            slh = np.float32(np.sin(xlhat))
            cth = np.float32(clh * cthclh + slh * cthslh)
            uhat = np.float32(omega * cth)
            u[i, j] = np.float32((ca * sl * slh + cl * clh) * uhat)
            v[i, j] = np.float32((ca * st * cl * slh - st * sl * clh + sa * ct * slh) * uhat)

    phold = fort_sp.gpot(np.float32(-dt), alpha, beta, omega, hzero, re, nlon, colat)
    phi = fort_sp.gpot(np.float32(0.0), alpha, beta, omega, hzero, re, nlon, colat)
    phold = smooth_fortran(phold, nlon, wshagc, wshsgc, lwork)
    phi = smooth_fortran(phi, nlon, wshagc, wshsgc, lwork)

    pmax = np.float32(np.max(np.abs(phi)))
    p2 = np.float32(np.sqrt(np.sum(phi * phi, dtype=np.float32)))
    time = np.float32(0.0)
    tdt = np.float32(dt + dt)
    phi_report = phi.copy()
    pexact = fort_sp.gpot(time, alpha, beta, omega, hzero, re, nlon, colat)
    errm = np.float32(0.0)
    err2 = np.float32(0.0)
    for _ in range(ntime + 1):
        a, b, ierr_a = fort_sp.shagc(np.asarray(phi, dtype=np.float32), np.asarray(wshagc, dtype=np.float32), lwork)
        assert ierr_a == 0
        gdpht, gdphl, ierr_g = fort_sp.gradgc(nlon, np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), np.asarray(wvhsgc, dtype=np.float32), lwork)
        assert ierr_g == 0
        gdpht = np.asarray(gdpht, dtype=np.float32).reshape(nlat, nlon)
        gdphl = np.asarray(gdphl, dtype=np.float32).reshape(nlat, nlon)

        pexact = fort_sp.gpot(time, alpha, beta, omega, hzero, re, nlon, colat)
        phi_report = phi.copy()
        diff = np.asarray(pexact, dtype=np.float32) - phi
        errm = np.float32(np.max(np.abs(diff)) / pmax)
        err2 = np.float32(np.sqrt(np.sum(diff * diff, dtype=np.float32)) / p2)

        dpdt = np.float32(-1.0) * u * gdphl + v * gdpht
        phnew = phold + tdt * dpdt
        phold = phi
        phi = np.asarray(phnew, dtype=np.float32)
        time = np.float32(time + dt)
    return phi_report, np.asarray(pexact, dtype=np.float32), errm, err2, pmax, p2


def run_advec_compare(nlat: int, nlon: int, dt: float, ntime: int):
    phi_f, exact_f, errm_f, err2_f, pmax_f, p2_f = advec_fortran_reference(nlat, nlon, dt, ntime)
    phi_r, exact_r, errm_r, err2_r, pmax_r, p2_r, ierr = rust_sp.advec(nlat, nlon, np.float32(dt), ntime)
    assert ierr == 0
    print(f"\n{'=' * 80}\nadvec compare: nlat={nlat}, nlon={nlon}, dt={dt}, ntime={ntime}\n{'=' * 80}")
    summarize_diff("phi fortran-reference vs rust", phi_f, phi_r)
    summarize_diff("pexact fortran-reference vs rust", exact_f, exact_r)
    print("fortran metrics:", errm_f, err2_f, pmax_f, p2_f)
    print("rust metrics   :", errm_r, err2_r, pmax_r, p2_r)


if __name__ == "__main__":
    for case in [(5, 8, 0.0), (23, 45, 600.0), (23, 45, 12.0 * 3600.0)]:
        run_gpot_case(*case)
    for case in [(5, 8, 600.0, 1), (23, 45, 600.0, 2)]:
        run_advec_smoke(*case)
    for case in [(5, 8, 600.0, 1), (23, 45, 600.0, 2)]:
        run_advec_compare(*case)
