import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


# def summarize_diff(name, a, b, eps=1e-12):
#     a = np.asarray(a, dtype=np.float32)
#     b = np.asarray(b, dtype=np.float32)
#     diff = a - b
#     abs_diff = np.abs(diff)
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff|:", np.mean(abs_diff))
#     print("rms diff  :", np.sqrt(np.mean(abs_diff**2)))
#     idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
#     print("worst index:", idx)
#     print("fortran:", a[idx])
#     print("rust   :", b[idx])
#     print("diff   :", diff[idx])
#     denom = np.maximum(np.abs(b), eps)
#     print("max rel:", np.max(abs_diff / denom))
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


def scalar_saved_len(nlat, nlon):
    mmax = min(nlat, nlon // 2 + 1)
    imid = (nlat + 1) // 2
    return (imid * mmax * (2 * nlat - mmax + 1)) // 2 + nlon + 15


def vector_saved_len(nlat, nlon):
    mmax = min(nlat, (nlon + 1) // 2)
    imid = (nlat + 1) // 2
    return imid * mmax * (2 * nlat - mmax + 1) + nlon + 15


def init_work_len(nlat, nlon):
    mmax = min(nlat, nlon // 2 + 1)
    imid = (nlat + 1) // 2
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    return 5 * nlat * imid + labc


def low_level_lwork(nlat, nlon):
    return 4 * nlat * nlon + 2 * nlat * (nlat + 1)


def truncate_coeffs(a, b, mmode):
    a = np.asarray(a, dtype=np.float32).copy()
    b = np.asarray(b, dtype=np.float32).copy()
    nlat = a.shape[0]
    mp = mmode + 2
    if mp <= nlat:
        for n in range(mp - 1, nlat):
            a[: n + 1, n] = 0.0
            b[: n + 1, n] = 0.0
    return a, b


def as_field2(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        return a[:, :, 0]
    return a


def geo_vhaes(u, v, wvha, lwork):
    return fort_sp.vhaes(np.asarray(-v, np.float32), np.asarray(u, np.float32), np.asarray(wvha, np.float32), lwork)


def geo_vhses(nlon, br, bi, cr, ci, wvhs, lwork):
    v_math, w_math, ierr = fort_sp.vhses(
        nlon,
        np.asarray(br, np.float32),
        np.asarray(bi, np.float32),
        np.asarray(cr, np.float32),
        np.asarray(ci, np.float32),
        np.asarray(wvhs, np.float32),
        lwork,
    )
    return as_field2(w_math), -as_field2(v_math), ierr


def geo_vtses(nlon, br, bi, cr, ci, wvts, lwork):
    vt_math, wt_math, ierr = fort_sp.vtses(
        nlon,
        np.asarray(br, np.float32),
        np.asarray(bi, np.float32),
        np.asarray(cr, np.float32),
        np.asarray(ci, np.float32),
        np.asarray(wvts, np.float32),
        lwork,
    )
    return -as_field2(wt_math), as_field2(vt_math), ierr


def geo_grades(nlon, a, b, wvhs, lwork):
    v_math, w_math, ierr = fort_sp.grades(
        nlon,
        np.asarray(a, np.float32),
        np.asarray(b, np.float32),
        np.asarray(wvhs, np.float32),
        lwork,
    )
    return as_field2(w_math), -as_field2(v_math), ierr


def shallow_fortran_reference(nlat, nlon, mmode, itmax, dt):
    if not hasattr(fort_sp, "shallow_initial"):
        print("[warn] installed Fortran _spherepack has no shallow_initial yet; rebuild Fortran backend and rerun")
        return None

    lscalar = scalar_saved_len(nlat, nlon)
    lvector = vector_saved_len(nlat, nlon)
    lwork_init = init_work_len(nlat, nlon)
    lwork = low_level_lwork(nlat, nlon)
    ldwork_scalar = nlat + 1
    ldwork_vector = 2 * (nlat + 1)

    wsha, ierr = fort_sp.shaesi(nlat, nlon, lscalar, lwork_init, ldwork_scalar)
    assert ierr == 0
    wshs, ierr = fort_sp.shsesi(nlat, nlon, lscalar, lwork_init, ldwork_scalar)
    assert ierr == 0
    wvha, ierr = fort_sp.vhaesi(nlat, nlon, lvector, lwork_init, ldwork_vector)
    assert ierr == 0
    wvhs, ierr = fort_sp.vhsesi(nlat, nlon, lvector, lwork_init, ldwork_vector)
    assert ierr == 0
    wvts, ierr = fort_sp.vtsesi(nlat, nlon, lvector, lwork_init, ldwork_vector)
    assert ierr == 0

    u, v, p, f = fort_sp.shallow_initial(nlat, nlon)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    uxact = u.copy()
    vxact = v.copy()
    pxact = p.copy()

    vmax = np.float32(max(np.max(np.abs(uxact)), np.max(np.abs(vxact))))
    pmax = np.float32(np.max(np.abs(pxact)))
    v2max = np.float32(np.sum(uxact * uxact + vxact * vxact, dtype=np.float32))
    p2max = np.float32(np.sum(pxact * pxact, dtype=np.float32))
    pzero = np.float32(2.94e4)
    aa = np.float32(6.37122e6)
    tdt = np.float32(dt + dt)
    uold = np.zeros_like(u)
    vold = np.zeros_like(v)
    pold = np.zeros_like(p)
    metrics = np.zeros(5, dtype=np.float32)

    for ncycle in range(itmax + 1):
        br, bi, cr, ci, ierr = geo_vhaes(u, v, wvha, lwork)
        assert ierr == 0
        br, bi = truncate_coeffs(br, bi, mmode)
        cr, ci = truncate_coeffs(cr, ci, mmode)
        u, v, ierr = geo_vhses(nlon, br, bi, cr, ci, wvhs, lwork)
        assert ierr == 0

        a, b, ierr = fort_sp.shaes(np.asarray(p, dtype=np.float32), np.asarray(wsha, dtype=np.float32), lwork)
        assert ierr == 0
        a, b = truncate_coeffs(a, b, mmode)
        p, ierr = fort_sp.shses(nlon, np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), np.asarray(wshs, dtype=np.float32), lwork)
        assert ierr == 0
        p = as_field2(p)

        vort, ierr = fort_sp.vrtes(nlon, np.asarray(cr, dtype=np.float32), np.asarray(ci, dtype=np.float32), np.asarray(wshs, dtype=np.float32), lwork)
        assert ierr == 0
        vort = as_field2(vort)
        divg, ierr = fort_sp.dives(nlon, np.asarray(br, dtype=np.float32), np.asarray(bi, dtype=np.float32), np.asarray(wshs, dtype=np.float32), lwork)
        assert ierr == 0
        divg = as_field2(divg)
        ut, vt, ierr = geo_vtses(nlon, br, bi, cr, ci, wvts, lwork)
        assert ierr == 0
        gpdl, gpdt, ierr = geo_grades(nlon, a, b, wvhs, lwork)
        assert ierr == 0

        dudt = (u * (vt - divg) - v * ut - gpdl) / aa + f * v
        dvdt = -(u * (vort + ut) + v * vt + gpdt) / aa - f * u
        dpdt = -((p + pzero) * divg + v * gpdt + u * gpdl) / aa

        dvgm = np.max(np.abs(divg))
        dvmax = np.sum((u - uxact) ** 2 + (v - vxact) ** 2, dtype=np.float32)
        dpmax = np.sum((p - pxact) ** 2, dtype=np.float32)
        evmax = max(np.max(np.abs(v - vxact)), np.max(np.abs(u - uxact)))
        epmax = np.max(np.abs(p - pxact))
        metrics[:] = [evmax / vmax, epmax / pmax, np.sqrt(dvmax / v2max), np.sqrt(dpmax / p2max), dvgm]

        if ncycle == 0:
            uold = u.copy()
            vold = v.copy()
            pold = p.copy()

        unew = uold + tdt * dudt
        vnew = vold + tdt * dvdt
        pnew = pold + tdt * dpdt
        uold, vold, pold = u, v, p
        u, v, p = unew.astype(np.float32), vnew.astype(np.float32), pnew.astype(np.float32)

    return uold, vold, pold, uxact, vxact, pxact, metrics


def run_initial_compare(nlat, nlon):
    if not hasattr(fort_sp, "shallow_initial"):
        print("[warn] installed Fortran _spherepack has no shallow_initial yet; rebuild Fortran backend and rerun")
        return
    uf, vf, pf, ff = fort_sp.shallow_initial(nlat, nlon)
    ur, vr, pr, fr, ierr = rust_sp.shallow_initial(nlat, nlon)
    assert ierr == 0
    print(f"\n{'=' * 80}\nshallow initial: nlat={nlat}, nlon={nlon}\n{'=' * 80}")
    summarize_diff("u initial", uf, ur)
    summarize_diff("v initial", vf, vr)
    summarize_diff("p initial", pf, pr)
    summarize_diff("f initial", ff, fr)


def run_shallow_compare(nlat, nlon, mmode, itmax, dt):
    ref = shallow_fortran_reference(nlat, nlon, mmode, itmax, dt)
    if ref is None:
        return
    ur, vr, pr, ux, vx, px, mr, ierr = rust_sp.shallow(nlat, nlon, mmode, itmax, np.float32(dt))
    assert ierr == 0
    uf, vf, pf, uxf, vxf, pxf, mf = ref
    print(f"\n{'=' * 80}\nshallow compare: nlat={nlat}, nlon={nlon}, mmode={mmode}, itmax={itmax}, dt={dt}\n{'=' * 80}")
    summarize_diff("u final", uf, ur)
    summarize_diff("v final", vf, vr)
    summarize_diff("p final", pf, pr)
    summarize_diff("uxact", uxf, ux)
    summarize_diff("vxact", vxf, vx)
    summarize_diff("pxact", pxf, px)
    print("fortran metrics:", mf)
    print("rust metrics   :", np.asarray(mr))


if __name__ == "__main__":
    for case in [(9, 16), (17, 32)]:
        run_initial_compare(*case)
    for case in [(9, 16, 4, 1, 600.0), (17, 32, 8, 2, 600.0)]:
        run_shallow_compare(*case)
