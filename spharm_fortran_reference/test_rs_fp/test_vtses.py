import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


# def summarize_diff(name, a, b, eps=1e-12):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b

#     abs_b = np.abs(b)
#     abs_diff = np.abs(diff)

#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(abs_diff))
#     print("mean|diff| :", np.mean(abs_diff))
#     print("rms diff   :", np.sqrt(np.mean(abs_diff ** 2)))

#     idx = np.unravel_index(np.argmax(abs_diff), diff.shape)
#     print("worst index:", idx)
#     print("fortran    :", a[idx])
#     print("rust       :", b[idx])
#     print("diff       :", diff[idx])

#     denom = np.maximum(abs_b, eps)
#     rel = abs_diff / denom
#     print("\n-- relative error (vs b) --")
#     print("max rel  :", np.max(rel))
#     print("mean rel :", np.mean(rel))
#     print("rms rel  :", np.sqrt(np.mean(rel ** 2)))
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


def calc_wvts_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    lzimn = imid * mmax * (2 * nlat - mmax + 1) // 2
    lwvts = 2 * lzimn + nlon + 15
    lwork = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2 + 5 * imid * nlat
    ldwork = 2 * (nlat + 1)
    return lwvts, lwork, ldwork


def calc_synth_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    idv = nlat if ityp <= 2 else (nlat + 1) // 2
    return (2 * nt + 1) * idv * nlon


def make_coeffs(nlat: int, nt: int):
    base = np.arange(nlat * nlat * nt, dtype=np.float32).reshape(nlat, nlat, nt)
    br = 0.05 + 0.011 * base
    bi = -0.02 + 0.007 * base
    cr = 0.03 - 0.009 * base
    ci = -0.01 + 0.005 * base
    return br.astype(np.float32), bi.astype(np.float32), cr.astype(np.float32), ci.astype(np.float32)


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lwvts, lwork_init, ldwork = calc_wvts_sizes(nlat, nlon)
    wvts_f, ierr0_f = fort_sp.vtsesi(nlat, nlon, lwvts, lwork_init, ldwork)
    wvts_r, ierr0_r = rust_sp.vtsesi(nlat, nlon, lwvts, ldwork)
    assert ierr0_f == 0, ("fortran vtsesi failed", nlat, nlon, lwvts, lwork_init, ldwork, ierr0_f)
    assert ierr0_r == 0, ("rust vtsesi failed", nlat, nlon, lwvts, ldwork, ierr0_r)

    br, bi, cr, ci = make_coeffs(nlat, nt)
    if ityp in (1, 4, 7):
        cr.fill(0.0)
        ci.fill(0.0)
    if ityp in (2, 5, 8):
        br.fill(0.0)
        bi.fill(0.0)

    lwork = calc_synth_lwork(nlat, nlon, nt, ityp)
    wvts_f = np.asarray(wvts_f, dtype=np.float32)
    wvts_r = np.asarray(wvts_r, dtype=np.float32)

    if ityp == 0:
        vt_f, wt_f, ierr_f = fort_sp.vtses(nlon, br, bi, cr, ci, wvts_f, lwork)
        vt_r, wt_r, ierr_r = rust_sp.vtses(br, bi, cr, ci, wvts_r, lwork)
        vt_rf, wt_rf, _ = rust_sp.vtses(br, bi, cr, ci, wvts_f, lwork)
        vt_fr, wt_fr, _ = fort_sp.vtses(nlon, br, bi, cr, ci, wvts_r, lwork)
    else:
        vt_f, wt_f, ierr_f = fort_sp.vtses(nlon, br, bi, cr, ci, wvts_f, lwork, ityp=ityp)
        vt_r, wt_r, ierr_r = rust_sp.vtses_ityp(br, bi, cr, ci, ityp, wvts_r, lwork)
        vt_rf, wt_rf, _ = rust_sp.vtses_ityp(br, bi, cr, ci, ityp, wvts_f, lwork)
        vt_fr, wt_fr, _ = fort_sp.vtses(nlon, br, bi, cr, ci, wvts_r, lwork, ityp=ityp)

    print(f"\n{'=' * 80}\nvtses: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    print("ierror init fortran =", ierr0_f)
    print("ierror init rust    =", ierr0_r)
    print("ierror fortran      =", ierr_f)
    print("ierror rust         =", ierr_r)
    summarize_diff("vtsesi init", wvts_f, wvts_r)
    summarize_diff("vtses vt", vt_f, vt_r)
    summarize_diff("vtses wt", wt_f, wt_r)
    summarize_diff("rust(vtses, fortran init) vt", vt_f, vt_rf)
    summarize_diff("rust(vtses, fortran init) wt", wt_f, wt_rf)
    summarize_diff("fortran(vtses, rust init) vt", vt_f, vt_fr)
    summarize_diff("fortran(vtses, rust init) wt", wt_f, wt_fr)


if __name__ == "__main__":
    for ityp in range(9):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
