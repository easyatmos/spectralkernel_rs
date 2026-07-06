import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


# def summarize_diff(name, a, b):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     diff = a - b
#     print(f"\n[{name}]")
#     print("shape:", a.shape, b.shape)
#     print("dtype:", a.dtype, b.dtype)
#     print("max |diff| :", np.max(np.abs(diff)))
#     print("mean|diff| :", np.mean(np.abs(diff)))
#     print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
#     idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
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


def calc_vhagc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhagc, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def make_vec_grid(nlat: int, nlon: int, nt: int):
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        phase = k + 1
        v[:, :, k] = (
            np.cos(phase * lon)[None, :] * (1.0 + lat[:, None])
            + 0.15 * np.sin((phase + 2) * lon)[None, :] * (1.0 - lat[:, None] ** 2)
        )
        w[:, :, k] = (
            np.sin((phase + 1) * lon)[None, :] * (1.0 - lat[:, None])
            + 0.2 * np.cos((phase + 3) * lon)[None, :] * (1.0 + lat[:, None] ** 2)
        )
    return v.astype(np.float32), w.astype(np.float32)


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lvhagc, ldwork = calc_vhagc_sizes(nlat, nlon)
    wvhagc_f, ierr0_f = fort_sp.vhagci(nlat, nlon, lvhagc, ldwork)
    wvhagc_r, ierr0_r = rust_sp.vhagci(nlat, nlon, lvhagc, ldwork)
    assert ierr0_f == 0, ("fortran vhagci failed", nlat, nlon, lvhagc, ldwork, ierr0_f)
    assert ierr0_r == 0, ("rust vhagci failed", nlat, nlon, lvhagc, ldwork, ierr0_r)

    v, w = make_vec_grid(nlat, nlon, nt)
    lwork = calc_lwork(nlat, nlon, nt, ityp)

    if ityp == 0:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhagc(v, w, np.asarray(wvhagc_f, dtype=np.float32), lwork)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhagc(v, w, np.asarray(wvhagc_r, dtype=np.float32), lwork)
        br_rf, bi_rf, cr_rf, ci_rf, _ = rust_sp.vhagc(v, w, np.asarray(wvhagc_f, dtype=np.float32), lwork)
        br_fr, bi_fr, cr_fr, ci_fr, _ = fort_sp.vhagc(v, w, np.asarray(wvhagc_r, dtype=np.float32), lwork)
    else:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhagc(v, w, np.asarray(wvhagc_f, dtype=np.float32), lwork, ityp=ityp)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhagc_ityp(v, w, ityp, np.asarray(wvhagc_r, dtype=np.float32), lwork)
        br_rf, bi_rf, cr_rf, ci_rf, _ = rust_sp.vhagc_ityp(v, w, ityp, np.asarray(wvhagc_f, dtype=np.float32), lwork)
        br_fr, bi_fr, cr_fr, ci_fr, _ = fort_sp.vhagc(v, w, np.asarray(wvhagc_r, dtype=np.float32), lwork, ityp=ityp)

    print(f"\n{'=' * 80}\nvhagc: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    print("ierror init fortran =", ierr0_f)
    print("ierror init rust    =", ierr0_r)
    print("ierror fortran      =", ierr_f)
    print("ierror rust         =", ierr_r)
    summarize_diff("vhagc init", wvhagc_f, wvhagc_r)
    summarize_diff("vhagc br", br_f, br_r)
    summarize_diff("vhagc bi", bi_f, bi_r)
    summarize_diff("vhagc cr", cr_f, cr_r)
    summarize_diff("vhagc ci", ci_f, ci_r)
    summarize_diff("rust(vhagc, fortran init) br", br_f, br_rf)
    summarize_diff("rust(vhagc, fortran init) bi", bi_f, bi_rf)
    summarize_diff("rust(vhagc, fortran init) cr", cr_f, cr_rf)
    summarize_diff("rust(vhagc, fortran init) ci", ci_f, ci_rf)
    summarize_diff("fortran(vhagc, rust init) br", br_f, br_fr)
    summarize_diff("fortran(vhagc, rust init) bi", bi_f, bi_fr)
    summarize_diff("fortran(vhagc, rust init) cr", cr_f, cr_fr)
    summarize_diff("fortran(vhagc, rust init) ci", ci_f, ci_fr)


if __name__ == "__main__":
    for ityp in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
