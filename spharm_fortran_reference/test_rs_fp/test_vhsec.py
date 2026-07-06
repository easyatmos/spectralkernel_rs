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

    # --- 新增：量级 ---
    print("\n-- magnitude of a --")
    print("max |a| :", np.max(abs_a))
    print("mean|a| :", np.mean(abs_a))
    print("rms  |a|:", np.sqrt(np.mean(abs_a ** 2)))

    print("\n-- magnitude of b --")
    print("max |b| :", np.max(abs_b))
    print("mean|b| :", np.mean(abs_b))
    print("rms  |b|:", np.sqrt(np.mean(abs_b ** 2)))

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


def calc_vhaec_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhaec = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhaec, ldwork


def calc_vhsec_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhsec = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhsec, ldwork


def make_vec_grid(nlat: int, nlon: int, nt: int):
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = np.cos((k + 1) * lon)[None, :] * (1 + lat[:, None])
        w[:, :, k] = np.sin((k + 2) * lon)[None, :] * (1 - lat[:, None])
    return v, w


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (2 * nt * nlon + max(6 * imid, nlon))
    return imid * (2 * nt * nlon + max(6 * nlat, nlon))


def calc_vhses_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhses = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhses, lwork, ldwork


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lvhaec, ldwork_a = calc_vhaec_sizes(nlat, nlon)
    wvhaec, ierr0 = rust_sp.vhaeci(nlat, nlon, lvhaec, ldwork_a)
    assert ierr0 == 0
    lvhsec, ldwork_s = calc_vhsec_sizes(nlat, nlon)
    wvhsec, ierr1 = rust_sp.vhseci(nlat, nlon, lvhsec, ldwork_s)
    assert ierr1 == 0
    lvhses, init_lwork_s, ldwork_ss = calc_vhses_sizes(nlat, nlon)
    wvhses, ierr2 = rust_sp.vhsesi(nlat, nlon, lvhses, init_lwork_s, ldwork_ss)
    assert ierr2 == 0
    v, w = make_vec_grid(nlat, nlon, nt)
    lwork = calc_lwork(nlat, nlon, nt, ityp)
    if ityp == 0:
        br, bi, cr, ci, ierr_a = rust_sp.vhaec(v, w, np.asarray(wvhaec, dtype=np.float32), lwork)
        v_f, w_f, ierr_f = fort_sp.vhsec(nlon, br, bi, cr, ci, np.asarray(wvhsec, dtype=np.float32), lwork)
        v_r, w_r, ierr_r = rust_sp.vhsec(br, bi, cr, ci, np.asarray(wvhsec, dtype=np.float32), lwork)
        v_s, w_s, ierr_s = rust_sp.vhses(br, bi, cr, ci, np.asarray(wvhses, dtype=np.float32), lwork)
    else:
        br, bi, cr, ci, ierr_a = rust_sp.vhaec_ityp(v, w, ityp, np.asarray(wvhaec, dtype=np.float32), lwork)
        v_f, w_f, ierr_f = fort_sp.vhsec(nlon, br, bi, cr, ci, np.asarray(wvhsec, dtype=np.float32), lwork, ityp=ityp)
        v_r, w_r, ierr_r = rust_sp.vhsec_ityp(br, bi, cr, ci, ityp, np.asarray(wvhsec, dtype=np.float32), lwork)
        v_s, w_s, ierr_s = rust_sp.vhses_ityp(br, bi, cr, ci, ityp, np.asarray(wvhses, dtype=np.float32), lwork)
    print(f"\n{'=' * 80}\nvhsec: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    print("ierror analysis rust   =", ierr_a)
    print("ierror synthesis fort  =", ierr_f)
    print("ierror synthesis rust  =", ierr_r)
    print("ierror synthesis stored=", ierr_s)
    summarize_diff("vhsec v", v_f, v_r)
    summarize_diff("vhsec w", w_f, w_r)
    summarize_diff("rust vhsec vs vhses v", v_r, v_s)
    summarize_diff("rust vhsec vs vhses w", w_r, w_s)


if __name__ == "__main__":
#     for ityp in (0, 1, 2):
#         for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
#             run_case(*case, ityp=ityp)

    for ityp in (0, 1, 2, 3, 6):
    # for ityp in (3, 6):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
