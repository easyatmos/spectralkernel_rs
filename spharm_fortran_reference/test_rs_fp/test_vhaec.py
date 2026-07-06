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


def calc_vhaec_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhaec = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhaec, ldwork


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


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lvhaec, ldwork = calc_vhaec_sizes(nlat, nlon)
    wvhaec_f, ierr0_f = fort_sp.vhaeci(nlat, nlon, lvhaec, ldwork)
    wvhaec_r, ierr0_r = rust_sp.vhaeci(nlat, nlon, lvhaec, ldwork)
    assert ierr0_f == 0
    assert ierr0_r == 0
    lvhaes = min(nlat, (nlon + 1) // 2) * ((nlat + 1) // 2) * (2 * nlat - min(nlat, (nlon + 1) // 2) + 1) + nlon + 15
    init_lwork = 3 * max(min(nlat, (nlon + 1) // 2) - 2, 0) * (2 * nlat - min(nlat, (nlon + 1) // 2) - 1) // 2 + 5 * ((nlat + 1) // 2) * nlat
    ldwork_es = 2 * (nlat + 1)
    wvhaes, ierr_es = rust_sp.vhaesi(nlat, nlon, lvhaes, init_lwork, ldwork_es)
    assert ierr_es == 0
    wvhaes_f, ierr_es_f = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork, ldwork_es)
    assert ierr_es_f == 0
    v, w = make_vec_grid(nlat, nlon, nt)
    lwork = calc_lwork(nlat, nlon, nt, ityp)
    if ityp == 0:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhaec(v, w, np.asarray(wvhaec_f, dtype=np.float32), lwork)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhaec(v, w, np.asarray(wvhaec_r, dtype=np.float32), lwork)
        br_s, bi_s, cr_s, ci_s, _ = rust_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork)
        br_fs, bi_fs, cr_fs, ci_fs, _ = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork)
    else:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhaec(v, w, np.asarray(wvhaec_f, dtype=np.float32), lwork, ityp=ityp)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhaec_ityp(v, w, ityp, np.asarray(wvhaec_r, dtype=np.float32), lwork)
        br_s, bi_s, cr_s, ci_s, _ = rust_sp.vhaes_ityp(v, w, ityp, np.asarray(wvhaes, dtype=np.float32), lwork)
        br_fs, bi_fs, cr_fs, ci_fs, _ = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork, ityp=ityp)
    print(f"\n{'=' * 80}\nvhaec: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    summarize_diff("vhaec br", br_f, br_r)
    summarize_diff("vhaec bi", bi_f, bi_r)
    summarize_diff("vhaec cr", cr_f, cr_r)
    summarize_diff("vhaec ci", ci_f, ci_r)
    summarize_diff("rust vhaec vs vhaes br", br_r, br_s)
    summarize_diff("rust vhaec vs vhaes bi", bi_r, bi_s)
    summarize_diff("rust vhaec vs vhaes cr", cr_r, cr_s)
    summarize_diff("rust vhaec vs vhaes ci", ci_r, ci_s)
    summarize_diff("fortran vhaec vs vhaes br", br_f, br_fs)
    summarize_diff("fortran vhaec vs vhaes bi", bi_f, bi_fs)
    summarize_diff("fortran vhaec vs vhaes cr", cr_f, cr_fs)
    summarize_diff("fortran vhaec vs vhaes ci", ci_f, ci_fs)


if __name__ == "__main__":
    for ityp in (0, 1, 2):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
