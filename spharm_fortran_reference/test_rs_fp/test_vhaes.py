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


def summarize_init_window(name, a, b, radius=3):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    idx = int(np.argmax(np.abs(diff)))
    lo = max(0, idx - radius)
    hi = min(a.size, idx + radius + 1)
    print(f"\n[{name} window]")
    print("worst flat index:", idx)
    print("fortran window:", a[lo:hi])
    print("rust window   :", b[lo:hi])
    print("diff window   :", diff[lo:hi])


def calc_vhaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhaes = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhaes, lwork, ldwork


def make_vec_grid(nlat: int, nlon: int, nt: int):
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = np.cos((k + 1) * lon)[None, :] * (1 + lat[:, None])
        w[:, :, k] = np.sin((k + 2) * lon)[None, :] * (1 - lat[:, None])
    return v, w


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lvhaes, init_lwork, ldwork = calc_vhaes_sizes(nlat, nlon)
    wvhaes_f, ierr0_f = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork, ldwork)
    wvhaes_r, ierr0_r = rust_sp.vhaesi(nlat, nlon, lvhaes, init_lwork, ldwork)
    assert ierr0_f == 0
    assert ierr0_r == 0
    v, w = make_vec_grid(nlat, nlon, nt)
    lwork = (2 * nt + 1) * nlat * nlon
    if ityp == 0:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhaes(v, w, np.asarray(wvhaes_r, dtype=np.float32), lwork)
        br_ff, bi_ff, cr_ff, ci_ff, _ = rust_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork)
        br_rf, bi_rf, cr_rf, ci_rf, _ = fort_sp.vhaes(v, w, np.asarray(wvhaes_r, dtype=np.float32), lwork)
    else:
        br_f, bi_f, cr_f, ci_f, ierr_f = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork, ityp=ityp)
        br_r, bi_r, cr_r, ci_r, ierr_r = rust_sp.vhaes_ityp(v, w, ityp, np.asarray(wvhaes_r, dtype=np.float32), lwork)
        br_ff, bi_ff, cr_ff, ci_ff, _ = rust_sp.vhaes_ityp(v, w, ityp, np.asarray(wvhaes_f, dtype=np.float32), lwork)
        br_rf, bi_rf, cr_rf, ci_rf, _ = fort_sp.vhaes(v, w, np.asarray(wvhaes_r, dtype=np.float32), lwork, ityp=ityp)
    print(f"\n{'=' * 80}\nvhaes: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    summarize_diff("vhaesi wvhaes", wvhaes_f, wvhaes_r)
    summarize_init_window("vhaesi wvhaes", wvhaes_f, wvhaes_r)
    print("ierror fortran =", ierr_f)
    print("ierror rust    =", ierr_r)
    # 下面这些普通对照在前面已经反复看过，先注释掉，避免淹没真正的定位信息。
    # summarize_diff("vhaes br", br_f, br_r)
    # summarize_diff("vhaes bi", bi_f, bi_r)
    # summarize_diff("vhaes cr", cr_f, cr_r)
    # summarize_diff("vhaes ci", ci_f, ci_r)

    # 当前重点：定位 init 差异，以及“主过程 + Fortran init / Rust init”交叉组合下仍失败的量。
    if ityp in (0, 1):
        summarize_diff("rust(vhaes, fortran init) br", br_f, br_ff)
        summarize_diff("rust(vhaes, fortran init) bi", bi_f, bi_ff)
        summarize_diff("rust(vhaes, fortran init) cr", cr_f, cr_ff)
        summarize_diff("fortran(vhaes, rust init) br", br_f, br_rf)
        summarize_diff("fortran(vhaes, rust init) bi", bi_f, bi_rf)
        summarize_diff("fortran(vhaes, rust init) ci", ci_f, ci_rf)
    elif ityp == 2:
        summarize_diff("rust(vhaes, fortran init) cr", cr_f, cr_ff)
        summarize_diff("fortran(vhaes, rust init) ci", ci_f, ci_rf)
    elif ityp in (6, 7):
        summarize_diff("rust(vhaes, fortran init) br", br_f, br_ff)
        summarize_diff("rust(vhaes, fortran init) bi", bi_f, bi_ff)
        summarize_diff("fortran(vhaes, rust init) br", br_f, br_rf)
        summarize_diff("fortran(vhaes, rust init) bi", bi_f, bi_rf)
    elif ityp == 8:
        summarize_diff("rust(vhaes, fortran init) cr", cr_f, cr_ff)
        summarize_diff("fortran(vhaes, rust init) ci", ci_f, ci_rf)


if __name__ == "__main__":
    # 小尺寸里很多量已经接近机器误差，这里先聚焦真正暴露问题的大尺寸。
    # 如需回看小尺寸，可再放开下面被注释的 cases。
    for ityp in (0, 1, 2, 6, 7, 8):
        for case in [
            # (3, 4, 1),
            # (4, 4, 1),
            # (5, 8, 2),
            (73, 144, 1)
        ]:
            run_case(*case, ityp=ityp)
