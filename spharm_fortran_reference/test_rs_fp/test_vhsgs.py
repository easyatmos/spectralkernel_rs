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


def calc_vhags_sizes(nlat: int, nlon: int):
    lvhags = (nlat + 1) * (nlat + 1) * nlat // 2 + nlon + 15
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhags, ldwork


def calc_vhsgs_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    lvhsgs_synth = n1 * imid * (2 * nlat - n1 + 1) // 2 * 2 + nlon + 15
    lvhsgs_init = 2 * imid * lmn + nlon + 15
    lvhsgs = max(lvhsgs_synth, lvhsgs_init)
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhsgs, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    idv = nlat if ityp <= 2 else (nlat + 1) // 2
    return (2 * nt + 1) * idv * nlon


def make_vec_grid(nlat: int, nlon: int, nt: int):
    lat = np.linspace(-1.0, 1.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        phase = k + 1
        v[:, :, k] = (
            np.cos(phase * lon)[None, :] * (1.0 + lat[:, None])
            + 0.1 * np.sin((phase + 2) * lon)[None, :] * (1.0 - lat[:, None] ** 2)
        )
        w[:, :, k] = (
            np.sin((phase + 1) * lon)[None, :] * (1.0 - lat[:, None])
            + 0.15 * np.cos((phase + 3) * lon)[None, :] * (1.0 + lat[:, None] ** 2)
        )
    return v.astype(np.float32), w.astype(np.float32)


def zero_all_but_m0(br, bi, cr, ci):
    br = np.array(br, copy=True)
    bi = np.array(bi, copy=True)
    cr = np.array(cr, copy=True)
    ci = np.array(ci, copy=True)
    br[1:, ...] = 0.0
    bi[1:, ...] = 0.0
    cr[1:, ...] = 0.0
    ci[1:, ...] = 0.0
    return br, bi, cr, ci


def zero_all_but_m1(br, bi, cr, ci):
    br = np.array(br, copy=True)
    bi = np.array(bi, copy=True)
    cr = np.array(cr, copy=True)
    ci = np.array(ci, copy=True)
    br[0, ...] = 0.0
    bi[0, ...] = 0.0
    cr[0, ...] = 0.0
    ci[0, ...] = 0.0
    if br.shape[0] > 2:
        br[2:, ...] = 0.0
        bi[2:, ...] = 0.0
        cr[2:, ...] = 0.0
        ci[2:, ...] = 0.0
    return br, bi, cr, ci


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    lvhags, ldwork_a = calc_vhags_sizes(nlat, nlon)
    wvhags_f, ierr0_f = fort_sp.vhagsi(nlat, nlon, lvhags, ldwork_a)
    wvhags_r, ierr0_r = rust_sp.vhagsi(nlat, nlon, lvhags, ldwork_a)
    assert ierr0_f == 0
    assert ierr0_r == 0

    lvhsgs, ldwork_s = calc_vhsgs_sizes(nlat, nlon)
    wvhsgs_f, ierr1_f = fort_sp.vhsgsi(nlat, nlon, lvhsgs, ldwork_s)
    wvhsgs_r, ierr1_r = rust_sp.vhsgsi(nlat, nlon, lvhsgs, ldwork_s)
    assert ierr1_f == 0
    assert ierr1_r == 0

    v, w = make_vec_grid(nlat, nlon, nt)
    lwork_a = calc_lwork(nlat, nlon, nt, ityp)
    if ityp == 0:
        br_f, bi_f, cr_f, ci_f, ierr_af = fort_sp.vhags(v, w, np.asarray(wvhags_f, dtype=np.float32), lwork_a)
        br_r, bi_r, cr_r, ci_r, ierr_ar = rust_sp.vhags(v, w, np.asarray(wvhags_r, dtype=np.float32), lwork_a)
        v_f, w_f, ierr_f = fort_sp.vhsgs(
            nlon,
            br_f,
            bi_f,
            cr_f,
            ci_f,
            np.asarray(wvhsgs_f, dtype=np.float32),
            lwork_a,
        )
        v_r, w_r, ierr_r = rust_sp.vhsgs(
            br_r,
            bi_r,
            cr_r,
            ci_r,
            np.asarray(wvhsgs_r, dtype=np.float32),
            lwork_a,
        )
        v_rf, w_rf, _ = rust_sp.vhsgs(
            br_f,
            bi_f,
            cr_f,
            ci_f,
            np.asarray(wvhsgs_f, dtype=np.float32),
            lwork_a,
        )
        v_fr, w_fr, _ = fort_sp.vhsgs(
            nlon,
            br_r,
            bi_r,
            cr_r,
            ci_r,
            np.asarray(wvhsgs_r, dtype=np.float32),
            lwork_a,
        )
    else:
        br_f, bi_f, cr_f, ci_f, ierr_af = fort_sp.vhags(v, w, np.asarray(wvhags_f, dtype=np.float32), lwork_a, ityp=ityp)
        br_r, bi_r, cr_r, ci_r, ierr_ar = rust_sp.vhags_ityp(v, w, ityp, np.asarray(wvhags_r, dtype=np.float32), lwork_a)
        v_f, w_f, ierr_f = fort_sp.vhsgs(
            nlon,
            br_f,
            bi_f,
            cr_f,
            ci_f,
            np.asarray(wvhsgs_f, dtype=np.float32),
            lwork_a,
            ityp=ityp,
        )
        v_r, w_r, ierr_r = rust_sp.vhsgs_ityp(
            br_r,
            bi_r,
            cr_r,
            ci_r,
            ityp,
            np.asarray(wvhsgs_r, dtype=np.float32),
            lwork_a,
        )
        v_rf, w_rf, _ = rust_sp.vhsgs_ityp(
            br_f,
            bi_f,
            cr_f,
            ci_f,
            ityp,
            np.asarray(wvhsgs_f, dtype=np.float32),
            lwork_a,
        )
        v_fr, w_fr, _ = fort_sp.vhsgs(
            nlon,
            br_r,
            bi_r,
            cr_r,
            ci_r,
            np.asarray(wvhsgs_r, dtype=np.float32),
            lwork_a,
            ityp=ityp,
        )

    print(f"\n{'=' * 80}\nvhsgs: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    print("ierror init vhagsi fortran =", ierr0_f)
    print("ierror init vhagsi rust    =", ierr0_r)
    print("ierror init vhsgsi fortran =", ierr1_f)
    print("ierror init vhsgsi rust    =", ierr1_r)
    print("ierror analysis fortran    =", ierr_af)
    print("ierror analysis rust       =", ierr_ar)
    print("ierror synthesis fortran   =", ierr_f)
    print("ierror synthesis rust      =", ierr_r)
    summarize_diff("vhsgsi init", wvhsgs_f, wvhsgs_r)
    summarize_diff("vhsgs v", v_f, v_r)
    summarize_diff("vhsgs w", w_f, w_r)
    summarize_diff("rust(vhsgs, fortran init) v", v_f, v_rf)
    summarize_diff("rust(vhsgs, fortran init) w", w_f, w_rf)
    summarize_diff("fortran(vhsgs, rust init) v", v_f, v_fr)
    summarize_diff("fortran(vhsgs, rust init) w", w_f, w_fr)

    br0, bi0, cr0, ci0 = zero_all_but_m0(br_f, bi_f, cr_f, ci_f)
    br1, bi1, cr1, ci1 = zero_all_but_m1(br_f, bi_f, cr_f, ci_f)

    if ityp == 0:
        vm0_f, wm0_f, _ = fort_sp.vhsgs(nlon, br0, bi0, cr0, ci0, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
        vm0_r, wm0_r, _ = rust_sp.vhsgs(br0, bi0, cr0, ci0, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
        vm1_f, wm1_f, _ = fort_sp.vhsgs(nlon, br1, bi1, cr1, ci1, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
        vm1_r, wm1_r, _ = rust_sp.vhsgs(br1, bi1, cr1, ci1, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
    else:
        vm0_f, wm0_f, _ = fort_sp.vhsgs(nlon, br0, bi0, cr0, ci0, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a, ityp=ityp)
        vm0_r, wm0_r, _ = rust_sp.vhsgs_ityp(br0, bi0, cr0, ci0, ityp, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
        vm1_f, wm1_f, _ = fort_sp.vhsgs(nlon, br1, bi1, cr1, ci1, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a, ityp=ityp)
        vm1_r, wm1_r, _ = rust_sp.vhsgs_ityp(br1, bi1, cr1, ci1, ityp, np.asarray(wvhsgs_f, dtype=np.float32), lwork_a)
    summarize_diff("m0-only vhsgs v", vm0_f, vm0_r)
    summarize_diff("m0-only vhsgs w", wm0_f, wm0_r)
    summarize_diff("m1-only vhsgs v", vm1_f, vm1_r)
    summarize_diff("m1-only vhsgs w", wm1_f, wm1_r)


if __name__ == "__main__":
    for ityp in range(9):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, ityp=ityp)
