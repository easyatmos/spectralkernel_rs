import numpy as np
import sys

# class Tee:
#     def __init__(self, *files):
#         self.files = files
#     def write(self, data):
#         for f in self.files:
#             f.write(data)
#     def flush(self):
#         for f in self.files:
#             f.flush()

# log_file = open("test_vhses.log", "w")
# sys.stdout = Tee(sys.stdout, log_file)

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp

import os
os.environ["VHAGS_TRACE"] = "1"



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


def inspect_coeff_layout(name, arr, mp1, np1, k):
    a = np.asarray(arr)
    print(f"\n[{name} coeff layout]")
    print("shape       :", a.shape)
    print("dtype       :", a.dtype)
    print("strides     :", a.strides)
    print("C_CONTIGUOUS:", a.flags.c_contiguous)
    print("F_CONTIGUOUS:", a.flags.f_contiguous)
    print(f"value[{mp1-1},{np1-1},{k}] =", a[mp1 - 1, np1 - 1, k])

    flat_c = np.ravel(a, order="C")
    flat_f = np.ravel(a, order="F")
    nlat = a.shape[1]
    nt = a.shape[2]
    idx_rust = ((mp1 - 1) * nlat + (np1 - 1)) * nt + k
    idx_fortran_linear = (mp1 - 1) + a.shape[0] * ((np1 - 1) + nlat * k)
    print("idx_rust_formula        =", idx_rust)
    print("flat_c[idx_rust]        =", flat_c[idx_rust])
    print("flat_f[idx_rust]        =", flat_f[idx_rust])
    print("idx_fortran_linear      =", idx_fortran_linear)
    print("flat_f[idx_fortran_lin] =", flat_f[idx_fortran_linear])


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


def zero_m1_even_np1_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_all_but_m1(br, bi, cr, ci)
    if br.shape[1] > 0:
        for np1 in range(br.shape[1]):
            if (np1 + 1) % 2 == 1:
                br[1, np1, ...] = 0.0
                bi[1, np1, ...] = 0.0
                cr[1, np1, ...] = 0.0
                ci[1, np1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_odd_np1_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_all_but_m1(br, bi, cr, ci)
    if br.shape[1] > 0:
        for np1 in range(br.shape[1]):
            if (np1 + 1) % 2 == 0:
                br[1, np1, ...] = 0.0
                bi[1, np1, ...] = 0.0
                cr[1, np1, ...] = 0.0
                ci[1, np1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_odd_np1_br_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_odd_np1_only(br, bi, cr, ci)
    bi[1, ...] = 0.0
    cr[1, ...] = 0.0
    ci[1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_odd_np1_bi_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_odd_np1_only(br, bi, cr, ci)
    br[1, ...] = 0.0
    cr[1, ...] = 0.0
    ci[1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_even_np1_cos_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_even_np1_only(br, bi, cr, ci)
    bi[1, ...] = 0.0
    cr[1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_even_np1_br_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_even_np1_only(br, bi, cr, ci)
    bi[1, ...] = 0.0
    cr[1, ...] = 0.0
    ci[1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_even_np1_br_cos_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_even_np1_br_only(br, bi, cr, ci)
    # 仅保留通过 ve(cos) 进入的 br 项，去掉通过 wo(sin) 进入的伴随通道
    # 对应地将 wbar 相关的 wb 通道在系数侧置零无法直接做到，因此这里只保留 m=1
    # 并额外清空所有会经 sin 通道进入的 bi/cr/ci。
    return br, bi, cr, ci


def zero_m1_even_np1_ci_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_even_np1_only(br, bi, cr, ci)
    br[1, ...] = 0.0
    bi[1, ...] = 0.0
    cr[1, ...] = 0.0
    return br, bi, cr, ci


def zero_m1_even_np1_sin_only(br, bi, cr, ci):
    br, bi, cr, ci = zero_m1_even_np1_only(br, bi, cr, ci)
    br[1, ...] = 0.0
    ci[1, ...] = 0.0
    return br, bi, cr, ci


def calc_vhses_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhses = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhses, lwork, ldwork


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
    lvhaes, init_lwork_a, ldwork_a = calc_vhses_sizes(nlat, nlon)
    wvhaes_f, ierr0_f = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork_a, ldwork_a)
    wvhaes_r, ierr0_r = rust_sp.vhaesi(nlat, nlon, lvhaes, init_lwork_a, ldwork_a)
    assert ierr0_f == 0
    assert ierr0_r == 0
    lvhses, init_lwork_s, ldwork_s = calc_vhses_sizes(nlat, nlon)
    wvhses_f, ierr1_f = fort_sp.vhsesi(nlat, nlon, lvhses, init_lwork_s, ldwork_s)
    wvhses_r, ierr1_r = rust_sp.vhsesi(nlat, nlon, lvhses, init_lwork_s, ldwork_s)
    assert ierr1_f == 0
    assert ierr1_r == 0

    v, w = make_vec_grid(nlat, nlon, nt)
    lwork_a = (2 * nt + 1) * nlat * nlon
    if ityp == 0:
        br_fa, bi_fa, cr_fa, ci_fa, ierr_af = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork_a)
        br_ra, bi_ra, cr_ra, ci_ra, ierr_ar = rust_sp.vhaes(v, w, np.asarray(wvhaes_r, dtype=np.float32), lwork_a)
        v_f, w_f, ierr_f = fort_sp.vhses(
            nlon,
            br_fa,
            bi_fa,
            cr_fa,
            ci_fa,
            np.asarray(wvhses_f, dtype=np.float32),
            lwork_a,
        )
        v_r, w_r, ierr_r = rust_sp.vhses(br_ra, bi_ra, cr_ra, ci_ra, np.asarray(wvhses_r, dtype=np.float32), lwork_a)
        v_rf, w_rf, _ = rust_sp.vhses(br_fa, bi_fa, cr_fa, ci_fa, np.asarray(wvhses_f, dtype=np.float32), lwork_a)
        v_fr, w_fr, _ = fort_sp.vhses(nlon, br_ra, bi_ra, cr_ra, ci_ra, np.asarray(wvhses_r, dtype=np.float32), lwork_a)
    else:
        br_fa, bi_fa, cr_fa, ci_fa, ierr_af = fort_sp.vhaes(v, w, np.asarray(wvhaes_f, dtype=np.float32), lwork_a, ityp=ityp)
        br_ra, bi_ra, cr_ra, ci_ra, ierr_ar = rust_sp.vhaes_ityp(v, w, ityp, np.asarray(wvhaes_r, dtype=np.float32), lwork_a)
        v_f, w_f, ierr_f = fort_sp.vhses(
            nlon,
            br_fa,
            bi_fa,
            cr_fa,
            ci_fa,
            np.asarray(wvhses_f, dtype=np.float32),
            lwork_a,
            ityp=ityp,
        )
        v_r, w_r, ierr_r = rust_sp.vhses_ityp(br_ra, bi_ra, cr_ra, ci_ra, ityp, np.asarray(wvhses_r, dtype=np.float32), lwork_a)
        v_rf, w_rf, _ = rust_sp.vhses_ityp(br_fa, bi_fa, cr_fa, ci_fa, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a)
        v_fr, w_fr, _ = fort_sp.vhses(nlon, br_ra, bi_ra, cr_ra, ci_ra, np.asarray(wvhses_r, dtype=np.float32), lwork_a, ityp=ityp)

    print(f"\n{'=' * 80}\nvhses: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    summarize_diff("vhaesi wvhaes", wvhaes_f, wvhaes_r)
    summarize_init_window("vhaesi wvhaes", wvhaes_f, wvhaes_r)
    summarize_diff("vhsesi wvhses", wvhses_f, wvhses_r)
    summarize_init_window("vhsesi wvhses", wvhses_f, wvhses_r)
    print("ierror analysis fort   =", ierr_af)
    print("ierror analysis rust   =", ierr_ar)
    print("ierror synthesis fort  =", ierr_f)
    print("ierror synthesis rust  =", ierr_r)
    if ityp in (0, 2):
        inspect_coeff_layout("fortran br_fa", br_fa, 2, 4, 0)
        inspect_coeff_layout("fortran ci_fa", ci_fa, 2, 4, 0)
        inspect_coeff_layout("rust br_ra", br_ra, 2, 4, 0)
        inspect_coeff_layout("rust ci_ra", ci_ra, 2, 4, 0)
    summarize_diff("vhses v", v_f, v_r)
    summarize_diff("vhses w", w_f, w_r)
    summarize_diff("rust(vhses, fortran init) v", v_f, v_rf)
    summarize_diff("rust(vhses, fortran init) w", w_f, w_rf)
    summarize_diff("fortran(vhses, rust init) v", v_f, v_fr)
    summarize_diff("fortran(vhses, rust init) w", w_f, w_fr)

    if ityp in (0, 1, 2):
        br0, bi0, cr0, ci0 = zero_all_but_m0(br_fa, bi_fa, cr_fa, ci_fa)
        vm0_f, wm0_f, _ = fort_sp.vhses(
            nlon, br0, bi0, cr0, ci0, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm0_r, wm0_r, _ = rust_sp.vhses_ityp(
            br0, bi0, cr0, ci0, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m0-only vhses v", vm0_f, vm0_r)
        summarize_diff("m0-only vhses w", wm0_f, wm0_r)

        br1, bi1, cr1, ci1 = zero_all_but_m1(br_fa, bi_fa, cr_fa, ci_fa)
        vm1_f, wm1_f, _ = fort_sp.vhses(
            nlon, br1, bi1, cr1, ci1, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1_r, wm1_r, _ = rust_sp.vhses_ityp(
            br1, bi1, cr1, ci1, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-only vhses v", vm1_f, vm1_r)
        summarize_diff("m1-only vhses w", wm1_f, wm1_r)

        br1e, bi1e, cr1e, ci1e = zero_m1_even_np1_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1e_f, wm1e_f, _ = fort_sp.vhses(
            nlon, br1e, bi1e, cr1e, ci1e, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1e_r, wm1e_r, _ = rust_sp.vhses_ityp(
            br1e, bi1e, cr1e, ci1e, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1e_rf, wm1e_rf, _ = rust_sp.vhses_ityp(
            br1e, bi1e, cr1e, ci1e, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1e_fr, wm1e_fr, _ = fort_sp.vhses(
            nlon, br1e, bi1e, cr1e, ci1e, np.asarray(wvhses_r, dtype=np.float32), lwork_a, ityp=ityp
        )
        summarize_diff("m1-even-np1 vhses v", vm1e_f, vm1e_r)
        summarize_diff("m1-even-np1 vhses w", wm1e_f, wm1e_r)
        summarize_diff("m1-even-np1 rust(vhses, fortran init) v", vm1e_f, vm1e_rf)
        summarize_diff("m1-even-np1 fortran(vhses, rust init) v", vm1e_f, vm1e_fr)

        br1ec, bi1ec, cr1ec, ci1ec = zero_m1_even_np1_cos_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1ec_f, wm1ec_f, _ = fort_sp.vhses(
            nlon, br1ec, bi1ec, cr1ec, ci1ec, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1ec_r, wm1ec_r, _ = rust_sp.vhses_ityp(
            br1ec, bi1ec, cr1ec, ci1ec, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-even-cos-only vhses v", vm1ec_f, vm1ec_r)
        summarize_diff("m1-even-cos-only vhses w", wm1ec_f, wm1ec_r)

        br1ebr, bi1ebr, cr1ebr, ci1ebr = zero_m1_even_np1_br_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1ebr_f, wm1ebr_f, _ = fort_sp.vhses(
            nlon, br1ebr, bi1ebr, cr1ebr, ci1ebr, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1ebr_r, wm1ebr_r, _ = rust_sp.vhses_ityp(
            br1ebr, bi1ebr, cr1ebr, ci1ebr, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-even-br-only vhses v", vm1ebr_f, vm1ebr_r)
        summarize_diff("m1-even-br-only vhses w", wm1ebr_f, wm1ebr_r)

        br1ebc, bi1ebc, cr1ebc, ci1ebc = zero_m1_even_np1_br_cos_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1ebc_f, wm1ebc_f, _ = fort_sp.vhses(
            nlon, br1ebc, bi1ebc, cr1ebc, ci1ebc, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1ebc_r, wm1ebc_r, _ = rust_sp.vhses_ityp(
            br1ebc, bi1ebc, cr1ebc, ci1ebc, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-even-br-cos-view v", vm1ebc_f, vm1ebc_r)
        summarize_diff("m1-even-br-cos-view w", wm1ebc_f, wm1ebc_r)

        br1eci, bi1eci, cr1eci, ci1eci = zero_m1_even_np1_ci_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1eci_f, wm1eci_f, _ = fort_sp.vhses(
            nlon, br1eci, bi1eci, cr1eci, ci1eci, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1eci_r, wm1eci_r, _ = rust_sp.vhses_ityp(
            br1eci, bi1eci, cr1eci, ci1eci, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1eci_rf, wm1eci_rf, _ = rust_sp.vhses_ityp(
            br1eci, bi1eci, cr1eci, ci1eci, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1eci_fr, wm1eci_fr, _ = fort_sp.vhses(
            nlon, br1eci, bi1eci, cr1eci, ci1eci, np.asarray(wvhses_r, dtype=np.float32), lwork_a, ityp=ityp
        )
        summarize_diff("m1-even-ci-only vhses v", vm1eci_f, vm1eci_r)
        summarize_diff("m1-even-ci-only vhses w", wm1eci_f, wm1eci_r)
        summarize_diff("m1-even-ci-only rust(vhses, fortran init) v", vm1eci_f, vm1eci_rf)
        summarize_diff("m1-even-ci-only fortran(vhses, rust init) v", vm1eci_f, vm1eci_fr)

        br1es, bi1es, cr1es, ci1es = zero_m1_even_np1_sin_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1es_f, wm1es_f, _ = fort_sp.vhses(
            nlon, br1es, bi1es, cr1es, ci1es, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1es_r, wm1es_r, _ = rust_sp.vhses_ityp(
            br1es, bi1es, cr1es, ci1es, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-even-sin-only vhses v", vm1es_f, vm1es_r)
        summarize_diff("m1-even-sin-only vhses w", wm1es_f, wm1es_r)

        br1o, bi1o, cr1o, ci1o = zero_m1_odd_np1_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1o_f, wm1o_f, _ = fort_sp.vhses(
            nlon, br1o, bi1o, cr1o, ci1o, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1o_r, wm1o_r, _ = rust_sp.vhses_ityp(
            br1o, bi1o, cr1o, ci1o, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1o_rf, wm1o_rf, _ = rust_sp.vhses_ityp(
            br1o, bi1o, cr1o, ci1o, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        vm1o_fr, wm1o_fr, _ = fort_sp.vhses(
            nlon, br1o, bi1o, cr1o, ci1o, np.asarray(wvhses_r, dtype=np.float32), lwork_a, ityp=ityp
        )
        summarize_diff("m1-odd-np1 vhses v", vm1o_f, vm1o_r)
        summarize_diff("m1-odd-np1 vhses w", wm1o_f, wm1o_r)
        summarize_diff("m1-odd-np1 rust(vhses, fortran init) v", vm1o_f, vm1o_rf)
        summarize_diff("m1-odd-np1 fortran(vhses, rust init) v", vm1o_f, vm1o_fr)

        br1obr, bi1obr, cr1obr, ci1obr = zero_m1_odd_np1_br_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1obr_f, wm1obr_f, _ = fort_sp.vhses(
            nlon, br1obr, bi1obr, cr1obr, ci1obr, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1obr_r, wm1obr_r, _ = rust_sp.vhses_ityp(
            br1obr, bi1obr, cr1obr, ci1obr, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-odd-br-only vhses v", vm1obr_f, vm1obr_r)
        summarize_diff("m1-odd-br-only vhses w", wm1obr_f, wm1obr_r)

        br1obi, bi1obi, cr1obi, ci1obi = zero_m1_odd_np1_bi_only(br_fa, bi_fa, cr_fa, ci_fa)
        vm1obi_f, wm1obi_f, _ = fort_sp.vhses(
            nlon, br1obi, bi1obi, cr1obi, ci1obi, np.asarray(wvhses_f, dtype=np.float32), lwork_a, ityp=ityp
        )
        vm1obi_r, wm1obi_r, _ = rust_sp.vhses_ityp(
            br1obi, bi1obi, cr1obi, ci1obi, ityp, np.asarray(wvhses_f, dtype=np.float32), lwork_a
        )
        summarize_diff("m1-odd-bi-only vhses v", vm1obi_f, vm1obi_r)
        summarize_diff("m1-odd-bi-only vhses w", wm1obi_f, wm1obi_r)


if __name__ == "__main__":
    for ityp in (0, 1, 2):
        for case in [
            # (5, 8, 2),
            (73, 144, 1)
        ]:
            run_case(*case, ityp=ityp)
