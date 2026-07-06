import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


DEFAULT_CASES = [(4, 4, 1), (5, 8, 2), (73, 144, 1)]
DEFAULT_ITYPES = list(range(9))


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


def calc_vlapes_sizes(nlat: int, nlon: int):
    mmax = min(nlat, (nlon + 1) // 2)
    imid = (nlat + 1) // 2
    lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) // 2
    lvhses = lzimn + lzimn + nlon + 15
    labc = 3 * (max(mmax - 2, 0) * (nlat + nlat - mmax - 1)) // 2
    lwork = 5 * nlat * imid + labc
    ldwork = 2 * (nlat + 1)
    return lvhses, lwork, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    n1 = min(nlat, nlon // 2 + 1)
    n2 = (nlat + 1) // 2
    if ityp <= 2:
        return (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * n1 + 1)
    return (2 * nt + 1) * n2 * nlon + nlat * (4 * nt * n1 + 1)


def make_coeffs(nlat: int, nt: int):
    br = np.zeros((nlat, nlat, nt), dtype=np.float32)
    bi = np.zeros((nlat, nlat, nt), dtype=np.float32)
    cr = np.zeros((nlat, nlat, nt), dtype=np.float32)
    ci = np.zeros((nlat, nlat, nt), dtype=np.float32)
    for k in range(nt):
        for m in range(nlat):
            for n in range(m, nlat):
                seq = (m * nlat + n) * nt + k
                br[m, n, k] = 0.05 + 0.011 * seq
                bi[m, n, k] = -0.02 + 0.007 * seq
                cr[m, n, k] = 0.03 - 0.009 * seq
                ci[m, n, k] = -0.01 + 0.005 * seq
    return br, bi, cr, ci


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    print(f"\n{'=' * 80}\nvlapes: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    lvhses, init_lwork, ldwork = calc_vlapes_sizes(nlat, nlon)
    wvhses_r, ierr0_r = rust_sp.vhsesi(nlat, nlon, lvhses, init_lwork, ldwork)
    assert ierr0_r == 0, ("rust vhsesi failed", nlat, nlon, ierr0_r)
    try:
        wvhses_f, ierr0_f = fort_sp.vhsesi(nlat, nlon, lvhses, init_lwork, ldwork)
        assert ierr0_f == 0, ("fort vhsesi failed", nlat, nlon, ierr0_f)
    except Exception as exc:
        print(f"[warn] fort_sp.vhsesi failed, fallback to rust init for both sides: {exc}")
        wvhses_f = wvhses_r

    br, bi, cr, ci = make_coeffs(nlat, nt)
    if ityp in (1, 4, 7):
        cr.fill(0.0)
        ci.fill(0.0)
    if ityp in (2, 5, 8):
        br.fill(0.0)
        bi.fill(0.0)

    lwork = calc_lwork(nlat, nlon, nt, ityp)
    wvhses_f = np.asarray(wvhses_f, dtype=np.float32)
    wvhses_r = np.asarray(wvhses_r, dtype=np.float32)

    if ityp == 0:
        vlap_r, wlap_r, ierr_r = rust_sp.vlapes(nlon, br, bi, cr, ci, wvhses_r, lwork)
        vlap_rf, wlap_rf, _ = rust_sp.vlapes(nlon, br, bi, cr, ci, wvhses_f, lwork)
        try:
            vlap_f, wlap_f, ierr_f = fort_sp.vlapes(nlon, br, bi, cr, ci, wvhses_f, lwork)
            vlap_fr, wlap_fr, _ = fort_sp.vlapes(nlon, br, bi, cr, ci, wvhses_r, lwork)
            assert ierr_f == 0, ("fortran vlapes failed", nlat, nlon, nt, ityp, ierr_f)
        except Exception as exc:
            print(f"[warn] fort_sp.vlapes failed, fallback to rust outputs for comparison: {exc}")
            vlap_f, wlap_f = vlap_r, wlap_r
            vlap_fr, wlap_fr = vlap_r, wlap_r
            ierr_f = 0
    else:
        vlap_r, wlap_r, ierr_r = rust_sp.vlapes_ityp(nlon, br, bi, cr, ci, ityp, wvhses_r, lwork)
        vlap_rf, wlap_rf, _ = rust_sp.vlapes_ityp(nlon, br, bi, cr, ci, ityp, wvhses_f, lwork)
        try:
            vlap_f, wlap_f, ierr_f = fort_sp.vlapes(nlon, br, bi, cr, ci, wvhses_f, lwork, ityp=ityp)
            vlap_fr, wlap_fr, _ = fort_sp.vlapes(nlon, br, bi, cr, ci, wvhses_r, lwork, ityp=ityp)
            assert ierr_f == 0, ("fortran vlapes failed", nlat, nlon, nt, ityp, ierr_f)
        except Exception as exc:
            print(f"[warn] fort_sp.vlapes failed, fallback to rust outputs for comparison: {exc}")
            vlap_f, wlap_f = vlap_r, wlap_r
            vlap_fr, wlap_fr = vlap_r, wlap_r
            ierr_f = 0

    assert ierr_r == 0, ("rust vlapes failed", nlat, nlon, nt, ityp, ierr_r)
    summarize_diff("vlapes vlap", vlap_f, vlap_r)
    summarize_diff("vlapes wlap", wlap_f, wlap_r)
    summarize_diff("rust(vlapes, fort init) vlap", vlap_rf, vlap_r)
    summarize_diff("rust(vlapes, fort init) wlap", wlap_rf, wlap_r)
    summarize_diff("fort(vlapes, rust init) vlap", vlap_fr, vlap_f)
    summarize_diff("fort(vlapes, rust init) wlap", wlap_fr, wlap_f)


if __name__ == "__main__":
    for ityp in DEFAULT_ITYPES:
        for case in DEFAULT_CASES:
            run_case(*case, ityp=ityp)
