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


def calc_vhsec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhsec = 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhsec, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    branch_nosym_all = nlat * (2 * nt * nlon + max(6 * imid, nlon) + 1) + 4 * mn
    branch_sym_all = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 4 * mn + nlat
    return branch_nosym_all if ityp <= 2 else branch_sym_all


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
    print(f"\n{'=' * 80}\nvlapec: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    lvhsec, ldwork = calc_vhsec_sizes(nlat, nlon)
    wvhsec_f, ierr0_f = fort_sp.vhseci(nlat, nlon, lvhsec, ldwork)
    wvhsec_r, ierr0_r = rust_sp.vhseci(nlat, nlon, lvhsec, ldwork)
    assert ierr0_f == 0, ("fort vhseci failed", nlat, nlon, ierr0_f)
    assert ierr0_r == 0, ("rust vhseci failed", nlat, nlon, ierr0_r)

    br, bi, cr, ci = make_coeffs(nlat, nt)
    if ityp in (1, 4, 7):
        cr.fill(0.0)
        ci.fill(0.0)
    if ityp in (2, 5, 8):
        br.fill(0.0)
        bi.fill(0.0)

    lwork = calc_lwork(nlat, nlon, nt, ityp)
    wvhsec_f = np.asarray(wvhsec_f, dtype=np.float32)
    wvhsec_r = np.asarray(wvhsec_r, dtype=np.float32)

    if ityp == 0:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_f, lwork)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_r, lwork)
        vlap_rf, wlap_rf, _ = rust_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_f, lwork)
        vlap_fr, wlap_fr, _ = fort_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_r, lwork)
    else:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_f, lwork, ityp=ityp)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapec_ityp(nlon, br, bi, cr, ci, ityp, wvhsec_r, lwork)
        vlap_rf, wlap_rf, _ = rust_sp.vlapec_ityp(nlon, br, bi, cr, ci, ityp, wvhsec_f, lwork)
        vlap_fr, wlap_fr, _ = fort_sp.vlapec(nlon, br, bi, cr, ci, wvhsec_r, lwork, ityp=ityp)

    assert ierr_f == 0, ("fortran vlapec failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust vlapec failed", nlat, nlon, nt, ityp, ierr_r)
    summarize_diff("vlapec vlap", vlap_f, vlap_r)
    summarize_diff("vlapec wlap", wlap_f, wlap_r)
    summarize_diff("rust(vlapec, fort init) vlap", vlap_rf, vlap_r)
    summarize_diff("rust(vlapec, fort init) wlap", wlap_rf, wlap_r)
    summarize_diff("fort(vlapec, rust init) vlap", vlap_fr, vlap_f)
    summarize_diff("fort(vlapec, rust init) wlap", wlap_fr, wlap_f)


if __name__ == "__main__":
    for ityp in DEFAULT_ITYPES:
        for case in DEFAULT_CASES:
            run_case(*case, ityp=ityp)
