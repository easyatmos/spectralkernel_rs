import os
import numpy as np

os.environ["VLAPGC_TRACE"] = "1"
os.environ["VLAPGC_HIGH_ORDER_PULSE"] = "1"
os.environ["VLAPGC_TRACE_FILE"] = "mytest/vlapgc_trace.log"

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


DEFAULT_CASES = [(4, 4, 1), (5, 8, 2), (73, 144, 1)]
DEFAULT_ITYPES = list(range(9))
PULSE_POINTS = [(2, 2), (2, 3), (10, 10), (10, 20), (20, 30), (30, 50), (50, 60), (70, 73)]
HIGH_ORDER_PULSE_POINTS = [(30, 30), (40, 45), (50, 60), (60, 70), (70, 73)]


def summarize_diff(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    ref = np.asarray(b)
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(diff) / denom
    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)
    print("max |diff| :", np.max(np.abs(diff)))
    print("mean|diff| :", np.mean(np.abs(diff)))
    print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
    print("max rel    :", np.max(rel))
    print("mean rel   :", np.mean(rel))
    print("rms rel    :", np.sqrt(np.mean(rel ** 2)))


def summarize_relative_error(name, a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    denom = np.maximum(np.abs(b), eps)
    rel = np.abs(diff) / denom
    ref_norm = np.linalg.norm(b.ravel())
    diff_norm = np.linalg.norm(diff.ravel())
    l2_rel = diff_norm / max(ref_norm, eps)
    print(f"\n[{name} relative error]")
    print("shape:", a.shape, b.shape)
    print("max rel   :", np.max(rel))
    print("mean rel  :", np.mean(rel))
    print("rms rel   :", np.sqrt(np.mean(rel ** 2)))
    print("rms abs   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
    print("L2 rel    :", l2_rel)
    nz = np.abs(b) > eps
    if np.any(nz):
        print("max rel nz:", np.max(np.abs(diff[nz]) / np.maximum(np.abs(b[nz]), eps)))
        print("rms rel nz:", np.sqrt(np.mean((np.abs(diff[nz]) / np.maximum(np.abs(b[nz]), eps)) ** 2)))


def summarize_masked_relative_error(name, a, b, tau=1e-2, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    ref_abs = np.abs(b)
    thresh = tau * np.max(ref_abs)
    mask = ref_abs > thresh
    if not np.any(mask):
        print(f"\n[{name} masked relative error]")
        print("mask empty; skipped")
        return
    rel = np.abs(diff[mask]) / np.maximum(ref_abs[mask], eps)
    print(f"\n[{name} masked relative error]")
    print("shape:", a.shape, b.shape)
    print("tau      :", tau)
    print("mask_nnz :", int(np.count_nonzero(mask)))
    print("mask_max :", float(np.max(ref_abs[mask])))
    print("mask_min :", float(np.min(ref_abs[mask])))
    print("max rel  :", float(np.max(rel)))
    print("mean rel :", float(np.mean(rel)))
    print("rms rel  :", float(np.sqrt(np.mean(rel ** 2))))


def summarize_nonzero_mask(name, arr, mask, tol=0.0):
    arr = np.asarray(arr)
    mask = np.asarray(mask, dtype=bool)
    abs_arr = np.abs(arr)
    nz = abs_arr > tol
    masked_nz = int(np.count_nonzero(nz & mask))
    masked_total = int(np.count_nonzero(mask))
    masked_max = float(np.max(abs_arr[mask])) if masked_total else 0.0
    print(f"[{name}] masked_nnz={masked_nz}/{masked_total} masked_max={masked_max}")


def summarize_layer_diff(name, a, b, axis=0):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    if diff.ndim == 2:
        layers = [diff]
    else:
        layers = [np.take(diff, idx, axis=axis) for idx in range(diff.shape[axis])]
    print(f"\n[{name} layer diff]")
    for idx, layer in enumerate(layers, start=1):
        print(
            f"  layer={idx:02d} max={np.max(layer):.6g} rms={np.sqrt(np.mean(layer ** 2)):.6g} mean={np.mean(layer):.6g}"
        )


def calc_vhsgc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhsgc = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


def calc_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    branch_nosym_all = nlat * (2 * nt * nlon + max(6 * imid, nlon) + 1) + 4 * mn
    branch_sym_all = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 4 * mn + nlat
    return max(branch_nosym_all, branch_sym_all)


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


def make_single_mode_coeffs(nlat: int, nt: int, field: str, m: int, n: int, k: int = 0, value: float = 1.0):
    br = np.zeros((nlat, nlat, nt), dtype=np.float32)
    bi = np.zeros((nlat, nlat, nt), dtype=np.float32)
    cr = np.zeros((nlat, nlat, nt), dtype=np.float32)
    ci = np.zeros((nlat, nlat, nt), dtype=np.float32)
    target = {"br": br, "bi": bi, "cr": cr, "ci": ci}[field]
    target[m - 1, n - 1, k] = np.float32(value)
    return br, bi, cr, ci


def build_vlap_coeffs(br, bi, cr, ci, nlat: int, nlon: int, nt: int, ityp: int):
    mmax = min(nlat, (nlon + 1) // 2)
    brlap = np.zeros_like(br)
    bilap = np.zeros_like(bi)
    crlap = np.zeros_like(cr)
    cilap = np.zeros_like(ci)
    fnn = np.zeros((nlat,), dtype=np.float32)
    for n in range(1, nlat):
        fn = np.float32(n)
        fnn[n] = -(fn * (fn + np.float32(1.0)))

    if ityp in (0, 3, 6):
        for k in range(nt):
            for n in range(1, nlat):
                brlap[0, n, k] = fnn[n] * br[0, n, k]
                bilap[0, n, k] = fnn[n] * bi[0, n, k]
                crlap[0, n, k] = fnn[n] * cr[0, n, k]
                cilap[0, n, k] = fnn[n] * ci[0, n, k]
            for m in range(1, mmax):
                for n in range(m, nlat):
                    brlap[m, n, k] = fnn[n] * br[m, n, k]
                    bilap[m, n, k] = fnn[n] * bi[m, n, k]
                    crlap[m, n, k] = fnn[n] * cr[m, n, k]
                    cilap[m, n, k] = fnn[n] * ci[m, n, k]
    elif ityp in (1, 4, 7):
        for k in range(nt):
            for n in range(1, nlat):
                brlap[0, n, k] = fnn[n] * br[0, n, k]
                bilap[0, n, k] = fnn[n] * bi[0, n, k]
            for m in range(1, mmax):
                for n in range(m, nlat):
                    brlap[m, n, k] = fnn[n] * br[m, n, k]
                    bilap[m, n, k] = fnn[n] * bi[m, n, k]
    else:
        for k in range(nt):
            for n in range(1, nlat):
                crlap[0, n, k] = fnn[n] * cr[0, n, k]
                cilap[0, n, k] = fnn[n] * ci[0, n, k]
            for m in range(1, mmax):
                for n in range(m, nlat):
                    crlap[m, n, k] = fnn[n] * cr[m, n, k]
                    cilap[m, n, k] = fnn[n] * ci[m, n, k]

    return brlap, bilap, crlap, cilap


def coeff_region_masks(nlat: int, nt: int, nlon: int):
    mmax = min(nlat, (nlon + 1) // 2)
    valid = np.zeros((nlat, nlat, nt), dtype=bool)
    invalid = np.zeros_like(valid)
    truncated = np.zeros_like(valid)
    for m in range(nlat):
        for n in range(nlat):
            if m >= mmax:
                truncated[m, n, :] = True
            elif m > n:
                invalid[m, n, :] = True
            else:
                valid[m, n, :] = True
    return valid, invalid, truncated


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0, *, single_mode=None):
    print(f"\n{'=' * 80}\nvlapgc: nlat={nlat}, nlon={nlon}, nt={nt}, ityp={ityp}\n{'=' * 80}")
    lvhsgc, ldwork = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr0 = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork)
    assert ierr0 == 0, ("vhsgci failed", nlat, nlon, ierr0)

    if single_mode is None:
        br, bi, cr, ci = make_coeffs(nlat, nt)
        if ityp in (1, 4, 7):
            cr.fill(0.0)
            ci.fill(0.0)
        if ityp in (2, 5, 8):
            br.fill(0.0)
            bi.fill(0.0)
    else:
        field, m, n, k, value = single_mode
        br, bi, cr, ci = make_single_mode_coeffs(nlat, nt, field, m, n, k, value)

    lwork = calc_lwork(nlat, nlon, nt, ityp)
    wvhsgc = np.asarray(wvhsgc, dtype=np.float32)
    brlap, bilap, crlap, cilap = build_vlap_coeffs(br, bi, cr, ci, nlat, nlon, nt, ityp)
    valid_mask, invalid_mask, truncated_mask = coeff_region_masks(nlat, nt, nlon)
    summarize_nonzero_mask("brlap valid", brlap, valid_mask)
    summarize_nonzero_mask("brlap invalid", brlap, invalid_mask)
    summarize_nonzero_mask("brlap truncated", brlap, truncated_mask)
    summarize_nonzero_mask("bilap valid", bilap, valid_mask)
    summarize_nonzero_mask("bilap invalid", bilap, invalid_mask)
    summarize_nonzero_mask("bilap truncated", bilap, truncated_mask)
    summarize_nonzero_mask("crlap valid", crlap, valid_mask)
    summarize_nonzero_mask("crlap invalid", crlap, invalid_mask)
    summarize_nonzero_mask("crlap truncated", crlap, truncated_mask)
    summarize_nonzero_mask("cilap valid", cilap, valid_mask)
    summarize_nonzero_mask("cilap invalid", cilap, invalid_mask)
    summarize_nonzero_mask("cilap truncated", cilap, truncated_mask)

    if ityp == 0:
        vh_v_f, vh_w_f, vh_ierr_f = fort_sp.vhsgc(nlon, brlap, bilap, crlap, cilap, wvhsgc, lwork)
        vh_v_r, vh_w_r, vh_ierr_r = rust_sp.vhsgc(brlap, bilap, crlap, cilap, wvhsgc, lwork)
    else:
        vh_v_f, vh_w_f, vh_ierr_f = fort_sp.vhsgc(nlon, brlap, bilap, crlap, cilap, wvhsgc, lwork, ityp=ityp)
        vh_v_r, vh_w_r, vh_ierr_r = rust_sp.vhsgc_ityp(brlap, bilap, crlap, cilap, ityp, wvhsgc, lwork)

    assert vh_ierr_f == 0, ("fortran low-level vhsgc failed", nlat, nlon, nt, ityp, vh_ierr_f)
    assert vh_ierr_r == 0, ("rust low-level vhsgc failed", nlat, nlon, nt, ityp, vh_ierr_r)
    summarize_diff("vlapgc low-level vhsgc v", vh_v_f, vh_v_r)
    summarize_diff("vlapgc low-level vhsgc w", vh_w_f, vh_w_r)
    summarize_relative_error("vlapgc low-level vhsgc v", vh_v_f, vh_v_r)
    summarize_relative_error("vlapgc low-level vhsgc w", vh_w_f, vh_w_r)
    summarize_masked_relative_error("vlapgc low-level vhsgc v", vh_v_f, vh_v_r)
    summarize_masked_relative_error("vlapgc low-level vhsgc w", vh_w_f, vh_w_r)
    summarize_layer_diff("vlapgc low-level vhsgc v", vh_v_f, vh_v_r, axis=2 if nt > 1 else 0)
    summarize_layer_diff("vlapgc low-level vhsgc w", vh_w_f, vh_w_r, axis=2 if nt > 1 else 0)

    if ityp == 0:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapgc(nlon, br, bi, cr, ci, wvhsgc, lwork)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapgc(nlon, br, bi, cr, ci, wvhsgc, lwork)
    else:
        vlap_f, wlap_f, ierr_f = fort_sp.vlapgc(nlon, br, bi, cr, ci, wvhsgc, lwork, ityp=ityp)
        vlap_r, wlap_r, ierr_r = rust_sp.vlapgc_ityp(nlon, br, bi, cr, ci, ityp, wvhsgc, lwork)

    assert ierr_f == 0, ("fortran vlapgc failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust vlapgc failed", nlat, nlon, nt, ityp, ierr_r)
    summarize_diff("vlapgc vlap", vlap_f, vlap_r)
    summarize_diff("vlapgc wlap", wlap_f, wlap_r)
    summarize_relative_error("vlapgc vlap", vlap_f, vlap_r)
    summarize_relative_error("vlapgc wlap", wlap_f, wlap_r)
    summarize_masked_relative_error("vlapgc vlap", vlap_f, vlap_r)
    summarize_masked_relative_error("vlapgc wlap", wlap_f, wlap_r)
    summarize_layer_diff("vlapgc vlap", vlap_f, vlap_r, axis=2 if nt > 1 else 0)
    summarize_layer_diff("vlapgc wlap", wlap_f, wlap_r, axis=2 if nt > 1 else 0)


def run_single_mode_suite():
    for ityp in (1, 2, 4, 5):
        for m, n in PULSE_POINTS:
            if m > n:
                continue
            if m > 73 or n > 73:
                continue
            for field in ("br", "bi", "cr", "ci"):
                print(f"\n--- pulse field={field} m={m} n={n} ityp={ityp} ---")
                run_case(73, 144, 1, ityp=ityp, single_mode=(field, m, n, 0, 1.0))


def run_high_order_single_mode_suite():
    for ityp in (1, 2, 4, 5):
        for m, n in HIGH_ORDER_PULSE_POINTS:
            if m > n:
                continue
            if m > 73 or n > 73:
                continue
            for field in ("br", "bi", "cr", "ci"):
                print(f"\n--- high-order pulse field={field} m={m} n={n} ityp={ityp} ---")
                run_case(73, 144, 1, ityp=ityp, single_mode=(field, m, n, 0, 1.0))


if __name__ == "__main__":
    if os.environ.get("VLAPGC_HIGH_ORDER_PULSE", "0") not in ("0", ""):
        run_high_order_single_mode_suite()
    elif os.environ.get("VLAPGC_PULSE", "0") not in ("0", ""):
        run_single_mode_suite()
    else:
        cases = DEFAULT_CASES
        itypes = DEFAULT_ITYPES if os.environ.get("VLAPGC_ALL_ITYP", "1") not in ("0", "") else [0, 1, 2, 4, 5]
        for ityp in itypes:
            for case in cases:
                run_case(*case, ityp=ityp)
