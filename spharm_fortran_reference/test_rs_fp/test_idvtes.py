import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


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


def calc_shaes_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshaes = (l1 * l2 * (2 * nlat - l1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * l2 + 3 * ((l1 - 2) * (2 * nlat - l1 - 1)) // 2
    ldwork = nlat + 1
    return lshaes, lwork, ldwork


def calc_vhses_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhses = l1 * l2 * (2 * nlat - l1 + 1) + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lvhses, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    divg = np.zeros((nlat, nlon, nt), dtype=np.float32)
    vort = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        divg[:, :, k] = np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :] + 0.1 * np.sin(theta)[:, None]
        vort[:, :, k] = np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :] + 0.05 * np.cos(theta)[:, None]
    return divg, vort


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    divg, vort = make_scalar_grid(nlat, nlon, nt)
    lshaes, lwork_shaesi, ldwork_shaesi = calc_shaes_sizes(nlat, nlon)
    wshaes, ierr0 = fort_sp.shaesi(nlat, nlon, lshaes, lwork_shaesi, ldwork_shaesi)
    assert ierr0 == 0, ("shaesi failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shaes = nlat * (nt * nlon + max(3 * l2, nlon))
        ad, bd, ierr1 = fort_sp.shaes(divg, np.asarray(wshaes, dtype=np.float32), lwork_shaes)
        av, bv, ierr2 = fort_sp.shaes(vort, np.asarray(wshaes, dtype=np.float32), lwork_shaes)
    else:
        lwork_shaes = l2 * (nt * nlon + max(3 * nlat, nlon))
        ad, bd, ierr1 = fort_sp.shaes(divg, np.asarray(wshaes, dtype=np.float32), lwork_shaes, isym=isym)
        av, bv, ierr2 = fort_sp.shaes(vort, np.asarray(wshaes, dtype=np.float32), lwork_shaes, isym=isym)
    assert ierr1 == 0 and ierr2 == 0

    lvhses, ldwork_vhsesi = calc_vhses_sizes(nlat, nlon)
    l2 = (nlat + 1) // 2
    init_lwork_vhsesi = 3 * max(min(nlat, (nlon + 1) // 2) - 2, 0) * (2 * nlat - min(nlat, (nlon + 1) // 2) - 1) // 2 + 5 * l2 * nlat
    wvhses, ierr3 = fort_sp.vhsesi(nlat, nlon, lvhses, init_lwork_vhsesi, ldwork_vhsesi)
    assert ierr3 == 0

    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    imid = (nlat + 1) // 2
    lwork_eq0 = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 4 * mn + nlat
    lwork_ne0 = nlat * (2 * nt * nlon + max(6 * imid, nlon)) + 4 * mn + nlat
    lwork = max(lwork_eq0, lwork_ne0)

    if isym == 0:
        v_f, w_f, pd_f, pv_f, ierr_f = fort_sp.idvtes(nlon, ad, bd, av, bv, np.asarray(wvhses, dtype=np.float32), lwork)
        v_r, w_r, pd_r, pv_r, ierr_r = rust_sp.idvtes(nlon, ad, bd, av, bv, np.asarray(wvhses, dtype=np.float32), lwork)
    else:
        v_f, w_f, pd_f, pv_f, ierr_f = fort_sp.idvtes(nlon, ad, bd, av, bv, np.asarray(wvhses, dtype=np.float32), lwork, isym=isym)
        v_r, w_r, pd_r, pv_r, ierr_r = rust_sp.idvtes_isym(nlon, ad, bd, av, bv, isym, np.asarray(wvhses, dtype=np.float32), lwork)

    assert ierr_f == 0, ("fortran idvtes failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust idvtes failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("idvtes v", v_f, v_r)
    summarize_diff("idvtes w", w_f, w_r)
    summarize_diff("idvtes pertbd", pd_f, pd_r)
    summarize_diff("idvtes pertbv", pv_f, pv_r)


if __name__ == "__main__":
    for case in [(4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
