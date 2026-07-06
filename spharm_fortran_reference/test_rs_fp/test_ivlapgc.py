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


def calc_vhagc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhagc, ldwork


def calc_vhsgc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    idz = mmax * (2 * nlat - mmax + 1) // 2
    lzimn = idz * imid
    lvhsgc_init = 2 * (lzz1 + labc) + nlon + 15
    lvhsgc_use = 2 * lzimn + nlon + 15
    lvhsgc = max(lvhsgc_init, lvhsgc_use)
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


def idvw_from_ityp(nlat: int, ityp: int):
    return nlat if ityp <= 2 else (nlat + 1) // 2


def crop_half_sphere(arr, nlat: int, ityp: int):
    arr = np.asarray(arr)
    idvw = idvw_from_ityp(nlat, ityp)
    if idvw == nlat:
        return arr
    if arr.ndim == 2:
        return arr[:idvw, :]
    return arr[:idvw, :, :]


def make_vector_grid(nlat: int, nlon: int, nt: int):
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0, ("gaqd failed", nlat, ierr)
    theta = np.asarray(theta, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        phase = k + 1
        v[:, :, k] = (
            np.sin(theta)[:, None] * np.cos(phase * lon)[None, :]
            + 0.15 * np.cos(2.0 * theta)[:, None] * np.sin((phase + 1) * lon)[None, :]
        )
        w[:, :, k] = (
            np.cos(theta)[:, None] * np.sin(phase * lon)[None, :]
            + 0.10 * np.sin(3.0 * theta)[:, None] * np.cos((phase + 2) * lon)[None, :]
        )
    return v, w


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)

    lvhagc, ldwork_vhagc = calc_vhagc_sizes(nlat, nlon)
    wvhagc, ierr0 = fort_sp.vhagci(nlat, nlon, lvhagc, ldwork_vhagc)
    assert ierr0 == 0, ("vhagci failed", nlat, nlon, ierr0)
    imid = (nlat + 1) // 2
    if ityp <= 2:
        lwork_vhagc = nlat * (4 * nlon * nt + 6 * imid)
    else:
        lwork_vhagc = imid * (4 * nlon * nt + 6 * nlat)
    br, bi, cr, ci, ierr1 = fort_sp.vhagc(v, w, np.asarray(wvhagc, dtype=np.float32), lwork_vhagc, ityp=ityp)
    assert ierr1 == 0, ("vhagc failed", nlat, nlon, nt, ityp, ierr1)

    lvhsgc, ldwork_vhsgc = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr2 = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork_vhsgc)
    assert ierr2 == 0, ("vhsgci failed", nlat, nlon, ierr2)

    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    if ityp < 3:
        lwork_ivlapgc = nlat * (2 * nt * nlon + max(6 * imid, nlon) + 1) + (4 * mn if ityp == 0 else 2 * mn)
    else:
        lwork_ivlapgc = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + (4 * mn + nlat if ityp in (3, 6) else 2 * mn + nlat)

    v_f, w_f, ierr_f = fort_sp.ivlapgc(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        np.asarray(wvhsgc, dtype=np.float32),
        lwork_ivlapgc,
        ityp=ityp,
    )
    v_r, w_r, ierr_r = rust_sp.ivlapgc_ityp(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        ityp,
        np.asarray(wvhsgc, dtype=np.float32),
        lwork_ivlapgc,
    )
    assert ierr_f == 0, ("fortran ivlapgc failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust ivlapgc failed", nlat, nlon, nt, ityp, ierr_r)

    v_f = crop_half_sphere(v_f, nlat, ityp)
    w_f = crop_half_sphere(w_f, nlat, ityp)
    v_r = crop_half_sphere(v_r, nlat, ityp)
    w_r = crop_half_sphere(w_r, nlat, ityp)
    summarize_diff(f"ivlapgc v ityp={ityp}", v_f, v_r)
    summarize_diff(f"ivlapgc w ityp={ityp}", w_f, w_r)


if __name__ == "__main__":
    for case in [
        (4, 4, 1, 0),
        (5, 8, 2, 0),
        (5, 8, 1, 1),
        (5, 8, 1, 2),
        (5, 8, 1, 3),
        (5, 8, 1, 6),
        (73, 144, 1, 0),
    ]:
        run_case(*case)
