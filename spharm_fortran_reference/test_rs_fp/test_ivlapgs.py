import numpy as np

from spharm_fortran_reference.pyspharm import _spherepack as fort_sp
import spectralkernel_rs as rust_sp


def summarize_diff(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("dtype:", a.dtype, b.dtype)
    print("max |diff| :", np.max(np.abs(diff)))
    print("mean|diff| :", np.mean(np.abs(diff)))
    print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))


def calc_vhags_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    idz = mmax * (2 * nlat - mmax + 1) // 2
    lzimn = idz * imid
    lvhags_init = 2 * (imid * lmn) + nlon + 15
    lvhags_use = 2 * lzimn + nlon + 15
    lvhags = max(lvhags_init, lvhags_use)
    ldwork = (nlat * (3 * nlat + 9) + 2) // 2
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
            + 0.12 * np.cos(2.0 * theta)[:, None] * np.sin((phase + 1) * lon)[None, :]
        )
        w[:, :, k] = (
            np.cos(theta)[:, None] * np.sin((phase + 1) * lon)[None, :]
            + 0.08 * np.sin(3.0 * theta)[:, None] * np.cos((phase + 2) * lon)[None, :]
        )
    return v, w


def calc_vhags_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_ivlapgs_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 1) // 2)
    if ityp <= 2:
        return (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * l1 + 1)
    return (2 * nt + 1) * imid * nlon + nlat * (4 * nt * l1 + 1)


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)

    lvhags, ldwork_vhags = calc_vhags_sizes(nlat, nlon)
    wvhags, ierr0 = fort_sp.vhagsi(nlat, nlon, lvhags, ldwork_vhags)
    assert ierr0 == 0, ("vhagsi failed", nlat, nlon, ierr0)
    lwork_vhags = calc_vhags_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhags(v, w, np.asarray(wvhags, dtype=np.float32), lwork_vhags, ityp=ityp)
    assert ierr1 == 0, ("vhags failed", nlat, nlon, nt, ityp, ierr1)

    lvhsgs, ldwork_vhsgs = calc_vhsgs_sizes(nlat, nlon)
    wvhsgs, ierr2 = fort_sp.vhsgsi(nlat, nlon, lvhsgs, ldwork_vhsgs)
    assert ierr2 == 0, ("vhsgsi failed", nlat, nlon, ierr2)

    lwork_ivlapgs = calc_ivlapgs_lwork(nlat, nlon, nt, ityp)
    v_f, w_f, ierr_f = fort_sp.ivlapgs(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        np.asarray(wvhsgs, dtype=np.float32),
        lwork_ivlapgs,
        ityp=ityp,
    )
    v_r, w_r, ierr_r = rust_sp.ivlapgs_ityp(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        ityp,
        np.asarray(wvhsgs, dtype=np.float32),
        lwork_ivlapgs,
    )
    assert ierr_f == 0, ("fortran ivlapgs failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust ivlapgs failed", nlat, nlon, nt, ityp, ierr_r)

    v_f = crop_half_sphere(v_f, nlat, ityp)
    w_f = crop_half_sphere(w_f, nlat, ityp)
    v_r = crop_half_sphere(v_r, nlat, ityp)
    w_r = crop_half_sphere(w_r, nlat, ityp)
    summarize_diff(f"ivlapgs v ityp={ityp}", v_f, v_r)
    summarize_diff(f"ivlapgs w ityp={ityp}", w_f, w_r)


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
