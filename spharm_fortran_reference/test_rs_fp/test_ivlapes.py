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


def calc_vhaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    n2 = (nlat + 1) // 2
    lvhaes = n1 * n2 * (2 * nlat - n1 + 1) + nlon + 15
    lwork = 3 * max(n1 - 2, 0) * (2 * nlat - n1 - 1) // 2 + 5 * n2 * nlat
    ldwork = 2 * (nlat + 1)
    return lvhaes, lwork, ldwork


def calc_vhses_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    idz = mmax * (2 * nlat - mmax + 1) // 2
    lzimn = idz * imid
    lvhses = 2 * lzimn + nlon + 15
    labc = 3 * (max(mmax - 2, 0) * (nlat + nlat - mmax - 1)) // 2
    lwork = 5 * nlat * imid + labc
    ldwork = 2 * nlat + 2
    return lvhses, lwork, ldwork


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
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
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


def calc_vhaes_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_ivlapes_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 1) // 2)
    if ityp <= 2:
        return (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * l1 + 1)
    return (2 * nt + 1) * imid * nlon + nlat * (4 * nt * l1 + 1)


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)

    lvhaes, init_lwork_vhaes, ldwork_vhaes = calc_vhaes_sizes(nlat, nlon)
    wvhaes, ierr0 = fort_sp.vhaesi(nlat, nlon, lvhaes, init_lwork_vhaes, ldwork_vhaes)
    assert ierr0 == 0, ("vhaesi failed", nlat, nlon, ierr0)
    lwork_vhaes = calc_vhaes_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhaes(v, w, np.asarray(wvhaes, dtype=np.float32), lwork_vhaes, ityp=ityp)
    assert ierr1 == 0, ("vhaes failed", nlat, nlon, nt, ityp, ierr1)

    lvhses, lwork_vhsesi, ldwork_vhsesi = calc_vhses_sizes(nlat, nlon)
    wvhses, ierr2 = fort_sp.vhsesi(nlat, nlon, lvhses, lwork_vhsesi, ldwork_vhsesi)
    assert ierr2 == 0, ("vhsesi failed", nlat, nlon, ierr2)

    lwork_ivlapes = calc_ivlapes_lwork(nlat, nlon, nt, ityp)
    v_f, w_f, ierr_f = fort_sp.ivlapes(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        np.asarray(wvhses, dtype=np.float32),
        lwork_ivlapes,
        ityp=ityp,
    )
    v_r, w_r, ierr_r = rust_sp.ivlapes_ityp(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        ityp,
        np.asarray(wvhses, dtype=np.float32),
        lwork_ivlapes,
    )
    assert ierr_f == 0, ("fortran ivlapes failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust ivlapes failed", nlat, nlon, nt, ityp, ierr_r)

    v_f = crop_half_sphere(v_f, nlat, ityp)
    w_f = crop_half_sphere(w_f, nlat, ityp)
    v_r = crop_half_sphere(v_r, nlat, ityp)
    w_r = crop_half_sphere(w_r, nlat, ityp)
    summarize_diff(f"ivlapes v ityp={ityp}", v_f, v_r)
    summarize_diff(f"ivlapes w ityp={ityp}", w_f, w_r)


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
