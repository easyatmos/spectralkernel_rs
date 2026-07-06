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


def calc_vhaec_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhaec = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhaec, ldwork


def calc_vhsec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhsec = 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhsec, ldwork


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


def calc_vhaec_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    if ityp <= 2:
        return nlat * (4 * nlon * nt + 6 * imid)
    return imid * (4 * nlon * nt + 6 * nlat)


def calc_ivlapec_lwork(nlat: int, nlon: int, nt: int, ityp: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    if ityp < 3:
        if ityp == 0:
            return nlat * (2 * nt * nlon + max(6 * imid, nlon) + 1) + 4 * mn
        return nlat * (2 * nt * nlon + max(6 * imid, nlon) + 1) + 2 * mn
    if ityp in (3, 6):
        return imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 4 * mn + nlat
    return imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 2 * mn + nlat


def run_case(nlat: int, nlon: int, nt: int, ityp: int = 0):
    v, w = make_vector_grid(nlat, nlon, nt)

    lvhaec, ldwork_vhaec = calc_vhaec_sizes(nlat, nlon)
    wvhaec, ierr0 = fort_sp.vhaeci(nlat, nlon, lvhaec, ldwork_vhaec)
    assert ierr0 == 0, ("vhaeci failed", nlat, nlon, ierr0)
    lwork_vhaec = calc_vhaec_lwork(nlat, nlon, nt, ityp)
    br, bi, cr, ci, ierr1 = fort_sp.vhaec(v, w, np.asarray(wvhaec, dtype=np.float32), lwork_vhaec, ityp=ityp)
    assert ierr1 == 0, ("vhaec failed", nlat, nlon, nt, ityp, ierr1)

    lvhsec, ldwork_vhsec = calc_vhsec_sizes(nlat, nlon)
    wvhsec, ierr2 = fort_sp.vhseci(nlat, nlon, lvhsec, ldwork_vhsec)
    assert ierr2 == 0, ("vhseci failed", nlat, nlon, ierr2)

    lwork_ivlapec = calc_ivlapec_lwork(nlat, nlon, nt, ityp)
    v_f, w_f, ierr_f = fort_sp.ivlapec(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        np.asarray(wvhsec, dtype=np.float32),
        lwork_ivlapec,
        ityp=ityp,
    )
    v_r, w_r, ierr_r = rust_sp.ivlapec_ityp(
        nlon,
        np.asarray(br, dtype=np.float32),
        np.asarray(bi, dtype=np.float32),
        np.asarray(cr, dtype=np.float32),
        np.asarray(ci, dtype=np.float32),
        ityp,
        np.asarray(wvhsec, dtype=np.float32),
        lwork_ivlapec,
    )
    assert ierr_f == 0, ("fortran ivlapec failed", nlat, nlon, nt, ityp, ierr_f)
    assert ierr_r == 0, ("rust ivlapec failed", nlat, nlon, nt, ityp, ierr_r)

    v_f = crop_half_sphere(v_f, nlat, ityp)
    w_f = crop_half_sphere(w_f, nlat, ityp)
    v_r = crop_half_sphere(v_r, nlat, ityp)
    w_r = crop_half_sphere(w_r, nlat, ityp)
    summarize_diff(f"ivlapec v ityp={ityp}", v_f, v_r)
    summarize_diff(f"ivlapec w ityp={ityp}", w_f, w_r)


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
