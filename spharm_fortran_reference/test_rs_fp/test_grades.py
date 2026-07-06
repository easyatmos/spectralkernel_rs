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
    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("lhs        :", a[idx])
    print("rhs        :", b[idx])
    print("diff       :", diff[idx])


def calc_shaes_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 2) // 2)
    n2 = (nlat + 1) // 2
    lshaes = (n1 * n2 * (2 * nlat - n1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * n2 + 3 * ((n1 - 2) * (2 * nlat - n1 - 1)) // 2
    ldwork = nlat + 1
    return lshaes, lwork, ldwork


def calc_vhses_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    mmax = min(nlat, (nlon + 1) // 2)
    idz = (mmax * (2 * nlat - mmax + 1)) // 2
    lzimn = idz * imid
    lvhses = 2 * lzimn + nlon + 15
    l1 = min(nlat, nlon // 2) if nlon % 2 == 0 else min(nlat, (nlon + 1) // 2)
    labc = 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) // 2
    lwork = 5 * imid * nlat + labc
    ldwork = 2 * (nlat + 1)
    return lvhses, lwork, ldwork


def idvw_from_isym(nlat: int, isym: int):
    if isym == 0:
        return nlat
    return (nlat + 1) // 2


def crop_half_sphere(arr, nlat: int, isym: int):
    arr = np.asarray(arr)
    if isym == 0:
        return arr
    idvw = idvw_from_isym(nlat, isym)
    if arr.ndim == 2:
        return arr[:idvw, :]
    if arr.ndim == 3:
        return arr[:idvw, :, :]
    return arr


def make_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    out = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        out[:, :, k] = (
            np.cos((k + 1) * lon)[None, :]
            + np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.2 * (k + 1) * np.cos(theta)[:, None] ** 2
        )
    return out


def apply_isym_input(g, isym: int):
    g = np.asarray(g, dtype=np.float32)
    nlat = g.shape[0]
    if isym == 0:
        return g

    out = np.array(g, copy=True)
    for i in range(nlat):
        mi = nlat - 1 - i
        if isym == 1:
            out[mi, ...] = -out[i, ...]
        elif isym == 2:
            out[mi, ...] = out[i, ...]
    return out


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    g = apply_isym_input(make_grid(nlat, nlon, nt), isym)

    lshaes, lwork_shaesi, ldwork_shaesi = calc_shaes_sizes(nlat, nlon)
    wshaes, ierr0 = fort_sp.shaesi(nlat, nlon, lshaes, lwork_shaesi, ldwork_shaesi)
    assert ierr0 == 0, ("shaesi failed", nlat, nlon, ierr0)

    lwork_shaes = (nt + 1) * nlat * nlon
    a, b, ierr1 = fort_sp.shaes(g, np.asarray(wshaes, dtype=np.float32), lwork_shaes)
    assert ierr1 == 0, ("shaes failed", nlat, nlon, nt, isym, ierr1)

    lvhses, lwork_vhsesi, ldwork_vhsesi = calc_vhses_sizes(nlat, nlon)
    wvhses_f, ierr2 = fort_sp.vhsesi(nlat, nlon, lvhses, lwork_vhsesi, ldwork_vhsesi)
    assert ierr2 == 0, ("vhsesi failed", nlat, nlon, ierr2)

    lwork_grades = (2 * nt + 1) * idvw_from_isym(nlat, isym) * nlon + nlat * (2 * nt * min(nlat, (nlon + 1) // 2) + 1)

    if isym == 0:
        v_r, w_r, ierr_r = rust_sp.grades(a, b, np.asarray(wvhses_f, dtype=np.float32), lwork_grades)
        v_f, w_f, ierr_f = fort_sp.grades(nlon, a, b, np.asarray(wvhses_f, dtype=np.float32), lwork_grades)
    else:
        v_r, w_r, ierr_r = rust_sp.grades_isym(a, b, isym, np.asarray(wvhses_f, dtype=np.float32), lwork_grades)
        v_f, w_f, ierr_f = fort_sp.grades(nlon, a, b, np.asarray(wvhses_f, dtype=np.float32), lwork_grades, isym=isym)

    assert ierr_r == 0, ("rust grades failed", nlat, nlon, nt, isym, ierr_r)
    assert ierr_f == 0, ("fortran grades failed", nlat, nlon, nt, isym, ierr_f)

    v_f = crop_half_sphere(v_f, nlat, isym)
    w_f = crop_half_sphere(w_f, nlat, isym)

    print(f"\n{'=' * 80}\ngrades: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    summarize_diff("grades v: fortran vs rust", v_f, v_r)
    summarize_diff("grades w: fortran vs rust", w_f, w_r)


if __name__ == "__main__":
    for isym in (0, 1, 2):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_case(*case, isym=isym)
