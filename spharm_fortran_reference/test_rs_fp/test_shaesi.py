import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


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
    print("A value    :", a[idx])
    print("B value    :", b[idx])
    print("diff       :", diff[idx])


def calc_shaesi_sizes(nlat: int, nlon: int):
    if nlon % 2 == 0:
        l1 = min(nlat, (nlon + 2) // 2)
    else:
        l1 = min(nlat, (nlon + 1) // 2)

    if nlat % 2 == 0:
        l2 = nlat // 2
    else:
        l2 = (nlat + 1) // 2

    lshaes = (l1 * l2 * (2 * nlat - l1 + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * l2 + (3 * ((l1 - 2) * (2 * nlat - l1 - 1))) // 2
    ldwork = nlat + 1
    return lshaes, lwork, ldwork


def shaesi_from_backend(sp, nlat: int, nlon: int):
    lshaes, lwork, ldwork = calc_shaesi_sizes(nlat, nlon)
    return sp.shaesi(nlat, nlon, lshaes, lwork, ldwork)


def calc_lzimn(nlat: int, nlon: int) -> int:
    mmax = min(nlat, nlon // 2 + 1)
    imid = (nlat + 1) // 2
    return (imid * mmax * (2 * nlat - mmax + 1)) // 2


def check_shaesi(cases=None):
    print("\n" + "=" * 80)
    print("SHAESI COMPARISON")
    print("=" * 80)

    if cases is None:
        cases = [
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 16),
            (16, 32),
        ]

    for nlat, nlon in cases:
        print(f"\n-------------------- nlat = {nlat}, nlon = {nlon} --------------------")
        lshaes, lwork, ldwork = calc_shaesi_sizes(nlat, nlon)
        print("lshaes =", lshaes)
        print("lwork  =", lwork)
        print("ldwork =", ldwork)

        w_f, ierr_f = shaesi_from_backend(fort_sp, nlat, nlon)
        w_r, ierr_r = shaesi_from_backend(rust_sp, nlat, nlon)

        print("ierror fortran =", ierr_f)
        print("ierror rust    =", ierr_r)

        summarize_diff("shaesi wshaes", w_f, w_r)

        lzimn = calc_lzimn(nlat, nlon)
        summarize_diff("shaesi prefix(lzimn)", np.asarray(w_f)[:lzimn], np.asarray(w_r)[:lzimn])
        summarize_diff("shaesi suffix(hrffti)", np.asarray(w_f)[lzimn:], np.asarray(w_r)[lzimn:])

        print("fortran finite =", np.isfinite(w_f).all())
        print("rust finite    =", np.isfinite(w_r).all())


def check_shaesi_error_paths():
    print("\n" + "=" * 80)
    print("SHAESI ERROR PATH CHECK")
    print("=" * 80)

    cases = [
        (2, 4, 10, 10, 3),
        (3, 3, 10, 10, 4),
        (4, 4, 1, 10, 5),
        (4, 4, 100, 1, 5),
        (4, 4, 100, 100, 1),
    ]

    for nlat, nlon, lshaes, lwork, ldwork in cases:
        print(
            f"\ncase: nlat={nlat}, nlon={nlon}, lshaes={lshaes}, lwork={lwork}, ldwork={ldwork}"
        )
        w_f, ierr_f = fort_sp.shaesi(nlat, nlon, lshaes, lwork, ldwork)
        w_r, ierr_r = rust_sp.shaesi(nlat, nlon, lshaes, lwork, ldwork)
        print("ierror fortran =", ierr_f)
        print("ierror rust    =", ierr_r)
        print("len(fortran)   =", len(np.asarray(w_f).reshape(-1)))
        print("len(rust)      =", len(np.asarray(w_r).reshape(-1)))


if __name__ == "__main__":
    check_shaesi()
    check_shaesi_error_paths()
