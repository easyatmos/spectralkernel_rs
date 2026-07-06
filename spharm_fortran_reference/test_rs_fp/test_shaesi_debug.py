import numpy as np

import spectralkernel_rs as rust_sp
from spharm_fortran_reference.pyspharm import _spherepack as fort_sp


def calc_lzimn(nlat: int, nlon: int) -> int:
    mmax = min(nlat, nlon // 2 + 1)
    imid = (nlat + 1) // 2
    return (imid * mmax * (2 * nlat - mmax + 1)) // 2


def calc_zfinit_layout(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lim = nlat * imid
    mmax = min(nlat, nlon // 2 + 1)
    labc = ((max(mmax - 2, 0)) * (2 * nlat - mmax - 1)) // 2
    return imid, lim, mmax, labc


def calc_idz(nlat: int, nlon: int) -> int:
    mmax = min(nlat, nlon // 2 + 1)
    return (mmax * (2 * nlat - mmax + 1)) // 2


def summarize_diff(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    print(f"\n[{name}]")
    print("shape:", a.shape, b.shape)
    print("max |diff| :", np.max(np.abs(diff)))
    print("mean|diff| :", np.mean(np.abs(diff)))
    print("rms diff   :", np.sqrt(np.mean(np.abs(diff) ** 2)))
    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print("worst index:", idx)
    print("A value    :", a[idx])
    print("B value    :", b[idx])
    print("diff       :", diff[idx])


def check_shaesi_debug(cases=None):
    print("\n" + "=" * 80)
    print("SHAESI DEBUG SPLIT")
    print("=" * 80)

    if cases is None:
        cases = [
            (3, 4),
            (4, 4),
            (5, 8),
        ]

    for nlat, nlon in cases:
        print(f"\n-------------------- nlat = {nlat}, nlon = {nlon} --------------------")
        lshaes = calc_lzimn(nlat, nlon) + nlon + 15
        lwork = 5 * nlat * ((nlat + 1) // 2) + (3 * ((min(nlat, nlon // 2 + 1) - 2) * (2 * nlat - min(nlat, nlon // 2 + 1) - 1))) // 2
        ldwork = nlat + 1

        wf, _ = fort_sp.shaesi(nlat, nlon, lshaes, lwork, ldwork)
        wr, _ = rust_sp.shaesi(nlat, nlon, lshaes, lwork, ldwork)
        summarize_diff("shaesi prefix", np.asarray(wf)[:calc_lzimn(nlat, nlon)], np.asarray(wr)[:calc_lzimn(nlat, nlon)])
        summarize_diff("shaesi suffix", np.asarray(wf)[calc_lzimn(nlat, nlon):], np.asarray(wr)[calc_lzimn(nlat, nlon):])

        wzfin, abc = rust_sp.zfinit_debug(nlat, nlon)
        sea1 = rust_sp.sea1_debug(nlat, nlon)
        imid, lim, mmax, labc = calc_zfinit_layout(nlat, nlon)

        print("imid =", imid, "lim =", lim, "mmax =", mmax, "labc =", labc)
        print("len(wzfin) =", len(np.asarray(wzfin)))
        print("len(abc)   =", len(np.asarray(abc)))
        print("len(sea1)  =", len(np.asarray(sea1)))
        print("sea1 finite=", np.isfinite(sea1).all())
        print("fortran prefix head =", np.asarray(wf)[:min(12, calc_lzimn(nlat, nlon))])
        print("rust prefix head    =", np.asarray(wr)[:min(12, calc_lzimn(nlat, nlon))])
        print("rust sea1 head      =", np.asarray(sea1)[:min(12, len(np.asarray(sea1)))])
        print("rust wzfin head     =", np.asarray(wzfin)[:min(12, len(np.asarray(wzfin)))])

        prefix_f = np.asarray(wf)[:calc_lzimn(nlat, nlon)]
        prefix_r = np.asarray(wr)[:calc_lzimn(nlat, nlon)]
        prefix_diff = prefix_f - prefix_r
        worst = int(np.argmax(np.abs(prefix_diff)))
        idz = calc_idz(nlat, nlon)
        i_fortran = worst // idz + 1
        mn_fortran = worst % idz + 1
        print("prefix worst linear idx =", worst)
        print("prefix worst (mn,i)     =", (mn_fortran, i_fortran))


if __name__ == "__main__":
    check_shaesi_debug()
