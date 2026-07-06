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


def calc_vhags_sizes(nlat: int, nlon: int):
    lvhags = (nlat + 1) * (nlat + 1) * nlat // 2 + nlon + 15
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhags, ldwork


def calc_shsgs_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgs = nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) // 2 + nlon + 15
    lwork = 4 * nlat * (nlat + 2) + 2
    ldwork = nlat * (nlat + 4)
    return lshsgs, lwork, ldwork


def make_vec_grid(nlat: int, nlon: int, nt: int):
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0, ("gaqd failed", nlat, ierr)
    theta = np.asarray(theta, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.15 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
        )
        w[:, :, k] = (
            np.sin(theta)[:, None] * np.sin((k + 1) * lon)[None, :]
            + 0.1 * np.cos(theta)[:, None] * np.cos((k + 3) * lon)[None, :]
        )
    return v, w


def run_case(nlat: int, nlon: int, nt: int):
    v, w = make_vec_grid(nlat, nlon, nt)

    lvhags, ldwork_vhagsi = calc_vhags_sizes(nlat, nlon)
    wvhags, ierr0 = fort_sp.vhagsi(nlat, nlon, lvhags, ldwork_vhagsi)
    assert ierr0 == 0, ("vhagsi failed", nlat, nlon, ierr0)

    lwork_vhags = (2 * nt + 1) * nlat * nlon
    br, bi, _cr, _ci, ierr1 = fort_sp.vhags(v, w, np.asarray(wvhags, dtype=np.float32), lwork_vhags)
    assert ierr1 == 0, ("vhags failed", nlat, nlon, nt, ierr1)

    lshsgs, lwork_shsgsi, ldwork_shsgsi = calc_shsgs_sizes(nlat, nlon)
    wshsgs, ierr2 = fort_sp.shsgsi(nlat, nlon, lshsgs, lwork_shsgsi, ldwork_shsgsi)
    assert ierr2 == 0, ("shsgsi failed", nlat, nlon, ierr2)

    lwork_igradgs = nlat * ((nt + 1) * nlon + 2 * nt * min(nlat, (nlon + 2) // 2) + 1)
    sf_f, ierr_f = fort_sp.igradgs(nlon, br, bi, np.asarray(wshsgs, dtype=np.float32), lwork_igradgs)
    sf_r, ierr_r = rust_sp.igradgs(br, bi, np.asarray(wshsgs, dtype=np.float32), lwork_igradgs)

    assert ierr_f == 0, ("fortran igradgs failed", nlat, nlon, nt, ierr_f)
    assert ierr_r == 0, ("rust igradgs failed", nlat, nlon, nt, ierr_r)

    print(f"\n{'=' * 80}\nigradgs: nlat={nlat}, nlon={nlon}, nt={nt}\n{'=' * 80}")
    summarize_diff("igradgs sf: fortran vs rust", sf_f, sf_r)


if __name__ == "__main__":
    for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
        run_case(*case)
