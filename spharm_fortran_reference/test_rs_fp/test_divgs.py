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
    ldwork = nlat * (nlat + 4)
    return lshsgs, ldwork


def make_vec_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    v = np.zeros((nlat, nlon, nt), dtype=np.float32)
    w = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        v[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
        )
        w[:, :, k] = (
            np.sin(theta)[:, None] * np.sin((k + 1) * lon)[None, :]
            + 0.15 * np.cos(theta)[:, None] * np.cos((k + 3) * lon)[None, :]
        )
    return v, w


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    v, w = make_vec_grid(nlat, nlon, nt)

    lvhags, ldwork_vhagsi = calc_vhags_sizes(nlat, nlon)
    wvhags, ierr0 = fort_sp.vhagsi(nlat, nlon, lvhags, ldwork_vhagsi)
    assert ierr0 == 0, ("vhagsi failed", nlat, nlon, ierr0)

    lwork_vhags = (2 * nt + 1) * nlat * nlon
    br, bi, _cr, _ci, ierr1 = fort_sp.vhags(v, w, np.asarray(wvhags, dtype=np.float32), lwork_vhags)
    assert ierr1 == 0, ("vhags failed", nlat, nlon, nt, ierr1)

    lshsgs, ldwork_shsgsi = calc_shsgs_sizes(nlat, nlon)
    init_lwork_shsgsi = 4 * nlat * (nlat + 2) + 2
    wshsgs, ierr2 = fort_sp.shsgsi(nlat, nlon, lshsgs, init_lwork_shsgsi, ldwork_shsgsi)
    assert ierr2 == 0, ("shsgsi failed", nlat, nlon, ierr2)

    l2 = (nlat + 1) // 2
    l1 = min(nlat, (nlon + 2) // 2)
    if isym == 0:
        lwork_divgs = nlat * ((nt + 1) * nlon + 2 * nt * l1 + 1)
    else:
        lwork_divgs = (nt + 1) * l2 * nlon + nlat * (2 * nt * l1 + 1)

    divg_f, ierr_f = fort_sp.divgs(nlon, br, bi, np.asarray(wshsgs, dtype=np.float32), lwork_divgs, isym=isym)
    divg_r, ierr_r = rust_sp.divgs(nlon, br, bi, np.asarray(wshsgs, dtype=np.float32), lwork_divgs, isym=isym)

    assert ierr_f == 0, ("fortran divgs failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust divgs failed", nlat, nlon, nt, isym, ierr_r)

    print(f"\n{'=' * 80}\ndivgs: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    summarize_diff("divgs divg: fortran vs rust", divg_f, divg_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
