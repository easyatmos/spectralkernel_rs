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


def calc_shags_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshags = nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) // 2 + nlon + 15
    lwork = 4 * nlat * (nlat + 2) + 2
    ldwork = nlat * (nlat + 4)
    return lshags, lwork, ldwork


def calc_vhsgs_sizes(nlat: int, nlon: int):
    n1 = min(nlat, (nlon + 1) // 2)
    imid = (nlat + 1) // 2
    lmn = nlat * (nlat + 1) // 2
    lvhsgs_synth = n1 * imid * (2 * nlat - n1 + 1) // 2 * 2 + nlon + 15
    lvhsgs_init = 2 * imid * lmn + nlon + 15
    lvhsgs = max(lvhsgs_synth, lvhsgs_init)
    ldwork = (3 * nlat * (nlat + 3) + 2) // 2
    return lvhsgs, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    divg = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        divg[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.05
        )
    return divg


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    divg = make_scalar_grid(nlat, nlon, nt)
    lshags, lwork_shagsi, ldwork_shagsi = calc_shags_sizes(nlat, nlon)
    wshags, ierr0 = fort_sp.shagsi(nlat, nlon, lshags, lwork_shagsi, ldwork_shagsi)
    assert ierr0 == 0, ("shagsi failed", nlat, nlon, ierr0)

    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shags = nlat * (nt * nlon + max(3 * l2, nlon))
        a, b, ierr1 = fort_sp.shags(divg, np.asarray(wshags, dtype=np.float32), lwork_shags)
    else:
        lwork_shags = l2 * (nt * nlon + max(3 * nlat, nlon))
        a, b, ierr1 = fort_sp.shags(divg, np.asarray(wshags, dtype=np.float32), lwork_shags, isym=isym)
    assert ierr1 == 0, ("shags failed", nlat, nlon, nt, isym, ierr1)

    lvhsgs, ldwork_vhsgsi = calc_vhsgs_sizes(nlat, nlon)
    wvhsgs, ierr2 = fort_sp.vhsgsi(nlat, nlon, lvhsgs, ldwork_vhsgsi)
    assert ierr2 == 0, ("vhsgsi failed", nlat, nlon, ierr2)

    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    imid = (nlat + 1) // 2
    lwork_branch_fortran_eq0 = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 2 * mn + nlat
    lwork_branch_fortran_ne0 = nlat * (2 * nt * nlon + max(6 * imid, nlon)) + 2 * mn + nlat
    lwork_doc_eq0 = nlat * ((2 * nt + 1) * nlon + 2 * nt * min(nlat, (nlon + 1) // 2) + 1)
    lwork_doc_ne0 = (2 * nt + 1) * imid * nlon + nlat * (2 * nt * min(nlat, (nlon + 1) // 2) + 1)

    if isym == 0:
        lwork_idivgs = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_eq0)
        v_f, w_f, p_f, ierr_f = fort_sp.idivgs(nlon, a, b, np.asarray(wvhsgs, dtype=np.float32), lwork_idivgs)
        v_r, w_r, p_r, ierr_r = rust_sp.idivgs(a, b, np.asarray(wvhsgs, dtype=np.float32), lwork_idivgs)
    else:
        lwork_idivgs = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_ne0)
        v_f, w_f, p_f, ierr_f = fort_sp.idivgs(nlon, a, b, np.asarray(wvhsgs, dtype=np.float32), lwork_idivgs, isym=isym)
        v_r, w_r, p_r, ierr_r = rust_sp.idivgs_isym(a, b, isym, np.asarray(wvhsgs, dtype=np.float32), lwork_idivgs)

    assert ierr_f == 0, ("fortran idivgs failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust idivgs failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("idivgs v", v_f, v_r)
    summarize_diff("idivgs w", w_f, w_r)
    summarize_diff("idivgs pertrb", p_f, p_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
