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


def calc_shaes_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lshaes = (min(nlat, (nlon + 2) // 2) * l2 * (2 * nlat - min(nlat, (nlon + 2) // 2) + 1)) // 2 + nlon + 15
    lwork = 5 * nlat * l2 + (3 * ((min(nlat, (nlon + 2) // 2) - 2) * (2 * nlat - min(nlat, (nlon + 2) // 2) - 1))) // 2
    ldwork = nlat + 1
    return lshaes, lwork, ldwork


def calc_vhses_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhses = l1 * l2 * (2 * nlat - l1 + 1) + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lvhses, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    dv = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        dv[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.05
        )
    return dv


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    dv = make_scalar_grid(nlat, nlon, nt)
    lshaes, lwork_shaesi, ldwork_shaesi = calc_shaes_sizes(nlat, nlon)
    wshaes, ierr0 = fort_sp.shaesi(nlat, nlon, lshaes, lwork_shaesi, ldwork_shaesi)
    assert ierr0 == 0, ("shaesi failed", nlat, nlon, ierr0)

    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shaes = nlat * (2 * nt * nlon + max(6 * l2, nlon))
        a, b, ierr1 = fort_sp.shaes(dv, np.asarray(wshaes, dtype=np.float32), lwork_shaes)
    else:
        lwork_shaes = l2 * (2 * nt * nlon + max(6 * nlat, nlon))
        a, b, ierr1 = fort_sp.shaes(dv, np.asarray(wshaes, dtype=np.float32), lwork_shaes, isym=isym)
    assert ierr1 == 0, ("shaes failed", nlat, nlon, nt, isym, ierr1)

    lvhses, ldwork_vhsesi = calc_vhses_sizes(nlat, nlon)
    init_lwork_vhsesi = 3 * max(min(nlat, (nlon + 1) // 2) - 2, 0) * (2 * nlat - min(nlat, (nlon + 1) // 2) - 1) // 2 + 5 * ((nlat + 1) // 2) * nlat
    wvhses, ierr2 = fort_sp.vhsesi(nlat, nlon, lvhses, init_lwork_vhsesi, ldwork_vhsesi)
    assert ierr2 == 0, ("vhsesi failed", nlat, nlon, ierr2)

    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    imid = (nlat + 1) // 2
    lwork_branch_fortran_eq0 = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 2 * mn + nlat
    lwork_branch_fortran_ne0 = nlat * (2 * nt * nlon + max(6 * imid, nlon)) + 2 * mn + nlat
    lwork_doc_eq0 = nlat * ((2 * nt + 1) * nlon + 2 * nt * min(nlat, (nlon + 1) // 2) + 1)
    lwork_doc_ne0 = (2 * nt + 1) * imid * nlon + nlat * (2 * nt * min(nlat, (nlon + 1) // 2) + 1)

    if isym == 0:
        lwork_idives = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_eq0)
        v_f, w_f, p_f, ierr_f = fort_sp.idives(nlon, a, b, np.asarray(wvhses, dtype=np.float32), lwork_idives)
        v_r, w_r, p_r, ierr_r = rust_sp.idives(a, b, np.asarray(wvhses, dtype=np.float32), lwork_idives)
    else:
        lwork_idives = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_ne0)
        v_f, w_f, p_f, ierr_f = fort_sp.idives(nlon, a, b, np.asarray(wvhses, dtype=np.float32), lwork_idives, isym=isym)
        v_r, w_r, p_r, ierr_r = rust_sp.idives_isym(a, b, isym, np.asarray(wvhses, dtype=np.float32), lwork_idives)

    assert ierr_f == 0, ("fortran idives failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust idives failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("idives v", v_f, v_r)
    summarize_diff("idives w", w_f, w_r)
    summarize_diff("idives pertrb", p_f, p_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
