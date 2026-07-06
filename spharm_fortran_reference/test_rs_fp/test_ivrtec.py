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


def calc_shaec_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshaec = 2 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) // 2 + nlon + 15
    ldwork = 2 * (nlat + 1)
    return lshaec, ldwork


def calc_vhsec_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhsec = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat + 2
    return lvhsec, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    vort = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        vort[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.05
        )
    return vort


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    vort = make_scalar_grid(nlat, nlon, nt)
    lshaec, ldwork_shaeci = calc_shaec_sizes(nlat, nlon)
    wshaec, ierr0 = fort_sp.shaeci(nlat, nlon, lshaec, ldwork_shaeci)
    assert ierr0 == 0, ("shaeci failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shaec = nlat * (2 * nt * nlon + max(6 * l2, nlon))
        a, b, ierr1 = fort_sp.shaec(vort, np.asarray(wshaec, dtype=np.float32), lwork_shaec)
    else:
        lwork_shaec = l2 * (2 * nt * nlon + max(6 * nlat, nlon))
        a, b, ierr1 = fort_sp.shaec(vort, np.asarray(wshaec, dtype=np.float32), lwork_shaec, isym=isym)
    assert ierr1 == 0, ("shaec failed", nlat, nlon, nt, isym, ierr1)
    lvhsec, ldwork_vhseci = calc_vhsec_sizes(nlat, nlon)
    wvhsec, ierr2 = fort_sp.vhseci(nlat, nlon, lvhsec, ldwork_vhseci)
    assert ierr2 == 0, ("vhseci failed", nlat, nlon, ierr2)
    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    imid = (nlat + 1) // 2
    lwork_branch_fortran_eq0 = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 2 * mn + nlat
    lwork_branch_fortran_ne0 = nlat * (2 * nt * nlon + max(6 * imid, nlon)) + 2 * mn + nlat
    if isym == 0:
        lwork_ivrtec = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0)
        v_f, w_f, p_f, ierr_f = fort_sp.ivrtec(nlon, a, b, np.asarray(wvhsec, dtype=np.float32), lwork_ivrtec)
        v_r, w_r, p_r, ierr_r = rust_sp.ivrtec(nlon, a, b, np.asarray(wvhsec, dtype=np.float32), lwork_ivrtec)
    else:
        lwork_ivrtec = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0)
        v_f, w_f, p_f, ierr_f = fort_sp.ivrtec(nlon, a, b, np.asarray(wvhsec, dtype=np.float32), lwork_ivrtec, isym=isym)
        v_r, w_r, p_r, ierr_r = rust_sp.ivrtec_isym(nlon, a, b, isym, np.asarray(wvhsec, dtype=np.float32), lwork_ivrtec)
    assert ierr_f == 0, ("fortran ivrtec failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust ivrtec failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("ivrtec v", v_f, v_r)
    summarize_diff("ivrtec w", w_f, w_r)
    summarize_diff("ivrtec pertrb", p_f, p_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
