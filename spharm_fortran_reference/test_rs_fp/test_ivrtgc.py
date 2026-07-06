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


def calc_shagc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshagc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshagc, ldwork


def calc_vhsgc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhsgc = 2 * (lzz1 + labc) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


def make_scalar_grid(nlat: int, nlon: int, nt: int):
    theta = np.linspace(0.0, np.pi, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    vt = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        vt[:, :, k] = (
            np.cos(theta)[:, None] * np.cos((k + 1) * lon)[None, :]
            + 0.2 * np.sin(theta)[:, None] * np.sin((k + 2) * lon)[None, :]
            + 0.05
        )
    return vt


def run_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    vt = make_scalar_grid(nlat, nlon, nt)
    lshagc, ldwork_shagci = calc_shagc_sizes(nlat, nlon)
    wshagc, ierr0 = fort_sp.shagci(nlat, nlon, lshagc, ldwork_shagci)
    assert ierr0 == 0, ("shagci failed", nlat, nlon, ierr0)
    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_shagc = nlat * (nt * nlon + max(3 * l2, nlon))
        a, b, ierr1 = fort_sp.shagc(vt, np.asarray(wshagc, dtype=np.float32), lwork_shagc)
    else:
        lwork_shagc = l2 * (nt * nlon + max(3 * nlat, nlon))
        a, b, ierr1 = fort_sp.shagc(vt, np.asarray(wshagc, dtype=np.float32), lwork_shagc, isym=isym)
    assert ierr1 == 0, ("shagc failed", nlat, nlon, nt, isym, ierr1)
    lvhsgc, ldwork_vhsgci = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc, ierr2 = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork_vhsgci)
    assert ierr2 == 0, ("vhsgci failed", nlat, nlon, ierr2)
    mmax = min(nlat, (nlon + 1) // 2)
    mn = mmax * nlat * nt
    imid = (nlat + 1) // 2
    lwork_branch_fortran_eq0 = imid * (2 * nt * nlon + max(6 * nlat, nlon)) + 2 * mn + nlat
    lwork_branch_fortran_ne0 = nlat * (2 * nt * nlon + max(6 * imid, nlon)) + 2 * mn + nlat
    lwork_doc_eq0 = nlat * (2 * nt * nlon + max(6 * ((nlat + 1) // 2), nlon))
    lwork_doc_ne0 = ((nlat + 1) // 2) * (2 * nt * nlon + max(6 * nlat, nlon))

    if isym == 0:
        lwork_ivrtgc = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_eq0)
        v_f, w_f, p_f, ierr_f = fort_sp.ivrtgc(nlon, a, b, np.asarray(wvhsgc, dtype=np.float32), lwork_ivrtgc)
        v_r, w_r, p_r, ierr_r = rust_sp.ivrtgc(a, b, np.asarray(wvhsgc, dtype=np.float32), lwork_ivrtgc)
    else:
        lwork_ivrtgc = max(lwork_branch_fortran_eq0, lwork_branch_fortran_ne0, lwork_doc_ne0)
        v_f, w_f, p_f, ierr_f = fort_sp.ivrtgc(nlon, a, b, np.asarray(wvhsgc, dtype=np.float32), lwork_ivrtgc, isym=isym)
        v_r, w_r, p_r, ierr_r = rust_sp.ivrtgc_isym(a, b, isym, np.asarray(wvhsgc, dtype=np.float32), lwork_ivrtgc)
    assert ierr_f == 0, ("fortran ivrtgc failed", nlat, nlon, nt, isym, ierr_f)
    assert ierr_r == 0, ("rust ivrtgc failed", nlat, nlon, nt, isym, ierr_r)
    summarize_diff("ivrtgc v", v_f, v_r)
    summarize_diff("ivrtgc w", w_f, w_r)
    summarize_diff("ivrtgc pertrb", p_f, p_r)


if __name__ == "__main__":
    for case in [(3, 4, 1, 0), (4, 4, 1, 0), (5, 8, 2, 0), (73, 144, 1, 0)]:
        run_case(*case)
