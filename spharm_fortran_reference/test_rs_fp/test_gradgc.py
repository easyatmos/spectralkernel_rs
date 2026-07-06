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


def summarize_layout(name, arr):
    arr = np.asarray(arr)
    print(f"\n[{name} layout]")
    print("shape       :", arr.shape)
    print("dtype       :", arr.dtype)
    print("strides     :", arr.strides)
    print("C_CONTIGUOUS:", arr.flags['C_CONTIGUOUS'])
    print("F_CONTIGUOUS:", arr.flags['F_CONTIGUOUS'])
    print("flat(C)[:8] :", np.asarray(arr, order='C').reshape(-1)[:8])
    print("flat(F)[:8] :", np.asarray(arr, order='F').reshape(-1)[:8])


def has_fortran_lowlevel_grad_ops():
    return hasattr(fort_sp, "gradgc") and hasattr(fort_sp, "igradgc")


def calc_shagc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshagc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshagc, ldwork


def calc_vhsgc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 1) // 2)
    l2 = (nlat + 1) // 2
    lvhsgc = 4 * nlat * l2 + 3 * max(l1 - 2, 0) * (2 * nlat - l1 - 1) + nlon + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhsgc, ldwork


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


def calc_shsgc_sizes(nlat: int, nlon: int):
    l1 = min(nlat, (nlon + 2) // 2)
    l2 = (nlat + 1) // 2
    lshsgc = nlat * (2 * l2 + 3 * l1 - 2) + 3 * l1 * (1 - l1) // 2 + nlon + 15
    ldwork = nlat * (nlat + 4)
    return lshsgc, ldwork


def calc_vhagc_sizes(nlat: int, nlon: int):
    imid = (nlat + 1) // 2
    lzz1 = 2 * nlat * imid
    mmax = min(nlat, (nlon + 1) // 2)
    labc = 3 * max(mmax - 2, 0) * (2 * nlat - mmax - 1) // 2
    lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15
    ldwork = 2 * nlat * (nlat + 1) + 1
    return lvhagc, ldwork


def make_grid(nlat: int, nlon: int, nt: int):
    theta, _wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0, ("gaqd failed", nlat, ierr)
    lat = np.sin(0.5 * np.pi - np.asarray(theta, dtype=np.float32))
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False, dtype=np.float32)
    out = np.zeros((nlat, nlon, nt), dtype=np.float32)
    for k in range(nt):
        out[:, :, k] = (
            np.cos((k + 1) * lon)[None, :]
            + np.sin((k + 2) * lon)[None, :] * lat[:, None]
            + 0.25 * (k + 1) * lat[:, None] ** 2
        )
    return out


def gaussian_area_mean(field):
    field = np.asarray(field, dtype=np.float64)
    nlat = field.shape[0]
    _theta, wts, ierr = fort_sp.gaqd(nlat)
    assert ierr == 0, ("gaqd failed", nlat, ierr)
    wts = np.asarray(wts, dtype=np.float64)
    zonal_mean = np.mean(field, axis=1, keepdims=True)
    weighted = np.sum(zonal_mean * wts[:, None, None], axis=0, keepdims=True)
    norm = np.sum(wts)
    return weighted / norm


def run_low_level_case(nlat: int, nlon: int, nt: int, isym: int = 0):
    g = make_grid(nlat, nlon, nt)

    lshagc, ldwork_shagc = calc_shagc_sizes(nlat, nlon)
    wshagc, ierr0 = fort_sp.shagci(nlat, nlon, lshagc, ldwork_shagc)
    assert ierr0 == 0, ("shagci failed", nlat, nlon, ierr0)

    lwork_shagc = nlat * (nt * nlon + max(3 * ((nlat + 1) // 2), nlon))
    a, b, ierr1 = fort_sp.shagc(g, np.asarray(wshagc, dtype=np.float32), lwork_shagc)
    assert ierr1 == 0, ("shagc failed", nlat, nlon, nt, ierr1)

    if (nlat, nlon, nt) in {(4, 4, 1), (5, 8, 2), (73, 144, 1)}:
        summarize_layout("a(from fort_sp.shagc)", a)
        summarize_layout("b(from fort_sp.shagc)", b)

    lvhsgc, ldwork_vhsgc = calc_vhsgc_sizes(nlat, nlon)
    wvhsgc_f, ierr2 = fort_sp.vhsgci(nlat, nlon, lvhsgc, ldwork_vhsgc)
    assert ierr2 == 0, ("vhsgci failed", nlat, nlon, ierr2)

    l2 = (nlat + 1) // 2
    if isym == 0:
        lwork_gradgc = nlat * (2 * nt * nlon + max(6 * l2, nlon) + 2 * nlat * nt + 1)
        v_r, w_r, ierr3_r = rust_sp.gradgc(a, b, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc)
    else:
        lwork_gradgc = l2 * (2 * nt * nlon + max(6 * nlat, nlon)) + nlat * (2 * min(nlat, (nlon + 1) // 2) * nt + 1)
        v_r, w_r, ierr3_r = rust_sp.gradgc_isym(a, b, isym, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc)
    assert ierr3_r == 0, ("rust gradgc failed", nlat, nlon, nt, ierr3_r)

    a_c = np.array(a, dtype=np.float32, order='C', copy=True)
    b_c = np.array(b, dtype=np.float32, order='C', copy=True)
    if isym == 0:
        v_r_c, w_r_c, ierr3_r_c = rust_sp.gradgc(a_c, b_c, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc)
    else:
        v_r_c, w_r_c, ierr3_r_c = rust_sp.gradgc_isym(a_c, b_c, isym, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc)
    assert ierr3_r_c == 0, ("rust gradgc (C copy) failed", nlat, nlon, nt, ierr3_r_c)

    if has_fortran_lowlevel_grad_ops():
        if isym == 0:
            v_f, w_f, ierr3 = fort_sp.gradgc(nlon, a, b, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc)
        else:
            v_f, w_f, ierr3 = fort_sp.gradgc(nlon, a, b, np.asarray(wvhsgc_f, dtype=np.float32), lwork_gradgc, isym=isym)
        assert ierr3 == 0, ("gradgc failed", nlat, nlon, nt, ierr3)
        v_f = crop_half_sphere(v_f, nlat, isym)
        w_f = crop_half_sphere(w_f, nlat, isym)
    else:
        print("\n[warn] installed Fortran _spherepack has no gradgc/igradgc; skip low-level Fortran direct wrapper check")
        v_f, w_f = v_r, w_r

    print(f"\n{'=' * 80}\ngradgc/igradgc low-level closure: nlat={nlat}, nlon={nlon}, nt={nt}, isym={isym}\n{'=' * 80}")
    summarize_diff("gradgc v: fortran vs rust", v_f, v_r)
    summarize_diff("gradgc w: fortran vs rust", w_f, w_r)
    summarize_diff("gradgc v: fortran vs rust(C-copy input)", v_f, v_r_c)
    summarize_diff("gradgc w: fortran vs rust(C-copy input)", w_f, w_r_c)
    if isym == 0:
        lvhagc, ldwork_vhagc = calc_vhagc_sizes(nlat, nlon)
        wvhagc_f, ierr_vhagci = fort_sp.vhagci(nlat, nlon, lvhagc, ldwork_vhagc)
        assert ierr_vhagci == 0, ("vhagci failed", nlat, nlon, ierr_vhagci)

        lwork_vhagc = nlat * (4 * nlon * nt + 6 * l2)
        br_r, bi_r, _cr_r, _ci_r, ierr_vhagc_r = rust_sp.vhagc(
            np.asarray(v_r, dtype=np.float32),
            np.asarray(w_r, dtype=np.float32),
            np.asarray(wvhagc_f, dtype=np.float32),
            lwork_vhagc,
        )
        assert ierr_vhagc_r == 0, ("rust vhagc failed", nlat, nlon, nt, ierr_vhagc_r)

        lshsgc, ldwork_shsgc = calc_shsgc_sizes(nlat, nlon)
        wshsgc_f, ierr4 = fort_sp.shsgci(nlat, nlon, lshsgc, ldwork_shsgc)
        assert ierr4 == 0, ("shsgci failed", nlat, nlon, ierr4)

        if isym == 0:
            lwork_igradgc = nlat * (nt * nlon + max(3 * l2, nlon) + 2 * nt * min(nlat, (nlon + 2) // 2) + 1)
            sf_rec_r, ierr5_r = rust_sp.igradgc(br_r, bi_r, np.asarray(wshsgc_f, dtype=np.float32), lwork_igradgc)
        else:
            lwork_igradgc = l2 * (nt * nlon + max(3 * nlat, nlon)) + nlat * (2 * nt * min(nlat, (nlon + 2) // 2) + 1)
            sf_rec_r, ierr5_r = rust_sp.igradgc_isym(br_r, bi_r, isym, np.asarray(wshsgc_f, dtype=np.float32), lwork_igradgc)
        assert ierr5_r == 0, ("rust igradgc failed", nlat, nlon, nt, ierr5_r)

        if has_fortran_lowlevel_grad_ops():
            br_f, bi_f, _cr_f, _ci_f, ierr_vhagc_f = fort_sp.vhagc(
                np.asarray(v_f, dtype=np.float32),
                np.asarray(w_f, dtype=np.float32),
                np.asarray(wvhagc_f, dtype=np.float32),
                lwork_vhagc,
            )
            assert ierr_vhagc_f == 0, ("vhagc failed", nlat, nlon, nt, ierr_vhagc_f)
            if isym == 0:
                sf_rec_f, ierr5 = fort_sp.igradgc(
                    nlon,
                    np.asarray(br_f, dtype=np.float32),
                    np.asarray(bi_f, dtype=np.float32),
                    np.asarray(wshsgc_f, dtype=np.float32),
                    lwork_igradgc,
                )
            else:
                sf_rec_f, ierr5 = fort_sp.igradgc(
                    nlon,
                    np.asarray(br_f, dtype=np.float32),
                    np.asarray(bi_f, dtype=np.float32),
                    np.asarray(wshsgc_f, dtype=np.float32),
                    lwork_igradgc,
                    isym=isym,
                )
            assert ierr5 == 0, ("igradgc failed", nlat, nlon, nt, ierr5)
            sf_rec_f = crop_half_sphere(sf_rec_f, nlat, isym)
        else:
            sf_rec_f = sf_rec_r

        g_centered = g - gaussian_area_mean(g)
        sf_centered = sf_rec_f - gaussian_area_mean(sf_rec_f)
        sf_centered_r = sf_rec_r - gaussian_area_mean(sf_rec_r)
        summarize_diff("igradgc sf: fortran vs rust", sf_rec_f, sf_rec_r)
        # summarize_diff("sf(centered) vs igradgc(gradgc(sf))(centered)", g_centered, sf_centered)
        # summarize_diff("sf(centered) vs rust_igradgc(rust_gradgc(sf))(centered)", g_centered, sf_centered_r)


if __name__ == "__main__":
    for isym in (0, 1, 2):
        for case in [(3, 4, 1), (4, 4, 1), (5, 8, 2), (73, 144, 1)]:
            run_low_level_case(*case, isym=isym)
