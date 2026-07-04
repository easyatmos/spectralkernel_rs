use crate::gaussian_computed_vector::vtsgci_core;
use crate::hrfftb::hrfftb_impl;
use crate::sphcom_vector::{zvin_column, zwin_column};
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn infer_nlon_from_wvts(nlat: usize, ltotal: usize) -> Option<usize> {
    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    for nlon in 1..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);
        let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
        let need = 2 * (lzz1 + labc) + nlon + 15;
        if need == ltotal {
            return Some(nlon);
        }
    }
    None
}

fn init_mode_for_ityp(ityp: usize) -> i32 {
    match ityp {
        4 | 8 => 1,
        5 | 7 => 2,
        _ => 0,
    }
}

fn coeff_idx(nlat: usize, nt: usize, mp1: usize, np1: usize, k: usize) -> usize {
    ((mp1 - 1) * nlat + (np1 - 1)) * nt + k
}

fn slot(nlon: usize, nt: usize, i: usize, j: usize, k: usize) -> usize {
    let ii = i.saturating_sub(1);
    let jj = j.saturating_sub(1);
    (ii * nlon + jj) * nt + k
}

fn column_idx(imid: usize, i: usize, np1: usize) -> usize {
    (np1 - 1) * imid + (i - 1)
}

fn expand_vtsgc_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    idv: usize,
) -> Vec<f32> {
    if idv == nlat {
        return data;
    }
    let mut full = vec![0.0_f32; nlat * nlon * nt];
    for i in 0..idv {
        for j in 0..nlon {
            for k in 0..nt {
                full[(i * nlon + j) * nt + k] = data[(i * nlon + j) * nt + k];
            }
        }
    }
    full
}

/// Core Rust implementation of `vtsgc`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvts`: Parameter `wvts` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtsgc_impl(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvts: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, usize, usize, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let nlon = infer_nlon_from_wvts(nlat, wvts.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvts length"))?;

    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
    let need_lwvts = 2 * (lzz1 + labc) + nlon + 15;
    ierror = 9;
    if wvts.len() < need_lwvts {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    ierror = 10;
    let idv = if ityp <= 2 { nlat } else { imid };
    let min_lwork = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let size = br.len();
    if bi.len() != size || cr.len() != size || ci.len() != size {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let lwzvin = 2 * nlat * imid + labc;
    let wvbin = &wvts[..lwzvin];
    let wwbin = &wvts[lwzvin..2 * lwzvin];
    let wrfft = &wvts[2 * lwzvin..2 * lwzvin + nlon + 15];

    let br64 = br.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let bi64 = bi.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let cr64 = cr.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let ci64 = ci.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let wvbin64 = wvbin.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let wwbin64 = wwbin.iter().map(|&x| x as f64).collect::<Vec<_>>();

    let mut vte = vec![0.0_f64; idv * nlon * nt];
    let mut vto = vec![0.0_f64; idv * nlon * nt];
    let mut wte = vec![0.0_f64; idv * nlon * nt];
    let mut wto = vec![0.0_f64; idv * nlon * nt];

    let nlp1 = nlat + 1;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };
    let init_mode = init_mode_for_ityp(ityp);

    let vb_m0 = zvin_column(nlat, nlon, init_mode, 0, &wvbin64);

    match ityp {
        0 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                        wto[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                        wte[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(nlon, nt, i, 1, k)] +=
                            br64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(nlon, nt, i, 1, k)] -=
                            cr64[(np1 - 1) * nt + k] * vb_m0[column_idx(imid, i, np1)];
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    if mmax >= 2 {
        for mp1 in 2..=mmax {
            let m = mp1 - 1;
            let mp2 = mp1 + 1;
            let vb_col = zvin_column(nlat, nlon, init_mode, m, &wvbin64);
            let wb_col = zwin_column(nlat, nlon, init_mode, m, &wwbin64);

            if mp1 <= ndo1 {
                for k in 0..nt {
                    for np1 in (mp1..=ndo1).step_by(2) {
                        let cidx = coeff_idx(nlat, nt, mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let vbv = vb_col[column_idx(imid, i, np1)];
                            let wbv = wb_col[column_idx(imid, i, np1)];
                            match ityp {
                                0 => {
                                    vte[slot(nlon, nt, i, jc, k)] += br64[cidx] * vbv;
                                    vto[slot(nlon, nt, i, jc, k)] -= ci64[cidx] * wbv;
                                    vte[slot(nlon, nt, i, js, k)] += bi64[cidx] * vbv;
                                    vto[slot(nlon, nt, i, js, k)] += cr64[cidx] * wbv;
                                    wte[slot(nlon, nt, i, jc, k)] -= cr64[cidx] * vbv;
                                    wto[slot(nlon, nt, i, jc, k)] -= bi64[cidx] * wbv;
                                    wte[slot(nlon, nt, i, js, k)] -= ci64[cidx] * vbv;
                                    wto[slot(nlon, nt, i, js, k)] += br64[cidx] * wbv;
                                }
                                1 | 6 | 7 => {
                                    vte[slot(nlon, nt, i, jc, k)] += br64[cidx] * vbv;
                                    vte[slot(nlon, nt, i, js, k)] += bi64[cidx] * vbv;
                                    wto[slot(nlon, nt, i, jc, k)] -= bi64[cidx] * wbv;
                                    wto[slot(nlon, nt, i, js, k)] += br64[cidx] * wbv;
                                }
                                2 | 3 | 5 => {
                                    vto[slot(nlon, nt, i, jc, k)] -= ci64[cidx] * wbv;
                                    vto[slot(nlon, nt, i, js, k)] += cr64[cidx] * wbv;
                                    wte[slot(nlon, nt, i, jc, k)] -= cr64[cidx] * vbv;
                                    wte[slot(nlon, nt, i, js, k)] -= ci64[cidx] * vbv;
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let vbv = vb_col[column_idx(imid, imid, np1)];
                            match ityp {
                                0 => {
                                    vte[slot(nlon, nt, imid, jc, k)] += br64[cidx] * vbv;
                                    vte[slot(nlon, nt, imid, js, k)] += bi64[cidx] * vbv;
                                    wte[slot(nlon, nt, imid, jc, k)] -= cr64[cidx] * vbv;
                                    wte[slot(nlon, nt, imid, js, k)] -= ci64[cidx] * vbv;
                                }
                                1 | 6 | 7 => {
                                    vte[slot(nlon, nt, imid, jc, k)] += br64[cidx] * vbv;
                                    vte[slot(nlon, nt, imid, js, k)] += bi64[cidx] * vbv;
                                }
                                2 | 3 | 5 => {
                                    wte[slot(nlon, nt, imid, jc, k)] -= cr64[cidx] * vbv;
                                    wte[slot(nlon, nt, imid, js, k)] -= ci64[cidx] * vbv;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if mp2 <= ndo2 {
                for k in 0..nt {
                    for np1 in (mp2..=ndo2).step_by(2) {
                        let cidx = coeff_idx(nlat, nt, mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let vbv = vb_col[column_idx(imid, i, np1)];
                            let wbv = wb_col[column_idx(imid, i, np1)];
                            match ityp {
                                0 => {
                                    vto[slot(nlon, nt, i, jc, k)] += br64[cidx] * vbv;
                                    vte[slot(nlon, nt, i, jc, k)] -= ci64[cidx] * wbv;
                                    vto[slot(nlon, nt, i, js, k)] += bi64[cidx] * vbv;
                                    vte[slot(nlon, nt, i, js, k)] += cr64[cidx] * wbv;
                                    wto[slot(nlon, nt, i, jc, k)] -= cr64[cidx] * vbv;
                                    wte[slot(nlon, nt, i, jc, k)] -= bi64[cidx] * wbv;
                                    wto[slot(nlon, nt, i, js, k)] -= ci64[cidx] * vbv;
                                    wte[slot(nlon, nt, i, js, k)] += br64[cidx] * wbv;
                                }
                                1 | 3 | 4 => {
                                    vto[slot(nlon, nt, i, jc, k)] += br64[cidx] * vbv;
                                    vto[slot(nlon, nt, i, js, k)] += bi64[cidx] * vbv;
                                    wte[slot(nlon, nt, i, jc, k)] -= bi64[cidx] * wbv;
                                    wte[slot(nlon, nt, i, js, k)] += br64[cidx] * wbv;
                                }
                                2 | 6 | 8 => {
                                    vte[slot(nlon, nt, i, jc, k)] -= ci64[cidx] * wbv;
                                    vte[slot(nlon, nt, i, js, k)] += cr64[cidx] * wbv;
                                    wto[slot(nlon, nt, i, jc, k)] -= cr64[cidx] * vbv;
                                    wto[slot(nlon, nt, i, js, k)] -= ci64[cidx] * vbv;
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let wbv = wb_col[column_idx(imid, imid, np1)];
                            match ityp {
                                0 => {
                                    vte[slot(nlon, nt, imid, jc, k)] -= ci64[cidx] * wbv;
                                    vte[slot(nlon, nt, imid, js, k)] += cr64[cidx] * wbv;
                                    wte[slot(nlon, nt, imid, jc, k)] -= bi64[cidx] * wbv;
                                    wte[slot(nlon, nt, imid, js, k)] += br64[cidx] * wbv;
                                }
                                1 | 3 | 4 => {
                                    wte[slot(nlon, nt, imid, jc, k)] -= bi64[cidx] * wbv;
                                    wte[slot(nlon, nt, imid, js, k)] += br64[cidx] * wbv;
                                }
                                2 | 6 | 8 => {
                                    vte[slot(nlon, nt, imid, jc, k)] -= ci64[cidx] * wbv;
                                    vte[slot(nlon, nt, imid, js, k)] += cr64[cidx] * wbv;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    if ityp > 2 {
        for idx in 0..(idv * nlon * nt) {
            vte[idx] += vto[idx];
            wte[idx] += wto[idx];
        }
    }

    for k in 0..nt {
        let mut plane_v = vec![0.0_f32; idv * nlon];
        let mut plane_w = vec![0.0_f32; idv * nlon];
        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (i - 1);
                    plane_v[pidx] = vte[slot(nlon, nt, i, j, k)] as f32;
                    plane_w[pidx] = wte[slot(nlon, nt, i, j, k)] as f32;
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (imid + i - 1);
                    plane_v[pidx] = vto[slot(nlon, nt, i, j, k)] as f32;
                    plane_w[pidx] = wto[slot(nlon, nt, i, j, k)] as f32;
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (i - 1);
                    plane_v[pidx] = vte[slot(nlon, nt, i, j, k)] as f32;
                    plane_w[pidx] = wte[slot(nlon, nt, i, j, k)] as f32;
                }
            }
        }

        hrfftb_impl(idv, nlon, &mut plane_v, wrfft)?;
        hrfftb_impl(idv, nlon, &mut plane_w, wrfft)?;

        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (i - 1);
                    vte[slot(nlon, nt, i, j, k)] = plane_v[pidx] as f64;
                    wte[slot(nlon, nt, i, j, k)] = plane_w[pidx] as f64;
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (imid + i - 1);
                    vto[slot(nlon, nt, i, j, k)] = plane_v[pidx] as f64;
                    wto[slot(nlon, nt, i, j, k)] = plane_w[pidx] as f64;
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    let pidx = (j - 1) * idv + (i - 1);
                    vte[slot(nlon, nt, i, j, k)] = plane_v[pidx] as f64;
                    wte[slot(nlon, nt, i, j, k)] = plane_w[pidx] as f64;
                }
            }
        }
    }

    let mut vt = vec![0.0_f32; idv * nlon * nt];
    let mut wt = vec![0.0_f32; idv * nlon * nt];
    if ityp <= 2 {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    vt[slot(nlon, nt, i, j, k)] = (0.5_f64
                        * (vte[slot(nlon, nt, i, j, k)] + vto[slot(nlon, nt, i, j, k)]))
                        as f32;
                    wt[slot(nlon, nt, i, j, k)] = (0.5_f64
                        * (wte[slot(nlon, nt, i, j, k)] + wto[slot(nlon, nt, i, j, k)]))
                        as f32;
                    vt[slot(nlon, nt, nlp1 - i, j, k)] = (0.5_f64
                        * (vte[slot(nlon, nt, i, j, k)] - vto[slot(nlon, nt, i, j, k)]))
                        as f32;
                    wt[slot(nlon, nt, nlp1 - i, j, k)] = (0.5_f64
                        * (wte[slot(nlon, nt, i, j, k)] - wto[slot(nlon, nt, i, j, k)]))
                        as f32;
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    vt[slot(nlon, nt, i, j, k)] = (0.5_f64 * vte[slot(nlon, nt, i, j, k)]) as f32;
                    wt[slot(nlon, nt, i, j, k)] = (0.5_f64 * wte[slot(nlon, nt, i, j, k)]) as f32;
                }
            }
        }
    }
    if mlat != 0 {
        for k in 0..nt {
            for j in 1..=nlon {
                vt[slot(nlon, nt, imid, j, k)] = (0.5_f64 * vte[slot(nlon, nt, imid, j, k)]) as f32;
                wt[slot(nlon, nt, imid, j, k)] = (0.5_f64 * wte[slot(nlon, nt, imid, j, k)]) as f32;
            }
        }
    }

    Ok((vt, wt, idv, nlon, 0))
}

/// Core Rust implementation of `vtsgci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lwvts`: Parameter `lwvts` passed through to the routine.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtsgci_impl(nlat: i32, nlon: i32, lwvts: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 1 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let imid = (nlat + 1) / 2;
    let lzz1 = 2 * nlat * imid;
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwvts < 2 * (lzz1 + labc) + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < 3 * nlat + 2 {
        return (Vec::new(), ierror);
    }
    match vtsgci_core(nlat as usize, nlon as usize) {
        Ok(w) => (w, 0),
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
/// Rust entry point for `vtsgci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lwvts`: Parameter `lwvts` passed through to the routine.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vtsgci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lwvts: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvts, ierror) = vtsgci_impl(nlat, nlon, lwvts, ldwork);
    Ok((PyArray1::from_vec(py, wvts).to_owned(), ierror))
}

#[pyfunction]
/// Rust entry point for `vtsgc`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wvts`: Parameter `wvts` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vtsgc<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvts: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let bshape = br.shape().to_vec();
    if bshape != bi.shape() || bshape != cr.shape() || bshape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if bshape.len() != 2 && bshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vtsgc expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgc_impl(
        br_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("br is not standard-layout after copy"))?,
        bi_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("bi is not standard-layout after copy"))?,
        cr_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("cr is not standard-layout after copy"))?,
        ci_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("ci is not standard-layout after copy"))?,
        nlat,
        nt,
        0,
        wvts_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wvts is not standard-layout after copy"))?,
        lwork,
    )?;

    let vt = expand_vtsgc_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsgc_output(wt, nlat, nlon, nt, idv);
    let out_shape = if bshape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let vt_arr = ArrayD::from_shape_vec(IxDyn(&out_shape), vt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let wt_arr = ArrayD::from_shape_vec(IxDyn(&out_shape), wt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        vt_arr.into_pyarray(py).into_any().unbind(),
        wt_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Rust entry point for `vtsgc_ityp`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvts`: Parameter `wvts` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vtsgc_ityp<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvts: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let bshape = br.shape().to_vec();
    if bshape != bi.shape() || bshape != cr.shape() || bshape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if bshape.len() != 2 && bshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vtsgc_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgc_impl(
        br_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("br is not standard-layout after copy"))?,
        bi_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("bi is not standard-layout after copy"))?,
        cr_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("cr is not standard-layout after copy"))?,
        ci_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("ci is not standard-layout after copy"))?,
        nlat,
        nt,
        ityp,
        wvts_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wvts is not standard-layout after copy"))?,
        lwork,
    )?;

    let vt = expand_vtsgc_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsgc_output(wt, nlat, nlon, nt, idv);
    let out_shape = if bshape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let vt_arr = ArrayD::from_shape_vec(IxDyn(&out_shape), vt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let wt_arr = ArrayD::from_shape_vec(IxDyn(&out_shape), wt)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        vt_arr.into_pyarray(py).into_any().unbind(),
        wt_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
