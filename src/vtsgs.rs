use crate::gaussian_stored_vector::vtsgsi_core;
use crate::hrfftb::hrfftb_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn infer_nlon_from_wvts(nlat: usize, wvts: &[f32]) -> Option<(usize, usize, usize)> {
    let imid = (nlat + 1) / 2;
    let ltotal = wvts.len();

    for nlon in 1..=4 * nlat.max(4) {
        let mmax_synth = nlat.min((nlon + 1) / 2);
        let idz_synth = mmax_synth * (2 * nlat - mmax_synth + 1) / 2;
        let lzimn_synth = imid * idz_synth;
        let synth_required = 2 * lzimn_synth + nlon + 15;
        if ltotal == synth_required {
            return Some((nlon, mmax_synth, lzimn_synth));
        }

        let mmax_init = nlat.min(nlon / 2 + 1);
        let idz_init = mmax_init * (2 * nlat - mmax_init + 1) / 2;
        let lzimn_init = imid * idz_init;
        let init_required = 2 * lzimn_init + nlon + 15;
        if ltotal == init_required {
            // Fortran vtsgsi initializes a larger table (mmax = nlon/2+1 for even nlon),
            // but vtsgs consumes only the synthesis prefix with mmax = (nlon+1)/2.
            return Some((nlon, mmax_synth, lzimn_synth));
        }
    }

    let _ = wvts;
    None
}

fn expand_vtsgs_output(
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

pub fn vtsgs_impl(
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
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let size = br.len();
    if bi.len() != size || cr.len() != size || ci.len() != size {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let imid = (nlat + 1) / 2;
    let (nlon, mmax, lzimn) = infer_nlon_from_wvts(nlat, wvts)
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvts"))?;

    let idv = if ityp <= 2 { nlat } else { imid };
    ierror = 10;
    if lwork < (2 * nt + 1) * idv * nlon {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let mut vte = vec![0.0_f32; idv * nlon * nt];
    let mut vto = vec![0.0_f32; idv * nlon * nt];
    let mut wte = vec![0.0_f32; idv * nlon * nt];
    let mut wto = vec![0.0_f32; idv * nlon * nt];

    let vb = &wvts[..lzimn];
    let wb = &wvts[lzimn..2 * lzimn];
    let wrfft = &wvts[2 * lzimn..2 * lzimn + nlon + 15];

    let nlp1 = nlat + 1;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };

    let coeff = |mp1: usize, np1: usize, k: usize| ((mp1 - 1) * nlat + (np1 - 1)) * nt + k;
    let basis = |i: usize, mn: usize| (i - 1) + (mn - 1) * imid;
    let slot = |i: usize, j: usize, k: usize| ((i - 1) * nlon + (j - 1)) * nt + k;

    match ityp {
        0 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                        wto[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                        wte[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        vto[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        wte[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imid {
                        vte[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imm1 {
                        wto[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    if mmax >= 2 {
        for mp1 in 2..=mmax {
            let m = mp1 - 1;
            let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
            let mp2 = mp1 + 1;

            if mp1 <= ndo1 {
                for k in 0..nt {
                    for np1 in (mp1..=ndo1).step_by(2) {
                        let mn = mb + np1;
                        let cidx = coeff(mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let z = basis(i, mn);
                            match ityp {
                                0 => {
                                    vte[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vto[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vte[slot(i, js, k)] += bi[cidx] * vb[z];
                                    vto[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wte[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wto[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wte[slot(i, js, k)] -= ci[cidx] * vb[z];
                                    wto[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 6 | 7 => {
                                    vte[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vte[slot(i, js, k)] += bi[cidx] * vb[z];
                                    wto[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wto[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 3 | 5 => {
                                    vto[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vto[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wte[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wte[slot(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            match ityp {
                                0 => {
                                    vte[slot(imid, jc, k)] += br[cidx] * vb[z];
                                    vte[slot(imid, js, k)] += bi[cidx] * vb[z];
                                    wte[slot(imid, jc, k)] -= cr[cidx] * vb[z];
                                    wte[slot(imid, js, k)] -= ci[cidx] * vb[z];
                                }
                                1 | 6 | 7 => {
                                    vte[slot(imid, jc, k)] += br[cidx] * vb[z];
                                    vte[slot(imid, js, k)] += bi[cidx] * vb[z];
                                }
                                2 | 3 | 5 => {
                                    wte[slot(imid, jc, k)] -= cr[cidx] * vb[z];
                                    wte[slot(imid, js, k)] -= ci[cidx] * vb[z];
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
                        let mn = mb + np1;
                        let cidx = coeff(mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let z = basis(i, mn);
                            match ityp {
                                0 => {
                                    vto[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vte[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vto[slot(i, js, k)] += bi[cidx] * vb[z];
                                    vte[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wto[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wte[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wto[slot(i, js, k)] -= ci[cidx] * vb[z];
                                    wte[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 3 | 4 => {
                                    vto[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vto[slot(i, js, k)] += bi[cidx] * vb[z];
                                    wte[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wte[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 6 | 8 => {
                                    vte[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vte[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wto[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wto[slot(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            match ityp {
                                0 => {
                                    vte[slot(imid, jc, k)] -= ci[cidx] * wb[z];
                                    vte[slot(imid, js, k)] += cr[cidx] * wb[z];
                                    wte[slot(imid, jc, k)] -= bi[cidx] * wb[z];
                                    wte[slot(imid, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 3 | 4 => {
                                    wte[slot(imid, jc, k)] -= bi[cidx] * wb[z];
                                    wte[slot(imid, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 6 | 8 => {
                                    vte[slot(imid, jc, k)] -= ci[cidx] * wb[z];
                                    vte[slot(imid, js, k)] += cr[cidx] * wb[z];
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
                    plane_v[(j - 1) * idv + (i - 1)] = vte[slot(i, j, k)];
                    plane_w[(j - 1) * idv + (i - 1)] = wte[slot(i, j, k)];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    plane_v[(j - 1) * idv + (imid + i - 1)] = vto[slot(i, j, k)];
                    plane_w[(j - 1) * idv + (imid + i - 1)] = wto[slot(i, j, k)];
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    plane_v[(j - 1) * idv + (i - 1)] = vte[slot(i, j, k)];
                    plane_w[(j - 1) * idv + (i - 1)] = wte[slot(i, j, k)];
                }
            }
        }

        hrfftb_impl(idv, nlon, &mut plane_v, wrfft)?;
        hrfftb_impl(idv, nlon, &mut plane_w, wrfft)?;

        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    vte[slot(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                    wte[slot(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    vto[slot(i, j, k)] = plane_v[(j - 1) * idv + (imid + i - 1)];
                    wto[slot(i, j, k)] = plane_w[(j - 1) * idv + (imid + i - 1)];
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    vte[slot(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                    wte[slot(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
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
                    vt[slot(i, j, k)] = 0.5 * (vte[slot(i, j, k)] + vto[slot(i, j, k)]);
                    wt[slot(i, j, k)] = 0.5 * (wte[slot(i, j, k)] + wto[slot(i, j, k)]);
                    vt[slot(nlp1 - i, j, k)] = 0.5 * (vte[slot(i, j, k)] - vto[slot(i, j, k)]);
                    wt[slot(nlp1 - i, j, k)] = 0.5 * (wte[slot(i, j, k)] - wto[slot(i, j, k)]);
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    vt[slot(i, j, k)] = 0.5 * vte[slot(i, j, k)];
                    wt[slot(i, j, k)] = 0.5 * wte[slot(i, j, k)];
                }
            }
        }
    }
    if mlat != 0 {
        for k in 0..nt {
            for j in 1..=nlon {
                vt[slot(imid, j, k)] = 0.5 * vte[slot(imid, j, k)];
                wt[slot(imid, j, k)] = 0.5 * wte[slot(imid, j, k)];
            }
        }
    }

    Ok((vt, wt, idv, nlon, 0))
}

pub fn vtsgsi_impl(nlat: i32, nlon: i32, lwvts: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let mmax = nlat.min(nlon / 2 + 1);
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lwvts < 2 * lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 3 * nlat + 2 {
        return (Vec::new(), ierror);
    }

    match vtsgsi_core(nlat as usize, nlon as usize) {
        Ok(w) => {
            let mut out = vec![0.0_f32; usize::try_from(lwvts).unwrap_or(0)];
            let copy_len = out.len().min(w.len());
            out[..copy_len].copy_from_slice(&w[..copy_len]);
            (out, 0)
        }
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
pub fn vtsgsi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lwvts: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvts, ierror) = vtsgsi_impl(nlat, nlon, lwvts, ldwork);
    Ok((PyArray1::from_vec(py, wvts).to_owned(), ierror))
}

#[pyfunction]
pub fn vtsgs<'py>(
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
            "vtsgs expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgs_impl(
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

    let vt = expand_vtsgs_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsgs_output(wt, nlat, nlon, nt, idv);
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
pub fn vtsgs_ityp<'py>(
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
            "vtsgs_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgs_impl(
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

    let vt = expand_vtsgs_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsgs_output(wt, nlat, nlon, nt, idv);
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
