use crate::hrfftb::hrfftb_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn infer_nlon_from_wvhsgs(nlat: usize, wvhsgs: &[f32]) -> Option<(usize, usize, usize)> {
    let imid = (nlat + 1) / 2;
    let lmn = nlat * (nlat + 1) / 2;
    let lzimn = imid * lmn;
    let ltotal = wvhsgs.len();

    // Prioritize matching the "complete vhsgsi workspace length" first.
    // full = mmax * imid * (2*nlat - mmax + 1) + nlon + 15 + 2*nlat
    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);
        let full_required = mmax * imid * (2 * nlat - mmax + 1) + nlon + 15 + 2 * nlat;
        if ltotal == full_required {
            return Some((nlon, mmax, lzimn));
        }
    }

    // Match the "trimmed base length"
    // base = 2*lzimn + nlon + 15
    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);
        let base_required = 2 * lzimn + nlon + 15;
        if ltotal == base_required {
            return Some((nlon, mmax, lzimn));
        }
    }

    // Finally, do a loose fallback: if it's a longer full/base, and the rest is almost all zeros, also accept
    let mut fallback = None;

    for nlon in 4..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);

        let full_required = mmax * imid * (2 * nlat - mmax + 1) + nlon + 15 + 2 * nlat;
        if ltotal > full_required {
            let trailing_is_zero = wvhsgs[full_required..]
                .iter()
                .all(|&x| x.abs() <= f32::EPSILON);
            if trailing_is_zero {
                return Some((nlon, mmax, lzimn));
            }
            fallback.get_or_insert((nlon, mmax, lzimn));
        }

        let base_required = 2 * lzimn + nlon + 15;
        if ltotal > base_required {
            let trailing_is_zero = wvhsgs[base_required..]
                .iter()
                .all(|&x| x.abs() <= f32::EPSILON);
            if trailing_is_zero {
                return Some((nlon, mmax, lzimn));
            }
            fallback.get_or_insert((nlon, mmax, lzimn));
        }
    }

    fallback
}

/// Synthesize vector fields on a Gaussian grid using stored Legendre tables.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhsgs_impl(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhsgs: &[f32],
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
    let (nlon, mmax, lzimn) = infer_nlon_from_wvhsgs(nlat, wvhsgs)
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvhsgs"))?;

    let idv = if ityp <= 2 { nlat } else { imid };
    ierror = 10;
    if lwork < (2 * nt + 1) * idv * nlon {
        return Ok((Vec::new(), Vec::new(), 0, 0, ierror));
    }

    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];

    let vb = &wvhsgs[..lzimn];
    let wb = &wvhsgs[lzimn..2 * lzimn];
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
                    for i in 1..=imid {
                        ve[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                        we[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                        wo[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[slot(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[slot(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    if mmax >= 2 {
        for mp1 in 2..=mmax {
            let m = mp1 - 1;
            let mb = m * nlat - (m * (m + 1)) / 2;
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
                                    vo[slot(i, jc, k)] += br[cidx] * vb[z];
                                    ve[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vo[slot(i, js, k)] += bi[cidx] * vb[z];
                                    ve[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wo[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    we[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wo[slot(i, js, k)] -= ci[cidx] * vb[z];
                                    we[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 6 | 7 => {
                                    vo[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vo[slot(i, js, k)] += bi[cidx] * vb[z];
                                    we[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 3 | 5 => {
                                    ve[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    ve[slot(i, js, k)] += cr[cidx] * wb[z];
                                    wo[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[slot(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            match ityp {
                                0 => {
                                    ve[slot(imid, jc, k)] -= ci[cidx] * wb[z];
                                    ve[slot(imid, js, k)] += cr[cidx] * wb[z];
                                    we[slot(imid, jc, k)] -= bi[cidx] * wb[z];
                                    we[slot(imid, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 6 | 7 => {
                                    we[slot(imid, jc, k)] -= bi[cidx] * wb[z];
                                    we[slot(imid, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 3 | 5 => {
                                    ve[slot(imid, jc, k)] -= ci[cidx] * wb[z];
                                    ve[slot(imid, js, k)] += cr[cidx] * wb[z];
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
                                    ve[slot(i, jc, k)] += br[cidx] * vb[z];
                                    vo[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    ve[slot(i, js, k)] += bi[cidx] * vb[z];
                                    vo[slot(i, js, k)] += cr[cidx] * wb[z];
                                    we[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[slot(i, js, k)] -= ci[cidx] * vb[z];
                                    wo[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 3 | 4 => {
                                    ve[slot(i, jc, k)] += br[cidx] * vb[z];
                                    ve[slot(i, js, k)] += bi[cidx] * vb[z];
                                    wo[slot(i, jc, k)] -= bi[cidx] * wb[z];
                                    wo[slot(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 6 | 8 => {
                                    vo[slot(i, jc, k)] -= ci[cidx] * wb[z];
                                    vo[slot(i, js, k)] += cr[cidx] * wb[z];
                                    we[slot(i, jc, k)] -= cr[cidx] * vb[z];
                                    we[slot(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            match ityp {
                                0 => {
                                    ve[slot(imid, jc, k)] += br[cidx] * vb[z];
                                    ve[slot(imid, js, k)] += bi[cidx] * vb[z];
                                    we[slot(imid, jc, k)] -= cr[cidx] * vb[z];
                                    we[slot(imid, js, k)] -= ci[cidx] * vb[z];
                                }
                                1 | 3 | 4 => {
                                    ve[slot(imid, jc, k)] += br[cidx] * vb[z];
                                    ve[slot(imid, js, k)] += bi[cidx] * vb[z];
                                }
                                2 | 6 | 8 => {
                                    we[slot(imid, jc, k)] -= cr[cidx] * vb[z];
                                    we[slot(imid, js, k)] -= ci[cidx] * vb[z];
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
            ve[idx] += vo[idx];
            we[idx] += wo[idx];
        }
    }

    let synthesized_planes: Vec<(Vec<f32>, Vec<f32>)> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut plane_v = vec![0.0_f32; idv * nlon];
            let mut plane_w = vec![0.0_f32; idv * nlon];
            if ityp <= 2 {
                for i in 1..=imid {
                    for j in 1..=nlon {
                        plane_v[(j - 1) * idv + (i - 1)] = ve[slot(i, j, k)];
                        plane_w[(j - 1) * idv + (i - 1)] = we[slot(i, j, k)];
                    }
                }
                for i in 1..=imm1 {
                    for j in 1..=nlon {
                        plane_v[(j - 1) * idv + (imid + i - 1)] = vo[slot(i, j, k)];
                        plane_w[(j - 1) * idv + (imid + i - 1)] = wo[slot(i, j, k)];
                    }
                }
            } else {
                for i in 1..=idv {
                    for j in 1..=nlon {
                        plane_v[(j - 1) * idv + (i - 1)] = ve[slot(i, j, k)];
                        plane_w[(j - 1) * idv + (i - 1)] = we[slot(i, j, k)];
                    }
                }
            }
            hrfftb_impl(
                idv,
                nlon,
                &mut plane_v,
                &wvhsgs[2 * lzimn..2 * lzimn + nlon + 15],
            )?;
            hrfftb_impl(
                idv,
                nlon,
                &mut plane_w,
                &wvhsgs[2 * lzimn..2 * lzimn + nlon + 15],
            )?;
            Ok::<(Vec<f32>, Vec<f32>), PyErr>((plane_v, plane_w))
        })
        .collect::<PyResult<Vec<_>>>()?;

    for (k, (plane_v, plane_w)) in synthesized_planes.into_iter().enumerate() {
        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    ve[slot(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                    we[slot(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    vo[slot(i, j, k)] = plane_v[(j - 1) * idv + (imid + i - 1)];
                    wo[slot(i, j, k)] = plane_w[(j - 1) * idv + (imid + i - 1)];
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    ve[slot(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                    we[slot(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                }
            }
        }
    }

    let mut v = vec![0.0_f32; idv * nlon * nt];
    let mut w = vec![0.0_f32; idv * nlon * nt];
    if ityp <= 2 {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    v[slot(i, j, k)] = 0.5 * (ve[slot(i, j, k)] + vo[slot(i, j, k)]);
                    w[slot(i, j, k)] = 0.5 * (we[slot(i, j, k)] + wo[slot(i, j, k)]);
                    v[slot(nlp1 - i, j, k)] = 0.5 * (ve[slot(i, j, k)] - vo[slot(i, j, k)]);
                    w[slot(nlp1 - i, j, k)] = 0.5 * (we[slot(i, j, k)] - wo[slot(i, j, k)]);
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    v[slot(i, j, k)] = 0.5 * ve[slot(i, j, k)];
                    w[slot(i, j, k)] = 0.5 * we[slot(i, j, k)];
                }
            }
        }
    }
    if mlat != 0 {
        for k in 0..nt {
            for j in 1..=nlon {
                v[slot(imid, j, k)] = 0.5 * ve[slot(imid, j, k)];
                w[slot(imid, j, k)] = 0.5 * we[slot(imid, j, k)];
            }
        }
    }

    Ok((v, w, idv, nlon, 0))
}

fn expand_vhsgs_output(
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

#[pyfunction]
/// Python wrapper for `vhsgs_impl` using the default vector layout.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vhsgs<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
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
            "vhsgs expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let brbuf = br
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let bibuf = bi
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let crbuf = cr
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let cibuf = ci
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let wbuf = wvhsgs
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let (v, w, idv, nlon, ierror) = py
        .detach(|| {
            vhsgs_impl(&brbuf, &bibuf, &crbuf, &cibuf, nlat, nt, 0, &wbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let v = expand_vhsgs_output(v, nlat, nlon, nt, idv);
    let w = expand_vhsgs_output(w, nlat, nlon, nt, idv);
    let out_shape = if bshape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let v_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), w)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Python wrapper for `vhsgs_impl` with an explicit `ityp` selector.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhsgs`: Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vhsgs_ityp<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
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
            "vhsgs_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvhsgs_arr = wvhsgs.as_array().as_standard_layout().to_owned();
    let (v, w, idv, nlon, ierror) = vhsgs_impl(
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
        wvhsgs_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wvhsgs is not standard-layout after copy"))?,
        lwork,
    )?;
    let v = expand_vhsgs_output(v, nlat, nlon, nt, idv);
    let w = expand_vhsgs_output(w, nlat, nlon, nt, idv);
    let out_shape = if bshape.len() == 2 {
        vec![nlat, nlon]
    } else {
        vec![nlat, nlon, nt]
    };
    let v_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), w)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
