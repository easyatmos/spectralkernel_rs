use crate::hrfftb::hrfftb_impl;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Synthesize vector fields on a regular grid using stored Legendre tables.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the cosine coefficients, sine coefficients, and an error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhses_impl(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhses: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), 3));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    let size = br.len();
    if bi.len() != size || cr.len() != size || ci.len() != size {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    let expected_wvhses_len = lzimn + lzimn + nlon + 15;
    if wvhses.len() < expected_wvhses_len {
        return Err(PyValueError::new_err(format!(
            "wvhses size mismatch: expected at least {expected_wvhses_len}, got {} for nlon={nlon}, nlat={nlat}",
            wvhses.len()
        )));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    ierror = 10;
    if lwork < (2 * nt + 1) * idv * nlon {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];
    let vb = &wvhses[..lzimn];
    let wb = &wvhses[lzimn..2 * lzimn];
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };

    let coeff = |mp1: usize, np1: usize, k: usize| ((mp1 - 1) * nlat + (np1 - 1)) * nt + k;
    let basis = |i: usize, mn: usize| (i - 1) + (mn - 1) * imid;
    let work = |i: usize, j: usize, k: usize| ((i - 1) * nlon + (j - 1)) * nt + k;
    let slot = |i: usize, j: usize, k: usize| ((i - 1) * nlon + (j - 1)) * nt + k;

    match ityp {
        0 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let idx = basis(i, np1);
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[idx];
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[idx];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let idx = basis(i, np1);
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[idx];
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[idx];
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb[basis(i, np1)];
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb[basis(i, np1)];
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
                        for i in 1..=imm1 {
                            let z = basis(i, mn);
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            match ityp {
                                0 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z];
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    vo[work(i, js, k)] += bi[cidx] * vb[z];
                                    ve[work(i, js, k)] += cr[cidx] * wb[z];
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z];
                                    we[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z];
                                    vo[work(i, js, k)] += bi[cidx] * vb[z];
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 => {
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    ve[work(i, js, k)] += cr[cidx] * wb[z];
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                3 => {
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    ve[work(i, js, k)] += cr[cidx] * wb[z];
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                5 => {
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    ve[work(i, js, k)] += cr[cidx] * wb[z];
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                6 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z];
                                    vo[work(i, js, k)] += bi[cidx] * vb[z];
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                7 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z];
                                    vo[work(i, js, k)] += bi[cidx] * vb[z];
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            match ityp {
                                0 => {
                                    ve[work(imid, jc, k)] -= ci[cidx] * wb[z];
                                    ve[work(imid, js, k)] += cr[cidx] * wb[z];
                                    we[work(imid, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(imid, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 7 => {
                                    we[work(imid, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(imid, js, k)] += br[cidx] * wb[z];
                                }
                                6 => {
                                    we[work(imid, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(imid, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 3 | 5 => {
                                    ve[work(imid, jc, k)] -= ci[cidx] * wb[z];
                                    ve[work(imid, js, k)] += cr[cidx] * wb[z];
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
                        for i in 1..=imm1 {
                            let z = basis(i, mn);
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            match ityp {
                                0 => {
                                    ve[work(i, jc, k)] += br[cidx] * vb[z];
                                    ve[work(i, js, k)] += bi[cidx] * vb[z];
                                    vo[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    vo[work(i, js, k)] += cr[cidx] * wb[z];
                                    we[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    wo[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    we[work(i, js, k)] -= ci[cidx] * vb[z];
                                    wo[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                1 | 3 | 4 => {
                                    ve[work(i, jc, k)] += br[cidx] * vb[z];
                                    ve[work(i, js, k)] += bi[cidx] * vb[z];
                                    wo[work(i, jc, k)] -= bi[cidx] * wb[z];
                                    wo[work(i, js, k)] += br[cidx] * wb[z];
                                }
                                2 | 6 | 8 => {
                                    vo[work(i, jc, k)] -= ci[cidx] * wb[z];
                                    vo[work(i, js, k)] += cr[cidx] * wb[z];
                                    we[work(i, jc, k)] -= cr[cidx] * vb[z];
                                    we[work(i, js, k)] -= ci[cidx] * vb[z];
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = basis(imid, mn);
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            match ityp {
                                0 => {
                                    ve[work(imid, jc, k)] += br[cidx] * vb[z];
                                    ve[work(imid, js, k)] += bi[cidx] * vb[z];
                                    we[work(imid, jc, k)] -= cr[cidx] * vb[z];
                                    we[work(imid, js, k)] -= ci[cidx] * vb[z];
                                }
                                1 | 3 | 4 => {
                                    ve[work(imid, jc, k)] += br[cidx] * vb[z];
                                    ve[work(imid, js, k)] += bi[cidx] * vb[z];
                                }
                                2 | 6 | 8 => {
                                    we[work(imid, jc, k)] -= cr[cidx] * vb[z];
                                    we[work(imid, js, k)] -= ci[cidx] * vb[z];
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
                &wvhses[2 * lzimn..2 * lzimn + nlon + 15],
            )?;
            hrfftb_impl(
                idv,
                nlon,
                &mut plane_w,
                &wvhses[2 * lzimn..2 * lzimn + nlon + 15],
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

    let nlp1 = nlat + 1;
    let mut v = vec![0.0_f32; nlat * nlon * nt];
    let mut w = vec![0.0_f32; nlat * nlon * nt];
    if ityp <= 2 {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imm1 {
                    let top = (i * nlon + j) * nt + k;
                    let bot = ((nlp1 - (i + 1) - 1) * nlon + j) * nt + k;
                    let sym = (i * nlon + j) * nt + k;
                    v[top] = 0.5 * (ve[top] + vo[sym]);
                    w[top] = 0.5 * (we[top] + wo[sym]);
                    v[bot] = 0.5 * (ve[top] - vo[sym]);
                    w[bot] = 0.5 * (we[top] - wo[sym]);
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 0..nlon {
                for i in 0..imm1 {
                    let top = (i * nlon + j) * nt + k;
                    v[top] = 0.5 * ve[top];
                    w[top] = 0.5 * we[top];
                }
            }
        }
    }
    if mlat != 0 {
        for k in 0..nt {
            for j in 0..nlon {
                let idx = ((imid - 1) * nlon + j) * nt + k;
                v[idx] = 0.5 * ve[idx];
                w[idx] = 0.5 * we[idx];
            }
        }
    }

    Ok((v, w, 0))
}

fn infer_nlon_from_wvhses(nlat: usize, ltotal: usize) -> Option<usize> {
    let imid = (nlat + 1) / 2;
    for nlon in 1..=4 * nlat.max(4) {
        let n2 = imid;
        let mmax_a = nlat.min((nlon + 1) / 2);
        let ltotal_a = mmax_a * n2 * (nlat + nlat - mmax_a + 1) + nlon + 15;
        if ltotal_a == ltotal {
            return Some(nlon);
        }

        let mmax_b = nlat.min((nlon + 2) / 2);
        let ltotal_b = mmax_b * n2 * (nlat + nlat - mmax_b + 1) + nlon + 15;
        if ltotal_b == ltotal {
            return Some(nlon);
        }
    }
    None
}

#[pyfunction]
/// Python wrapper for `vhses_impl` using the default vector layout.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vhses<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
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
            "vhses expects rank-2 or rank-3 coefficient arrays",
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
    let wbuf = wvhses
        .as_array()
        .as_standard_layout()
        .to_owned()
        .into_raw_vec_and_offset()
        .0;
    let nlon = infer_nlon_from_wvhses(nlat, wbuf.len())
        .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvhses length"))?;
    let (v, w, ierror) = py
        .detach(|| {
            vhses_impl(
                &brbuf, &bibuf, &crbuf, &cibuf, nlat, nlon, nt, 0, &wbuf, lwork,
            )
            .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let nlon = if nlat == 0 {
        0
    } else {
        v.len() / (nlat * nt.max(1))
    };
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
/// Python wrapper for `vhses_impl` with an explicit `ityp` selector.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhses`: Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vhses_ityp<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
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
            "vhses_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvhses_arr = wvhses.as_array().as_standard_layout().to_owned();
    let (v, w, ierror) = vhses_impl(
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
        infer_nlon_from_wvhses(nlat, wvhses_arr.len())
            .ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvhses length"))?,
        nt,
        ityp,
        wvhses_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wvhses is not standard-layout after copy"))?,
        lwork,
    )?;
    let nlon = if nlat == 0 {
        0
    } else {
        v.len() / (nlat * nt.max(1))
    };
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
