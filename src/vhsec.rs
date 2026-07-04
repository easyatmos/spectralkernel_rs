use crate::hrfftb::hrfftb_impl;
use crate::sphcom_vector::{zvin_column, zwin_column};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn vhsec_impl(
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhsec: &[f32],
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
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let lzz1 = 2 * nlat * imid;
    let mut inferred = None;
    for nlon in 1..=4 * nlat.max(4) {
        let mmax = nlat.min((nlon + 1) / 2);
        let labc = 3 * (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
        if wvhsec.len() == 2 * (lzz1 + labc) + nlon + 15 {
            inferred = Some((nlon, mmax, labc));
            break;
        }
    }
    let (nlon, mmax, labc) =
        inferred.ok_or_else(|| PyValueError::new_err("failed to infer nlon from wvhsec"))?;

    ierror = 10;
    let min_lwork = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };
    let lwzvin = lzz1 + labc;
    let expected_wvhsec_len = 2 * lwzvin + nlon + 15;
    if wvhsec.len() < expected_wvhsec_len {
        return Err(PyValueError::new_err(format!(
            "wvhsec size mismatch: expected at least {expected_wvhsec_len}, got {}",
            wvhsec.len()
        )));
    }

    let wvbin = wvhsec[..lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();
    let wwbin = wvhsec[lwzvin..2 * lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();

    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];

    let coeff = |mp1: usize, np1: usize, k: usize| ((mp1 - 1) * nlat + (np1 - 1)) * nt + k;
    let work = |i: usize, j: usize, k: usize| ((i - 1) * nlon + (j - 1)) * nt + k;

    let vb_m0_flag = match ityp {
        4 | 8 => 1,
        5 | 7 => 2,
        _ => 0,
    };
    let vb0 = zvin_column(nlat, nlon, vb_m0_flag, 0, &wvbin);

    match ityp {
        0 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        ve[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        wo[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for np1 in (3..=ndo1).step_by(2) {
                    for i in 1..=imm1 {
                        let z = (np1 - 1) * imid + (i - 1);
                        vo[work(i, 1, k)] += br[(np1 - 1) * nt + k] * vb0[z] as f32;
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for np1 in (2..=ndo2).step_by(2) {
                    for i in 1..=imid {
                        let z = (np1 - 1) * imid + (i - 1);
                        we[work(i, 1, k)] -= cr[(np1 - 1) * nt + k] * vb0[z] as f32;
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
            let vb = zvin_column(nlat, nlon, 0, m, &wvbin);
            let wb = zwin_column(nlat, nlon, 0, m, &wwbin);

            if mp1 <= ndo1 {
                for k in 0..nt {
                    for np1 in (mp1..=ndo1).step_by(2) {
                        let cidx = coeff(mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let z = (np1 - 1) * imid + (i - 1);
                            match ityp {
                                0 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z] as f32;
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    vo[work(i, js, k)] += bi[cidx] * vb[z] as f32;
                                    ve[work(i, js, k)] += cr[cidx] * wb[z] as f32;
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z] as f32;
                                    we[work(i, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                1 | 7 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z] as f32;
                                    vo[work(i, js, k)] += bi[cidx] * vb[z] as f32;
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    we[work(i, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                2 | 3 | 5 => {
                                    ve[work(i, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    ve[work(i, js, k)] += cr[cidx] * wb[z] as f32;
                                    wo[work(i, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    wo[work(i, js, k)] -= ci[cidx] * vb[z] as f32;
                                }
                                6 => {
                                    vo[work(i, jc, k)] += br[cidx] * vb[z] as f32;
                                    vo[work(i, js, k)] += bi[cidx] * vb[z] as f32;
                                    we[work(i, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    we[work(i, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = (np1 - 1) * imid + (imid - 1);
                            match ityp {
                                0 => {
                                    ve[work(imid, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    ve[work(imid, js, k)] += cr[cidx] * wb[z] as f32;
                                    we[work(imid, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    we[work(imid, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                1 | 6 | 7 => {
                                    we[work(imid, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    we[work(imid, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                2 | 3 | 5 => {
                                    ve[work(imid, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    ve[work(imid, js, k)] += cr[cidx] * wb[z] as f32;
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
                        let cidx = coeff(mp1, np1, k);
                        let jc = 2 * mp1 - 2;
                        let js = 2 * mp1 - 1;
                        for i in 1..=imm1 {
                            let z = (np1 - 1) * imid + (i - 1);
                            match ityp {
                                0 => {
                                    ve[work(i, jc, k)] += br[cidx] * vb[z] as f32;
                                    ve[work(i, js, k)] += bi[cidx] * vb[z] as f32;
                                    vo[work(i, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    vo[work(i, js, k)] += cr[cidx] * wb[z] as f32;
                                    we[work(i, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    wo[work(i, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    we[work(i, js, k)] -= ci[cidx] * vb[z] as f32;
                                    wo[work(i, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                1 | 3 | 4 => {
                                    ve[work(i, jc, k)] += br[cidx] * vb[z] as f32;
                                    ve[work(i, js, k)] += bi[cidx] * vb[z] as f32;
                                    wo[work(i, jc, k)] -= bi[cidx] * wb[z] as f32;
                                    wo[work(i, js, k)] += br[cidx] * wb[z] as f32;
                                }
                                2 | 6 | 8 => {
                                    vo[work(i, jc, k)] -= ci[cidx] * wb[z] as f32;
                                    vo[work(i, js, k)] += cr[cidx] * wb[z] as f32;
                                    we[work(i, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    we[work(i, js, k)] -= ci[cidx] * vb[z] as f32;
                                }
                                _ => {}
                            }
                        }
                        if mlat != 0 {
                            let z = (np1 - 1) * imid + (imid - 1);
                            match ityp {
                                0 => {
                                    ve[work(imid, jc, k)] += br[cidx] * vb[z] as f32;
                                    ve[work(imid, js, k)] += bi[cidx] * vb[z] as f32;
                                    we[work(imid, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    we[work(imid, js, k)] -= ci[cidx] * vb[z] as f32;
                                }
                                1 | 3 | 4 => {
                                    ve[work(imid, jc, k)] += br[cidx] * vb[z] as f32;
                                    ve[work(imid, js, k)] += bi[cidx] * vb[z] as f32;
                                }
                                2 | 6 | 8 => {
                                    we[work(imid, jc, k)] -= cr[cidx] * vb[z] as f32;
                                    we[work(imid, js, k)] -= ci[cidx] * vb[z] as f32;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    let mut plane_v = vec![0.0_f32; idv * nlon];
    let mut plane_w = vec![0.0_f32; idv * nlon];
    for k in 0..nt {
        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    plane_v[(j - 1) * idv + (i - 1)] = ve[work(i, j, k)];
                    plane_w[(j - 1) * idv + (i - 1)] = we[work(i, j, k)];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    plane_v[(j - 1) * idv + (imid + i - 1)] = vo[work(i, j, k)];
                    plane_w[(j - 1) * idv + (imid + i - 1)] = wo[work(i, j, k)];
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    plane_v[(j - 1) * idv + (i - 1)] = match ityp {
                        6 | 7 => vo[work(i, j, k)], // v to odd
                        _ => ve[work(i, j, k)],     // 3/4/5 to even
                    };
                    plane_w[(j - 1) * idv + (i - 1)] = match ityp {
                        3 | 5 => wo[work(i, j, k)], // w to odd
                        _ => we[work(i, j, k)],     // 4/6/7/8 to even
                    };
                }
            }
        }

        hrfftb_impl(
            idv,
            nlon,
            &mut plane_v,
            &wvhsec[2 * lwzvin..2 * lwzvin + nlon + 15],
        )?;
        hrfftb_impl(
            idv,
            nlon,
            &mut plane_w,
            &wvhsec[2 * lwzvin..2 * lwzvin + nlon + 15],
        )?;

        if ityp <= 2 {
            for i in 1..=imid {
                for j in 1..=nlon {
                    ve[work(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                    we[work(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    vo[work(i, j, k)] = plane_v[(j - 1) * idv + (imid + i - 1)];
                    wo[work(i, j, k)] = plane_w[(j - 1) * idv + (imid + i - 1)];
                }
            }
        } else {
            for i in 1..=idv {
                for j in 1..=nlon {
                    match ityp {
                        6 | 7 => {
                            vo[work(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                        }
                        _ => {
                            ve[work(i, j, k)] = plane_v[(j - 1) * idv + (i - 1)];
                        }
                    }
                    match ityp {
                        3 | 5 => {
                            wo[work(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                        }
                        _ => {
                            we[work(i, j, k)] = plane_w[(j - 1) * idv + (i - 1)];
                        }
                    }
                }
            }
        }
    }

    let nlp1 = nlat + 1;
    let mut v = vec![0.0_f32; nlat * nlon * nt];
    let mut w = vec![0.0_f32; nlat * nlon * nt];
    let out = |i: usize, j: usize, k: usize| ((i - 1) * nlon + (j - 1)) * nt + k;
    if ityp <= 2 {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    v[out(i, j, k)] = 0.5 * (ve[work(i, j, k)] + vo[work(i, j, k)]);
                    w[out(i, j, k)] = 0.5 * (we[work(i, j, k)] + wo[work(i, j, k)]);
                    v[out(nlp1 - i, j, k)] = 0.5 * (ve[work(i, j, k)] - vo[work(i, j, k)]);
                    w[out(nlp1 - i, j, k)] = 0.5 * (we[work(i, j, k)] - wo[work(i, j, k)]);
                }
            }
        }
    } else {
        for k in 0..nt {
            for j in 1..=nlon {
                for i in 1..=imm1 {
                    v[out(i, j, k)] = 0.5
                        * match ityp {
                            6 | 7 => vo[work(i, j, k)],
                            _ => ve[work(i, j, k)],
                        };
                    w[out(i, j, k)] = 0.5
                        * match ityp {
                            3 | 5 => wo[work(i, j, k)],
                            _ => we[work(i, j, k)],
                        };
                }
            }
        }
    }
    if mlat != 0 {
        for k in 0..nt {
            for j in 1..=nlon {
                v[out(imid, j, k)] = 0.5
                    * match ityp {
                        6 | 7 => vo[work(imid, j, k)],
                        _ => ve[work(imid, j, k)],
                    };
                w[out(imid, j, k)] = 0.5
                    * match ityp {
                        3 | 5 => wo[work(imid, j, k)],
                        _ => we[work(imid, j, k)],
                    };
            }
        }
    }

    Ok((v, w, 0))
}

#[pyfunction]
pub fn vhsec<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
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
            "vhsec expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let brbuf = br.as_slice()?.to_vec();
    let bibuf = bi.as_slice()?.to_vec();
    let crbuf = cr.as_slice()?.to_vec();
    let cibuf = ci.as_slice()?.to_vec();
    let wbuf = wvhsec.as_slice()?.to_vec();
    let (v, w, ierror) = py
        .detach(|| {
            vhsec_impl(&brbuf, &bibuf, &crbuf, &cibuf, nlat, nt, 0, &wbuf, lwork)
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
pub fn vhsec_ityp<'py>(
    py: Python<'py>,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
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
            "vhsec_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let (v, w, ierror) = vhsec_impl(
        br.as_slice()?,
        bi.as_slice()?,
        cr.as_slice()?,
        ci.as_slice()?,
        nlat,
        nt,
        ityp,
        wvhsec.as_slice()?,
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
