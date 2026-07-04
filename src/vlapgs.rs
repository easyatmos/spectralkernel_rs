use crate::vhsgs::vhsgs_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_coeffs(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
    match view.ndim() {
        2 => {
            let nlat = view.shape()[0];
            let mut out = Vec::with_capacity(nlat * nlat);
            for m in 0..nlat {
                for n in 0..nlat {
                    out.push(view[[m, n]]);
                }
            }
            out
        }
        3 => {
            let nlat = view.shape()[0];
            let nt = view.shape()[2];
            let mut out = Vec::with_capacity(nlat * nlat * nt);
            for m in 0..nlat {
                for n in 0..nlat {
                    for k in 0..nt {
                        out.push(view[[m, n, k]]);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

fn expand_vlapgs_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
) -> Vec<f32> {
    let idv = if ityp <= 2 { nlat } else { nlat.div_ceil(2) };
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

pub fn vlapgs_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhsgs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let expected = nlat * nlat * nt;
    if br.len() != expected || bi.len() != expected || cr.len() != expected || ci.len() != expected
    {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let idz = (mmax * (2 * nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    let lsavmin = 2 * lzimn + nlon + 15;
    ierror = 9;
    if wvhsgs.len() < lsavmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mn = mmax * nlat * nt;
    let lwkmin = if ityp <= 2 {
        (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * mmax + 1)
    } else {
        (2 * nt + 1) * imid * nlon + nlat * (4 * nt * mmax + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut brlap = vec![0.0_f32; expected];
    let mut bilap = vec![0.0_f32; expected];
    let mut crlap = vec![0.0_f32; expected];
    let mut cilap = vec![0.0_f32; expected];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                -f * (f + 1.0)
            }
        })
        .collect();

    match ityp {
        0 | 3 | 6 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brlap[idx] = fnn[n - 1] * br[idx];
                    bilap[idx] = fnn[n - 1] * bi[idx];
                    crlap[idx] = fnn[n - 1] * cr[idx];
                    cilap[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brlap[idx] = fnn[n - 1] * br[idx];
                        bilap[idx] = fnn[n - 1] * bi[idx];
                        crlap[idx] = fnn[n - 1] * cr[idx];
                        cilap[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
        1 | 4 | 7 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brlap[idx] = fnn[n - 1] * br[idx];
                    bilap[idx] = fnn[n - 1] * bi[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brlap[idx] = fnn[n - 1] * br[idx];
                        bilap[idx] = fnn[n - 1] * bi[idx];
                    }
                }
            }
        }
        _ => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    crlap[idx] = fnn[n - 1] * cr[idx];
                    cilap[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        crlap[idx] = fnn[n - 1] * cr[idx];
                        cilap[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
    }

    let lwork_vhsgs = lwork.saturating_sub(4 * mn + nlat);
    let (vlap, wlap, _idvw, _nlon2, ierr) = vhsgs_impl(
        &brlap,
        &bilap,
        &crlap,
        &cilap,
        nlat,
        nt,
        ityp,
        wvhsgs,
        lwork_vhsgs,
    )?;
    Ok((vlap, wlap, ierr))
}

#[pyfunction]
pub fn vlapgs<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = br.shape().to_vec();
    if shape != bi.shape() || shape != cr.shape() || shape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "vlapgs expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = vlapgs_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        0,
        wvhsgs.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.clone().into_pyarray(py).into_any().unbind(),
            empty.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let v = expand_vlapgs_output(v, nlat, nlon, nt, 0);
    let w = expand_vlapgs_output(w, nlat, nlon, nt, 0);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
pub fn vlapgs_ityp<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = br.shape().to_vec();
    if shape != bi.shape() || shape != cr.shape() || shape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "vlapgs_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = vlapgs_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        ityp,
        wvhsgs.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.clone().into_pyarray(py).into_any().unbind(),
            empty.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let v = expand_vlapgs_output(v, nlat, nlon, nt, ityp);
    let w = expand_vlapgs_output(w, nlat, nlon, nt, ityp);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
