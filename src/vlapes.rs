use crate::vhses::vhses_impl;
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

pub fn vlapes_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhses: &[f32],
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

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let idz = (mmax * (2 * nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    let lwmin = 2 * lzimn + nlon + 15;
    ierror = 9;
    if wvhses.len() < lwmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let l1 = nlat.min(nlon / 2 + 1);
    let l2 = imid;
    let lwkmin = if ityp <= 2 {
        (2 * nt + 1) * nlat * nlon + nlat * (4 * nt * l1 + 1)
    } else {
        (2 * nt + 1) * l2 * nlon + nlat * (4 * nt * l1 + 1)
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
                -(idx as f32) * ((idx as f32) + 1.0)
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

    let mn = mmax * nlat * nt;
    let lwork_vhses = lwork.saturating_sub(4 * mn + nlat);
    vhses_impl(
        &brlap,
        &bilap,
        &crlap,
        &cilap,
        nlat,
        nlon,
        nt,
        ityp,
        wvhses,
        lwork_vhses,
    )
}

#[pyfunction]
pub fn vlapes<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    vlapes_ityp(py, nlon, br, bi, cr, ci, 0, wvhses, lwork)
}

#[pyfunction]
pub fn vlapes_ityp<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhses: PyReadonlyArrayDyn<'py, f32>,
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
            "vlapes expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (vlap, wlap, ierror) = vlapes_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        ityp,
        wvhses.as_slice()?,
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

    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let vlap_arr = ArrayD::from_shape_vec(out_shape.clone(), vlap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let wlap_arr = ArrayD::from_shape_vec(out_shape, wlap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        vlap_arr.into_pyarray(py).into_any().unbind(),
        wlap_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
