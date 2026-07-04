use crate::shsec::shsec_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_ab(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
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

fn infer_nlon_from_wshsec(nlat: usize, ltotal: usize) -> PyResult<usize> {
    let l2 = (nlat + 1) / 2;
    for cand in 4..=4 * nlat.max(4) {
        let l1 = nlat.min((cand + 2) / 2);
        let need = 2 * nlat * l2 + 3 * (l1.saturating_sub(2) * (2 * nlat - l1 - 1)) / 2 + cand + 15;
        if need == ltotal {
            return Ok(cand);
        }
    }
    Err(PyValueError::new_err(
        "failed to infer nlon from wshsec length",
    ))
}

pub fn vrtec_impl(
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    wshsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }

    let nlon = infer_nlon_from_wshsec(nlat, wshsec.len())?;

    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if cr.len() != ci.len() || cr.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("cr/ci size mismatch"));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let fnn = idx as f32;
                (fnn * (fnn + 1.0)).sqrt()
            }
        })
        .collect();

    let mmax = nlat.min((nlon + 1) / 2);
    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            a[idx] = sqnn[n - 1] * cr[idx];
            b[idx] = sqnn[n - 1] * ci[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = sqnn[n - 1] * cr[idx];
                b[idx] = sqnn[n - 1] * ci[idx];
            }
        }
    }

    shsec_impl(&a, &b, nlat, nt, 0, wshsec, lwork)
}

#[pyfunction]
pub fn vrtec<'py>(
    py: Python<'py>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, i32)> {
    let crshape = cr.shape().to_vec();
    let cishape = ci.shape().to_vec();
    if crshape != cishape {
        return Err(PyValueError::new_err(
            "cr and ci must have identical shapes",
        ));
    }
    if crshape.len() != 2 && crshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vrtec expects rank-2 or rank-3 cr/ci",
        ));
    }

    let nlat = crshape[0];
    let nt = if crshape.len() == 2 { 1 } else { crshape[2] };
    let crbuf = collect_logical_ab(cr.as_array());
    let cibuf = collect_logical_ab(ci.as_array());
    let (vort, ierror) = vrtec_impl(&crbuf, &cibuf, nlat, nt, wshsec.as_slice()?, lwork)?;
    let nlon = infer_nlon_from_wshsec(nlat, wshsec.len())?;
    let shape = if crshape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let vort_arr =
        ArrayD::from_shape_vec(shape, vort).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((vort_arr.into_pyarray(py).into_any().unbind(), ierror))
}
