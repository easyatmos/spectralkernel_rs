use crate::shsgs::shsgs_impl;
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

pub fn vrtgs_impl(
    nlon: usize,
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshsgs: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if cr.len() != ci.len() || cr.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("cr/ci size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = imid;
    let lp_i64 = (nlat as i64) * (3 * (l1 as i64 + l2 as i64) - 2)
        + ((l1 as i64 - 1) * (l2 as i64 * (2 * nlat as i64 - l1 as i64) - 3 * l1 as i64)) / 2
        + nlon as i64
        + 15;
    let lp = usize::try_from(lp_i64)
        .map_err(|_| PyValueError::new_err("invalid lshsgs minimum length computation"))?;
    ierror = 9;
    if wshsgs.len() < lp {
        return Ok((Vec::new(), ierror));
    }

    let ls = if isym == 0 { nlat } else { imid };
    let mab = nlat.min(nlon / 2 + 1);
    let mn = mab * nlat * nt;
    let lwkmin = ls * nlon * (nt + 1) + 2 * mn + nlat;
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), ierror));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                ((idx as f32) * (idx as f32 + 1.0)).sqrt()
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

    let lwork_shsgs = if isym == 0 {
        nlat * nlon * (nt + 1)
    } else {
        ls * nlon * (nt + 1)
    };
    shsgs_impl(&a, &b, nlat, nt, wshsgs, lwork_shsgs)
}

#[pyfunction]
pub fn vrtgs<'py>(
    py: Python<'py>,
    nlon: usize,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wshsgs: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    isym: Option<usize>,
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
            "vrtgs expects rank-2 or rank-3 cr/ci",
        ));
    }

    let nlat = crshape[0];
    let nt = if crshape.len() == 2 { 1 } else { crshape[2] };
    let isym = isym.unwrap_or(0);
    let crbuf = collect_logical_ab(cr.as_array());
    let cibuf = collect_logical_ab(ci.as_array());
    let (vort, ierror) = vrtgs_impl(
        nlon,
        &crbuf,
        &cibuf,
        nlat,
        isym,
        nt,
        wshsgs.as_slice()?,
        lwork,
    )?;
    let ivrt = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if crshape.len() == 2 {
        IxDyn(&[ivrt, nlon])
    } else {
        IxDyn(&[ivrt, nlon, nt])
    };
    let vort_arr =
        ArrayD::from_shape_vec(shape, vort).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((vort_arr.into_pyarray(py).into_any().unbind(), ierror))
}
