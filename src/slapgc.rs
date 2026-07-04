use crate::shsgc::shsgc_impl;
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

pub fn slapgc_impl(
    nlon: usize,
    a: &[f32],
    b: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshsgc: &[f32],
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
    if a.len() != b.len() || a.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("a/b size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = imid;
    let lwmin_i64 = (nlat as i64) * (2 * l2 as i64 + 3 * l1 as i64 - 2)
        + (3 * l1 as i64 * (1 - l1 as i64)) / 2
        + nlon as i64
        + 15;
    let lwmin = usize::try_from(lwmin_i64)
        .map_err(|_| PyValueError::new_err("invalid lshsgc minimum length computation"))?;
    ierror = 9;
    if wshsgc.len() < lwmin {
        return Ok((Vec::new(), ierror));
    }

    let mmax = nlat.min(nlon / 2 + 1);
    let ls = if isym == 0 { nlat } else { imid };
    let lwkmin = if isym == 0 {
        nlat * (2 * nt * nlon + (6 * l2).max(nlon) + 2 * nt * mmax + 1)
    } else {
        l2 * (2 * nt * nlon + (6 * nlat).max(nlon)) + nlat * (2 * nt * mmax + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), ierror));
    }

    let mut alap = vec![0.0_f32; nlat * nlat * nt];
    let mut blap = vec![0.0_f32; nlat * nlat * nt];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                f * (f + 1.0)
            }
        })
        .collect();

    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            alap[idx] = -fnn[n - 1] * a[idx];
            blap[idx] = -fnn[n - 1] * b[idx];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                alap[idx] = -fnn[n - 1] * a[idx];
                blap[idx] = -fnn[n - 1] * b[idx];
            }
        }
    }

    let lwork_shsgc = if isym == 0 {
        nlat * nlon * (nt + 1)
    } else {
        ls * nlon * nt + nlat * nlon
    };
    shsgc_impl(&alap, &blap, nlat, nt, wshsgc, lwork_shsgc)
}

#[pyfunction]
pub fn slapgc<'py>(
    py: Python<'py>,
    nlon: usize,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
    wshsgc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    isym: Option<usize>,
) -> PyResult<(Py<PyAny>, i32)> {
    let ashape = a.shape().to_vec();
    let bshape = b.shape().to_vec();
    if ashape != bshape {
        return Err(PyValueError::new_err("a and b must have identical shapes"));
    }
    if ashape.len() != 2 && ashape.len() != 3 {
        return Err(PyValueError::new_err("slapgc expects rank-2 or rank-3 a/b"));
    }

    let nlat = ashape[0];
    let nt = if ashape.len() == 2 { 1 } else { ashape[2] };
    let isym = isym.unwrap_or(0);
    let abuf = collect_logical_ab(a.as_array());
    let bbuf = collect_logical_ab(b.as_array());
    let (slap, ierror) = slapgc_impl(
        nlon,
        &abuf,
        &bbuf,
        nlat,
        isym,
        nt,
        wshsgc.as_slice()?,
        lwork,
    )?;
    let ids = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if ashape.len() == 2 {
        IxDyn(&[ids, nlon])
    } else {
        IxDyn(&[ids, nlon, nt])
    };
    let slap_arr =
        ArrayD::from_shape_vec(shape, slap).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((slap_arr.into_pyarray(py).into_any().unbind(), ierror))
}
