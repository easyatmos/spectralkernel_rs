use crate::gaussian_stored_vector::vhagsi_core;
use crate::vhags::{vhags_impl_latpar, vhags_impl_parallel};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_vw(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
    match view.ndim() {
        2 => {
            let nlat = view.shape()[0];
            let nlon = view.shape()[1];
            let mut out = Vec::with_capacity(nlat * nlon);
            for i in 0..nlat {
                for j in 0..nlon {
                    out.push(view[[i, j]]);
                }
            }
            out
        }
        3 => {
            let nlat = view.shape()[0];
            let nlon = view.shape()[1];
            let nt = view.shape()[2];
            let mut out = Vec::with_capacity(nlat * nlon * nt);
            for i in 0..nlat {
                for j in 0..nlon {
                    for k in 0..nt {
                        out.push(view[[i, j, k]]);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

pub fn vhagc_impl(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhagc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let need_lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15;
    ierror = 9;
    if wvhagc.len() < need_lvhagc {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    ierror = 10;
    let min_lwork = if ityp <= 2 {
        nlat * (4 * nlon * nt + 6 * imid)
    } else {
        imid * (4 * nlon * nt + 6 * nlat)
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let stored_init = vhagsi_core(nlat, nlon).map_err(|ierr| {
        PyValueError::new_err(format!("vhagc internal init failed with ierror={ierr}"))
    })?;

    vhags_impl_parallel(v, w, nlat, nlon, nt, ityp, &stored_init, lwork)
}

pub fn vhagc_impl_latpar(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhagc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let need_lvhagc = 2 * (lzz1 + labc) + nlon + imid + 15;
    ierror = 9;
    if wvhagc.len() < need_lvhagc {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    ierror = 10;
    let min_lwork = if ityp <= 2 {
        nlat * (4 * nlon * nt + 6 * imid)
    } else {
        imid * (4 * nlon * nt + 6 * nlat)
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let stored_init = vhagsi_core(nlat, nlon).map_err(|ierr| {
        PyValueError::new_err(format!("vhagc internal init failed with ierror={ierr}"))
    })?;

    vhags_impl_latpar(v, w, nlat, nlon, nt, ityp, &stored_init, lwork)
}

#[pyfunction]
pub fn vhagc<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhagc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err("vhagc expects rank-2 or rank-3 v/w"));
    }

    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());
    let wvbuf = wvhagc.as_slice()?.to_vec();
    let (br, bi, cr, ci, ierror) = py
        .detach(|| {
            vhagc_impl(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;

    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let br_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), br)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bi_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), bi)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), cr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ci_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), ci)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        br_arr.into_pyarray(py).into_any().unbind(),
        bi_arr.into_pyarray(py).into_any().unbind(),
        cr_arr.into_pyarray(py).into_any().unbind(),
        ci_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
pub fn vhagc_ityp<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhagc: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhagc_ityp expects rank-2 or rank-3 v/w",
        ));
    }

    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());
    let (br, bi, cr, ci, ierror) = vhagc_impl(
        &vbuf,
        &wbuf,
        nlat,
        nlon,
        nt,
        ityp,
        wvhagc.as_slice()?,
        lwork,
    )?;

    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let br_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), br)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bi_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), bi)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), cr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ci_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), ci)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        br_arr.into_pyarray(py).into_any().unbind(),
        bi_arr.into_pyarray(py).into_any().unbind(),
        cr_arr.into_pyarray(py).into_any().unbind(),
        ci_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
