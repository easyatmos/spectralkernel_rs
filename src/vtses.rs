use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{dvtk, dvtt, dwtk, dwtt};
use crate::vtsgs::vtsgs_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn direct_vt_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    for np1 in (m + 1)..=nlat {
        let coeff = dvtk(m as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            out[(np1 - 1) * imid + (i - 1)] = dvtt(m as i32, (np1 - 1) as i32, th, &coeff) as f32;
        }
    }
    out
}

fn direct_wt_column(nlat: usize, m: usize) -> Vec<f32> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / (nlat.saturating_sub(1) as f64);
    let mut out = vec![0.0_f32; imid * nlat];
    let mw = m.max(1);
    for np1 in (mw + 1)..=nlat {
        let coeff = dwtk(mw as i32, (np1 - 1) as i32);
        for i in 1..=imid {
            let th = (i - 1) as f64 * dt;
            out[(np1 - 1) * imid + (i - 1)] = dwtt(mw as i32, (np1 - 1) as i32, th, &coeff) as f32;
        }
    }
    out
}

fn expand_vtses_output(
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

/// Core Rust implementation of `vtsesi`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lwvts`: Parameter `lwvts` passed through to the routine.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vtsesi_impl(nlat: i32, nlon: i32, lwvts: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let mmax = nlat.min((nlon + 1) / 2);
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lwvts < 2 * lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < 2 * (nlat + 1) {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let imid_usize = usize::try_from(imid).unwrap_or(0);
    let mmax_usize = usize::try_from(mmax).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);
    let out_len = usize::try_from(lwvts).unwrap_or(0);
    let mut out = vec![0.0_f32; out_len];

    for mp1 in 1..=mmax_usize {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_vt_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=imid_usize {
                let src = (np1 - 1) * imid_usize + (i - 1);
                let dst = (i - 1) + (mn - 1) * imid_usize;
                if dst < out_len {
                    out[dst] = col[src];
                }
            }
        }
    }

    for mp1 in 1..=mmax_usize {
        let m = mp1 - 1;
        let mb = if m == 0 {
            0
        } else {
            m * (nlat_usize - 1) - (m * (m - 1)) / 2
        };
        let col = direct_wt_column(nlat_usize, m);
        for np1 in mp1..=nlat_usize {
            let mn = mb + np1;
            for i in 1..=imid_usize {
                let src = (np1 - 1) * imid_usize + (i - 1);
                let dst = lzimn_usize + (i - 1) + (mn - 1) * imid_usize;
                if dst < out_len {
                    out[dst] = col[src];
                }
            }
        }
    }

    let fft = hrffti_impl(nlon);
    for (i, value) in fft.iter().enumerate() {
        let dst = 2 * lzimn_usize + i;
        if dst < out_len {
            out[dst] = *value;
        }
    }

    (out, 0)
}

#[pyfunction]
/// Rust entry point for `vtsesi`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lwvts`: Parameter `lwvts` passed through to the routine.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vtsesi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lwvts: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvts, ierror) = vtsesi_impl(nlat, nlon, lwvts, ldwork);
    Ok((PyArray1::from_vec(py, wvts).to_owned(), ierror))
}

#[pyfunction]
/// Rust entry point for `vtses`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `wvts`: Parameter `wvts` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vtses<'py>(
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
            "vtses expects rank-2 or rank-3 coefficient arrays",
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

    let vt = expand_vtses_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtses_output(wt, nlat, nlon, nt, idv);
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
/// Rust entry point for `vtses_ityp`.
///
/// # Parameters
/// - `br`: First vector coefficient family in vector layout.
/// - `bi`: Second vector coefficient family in vector layout.
/// - `cr`: Third vector coefficient family in vector layout.
/// - `ci`: Fourth vector coefficient family in vector layout.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvts`: Parameter `wvts` passed through to the routine.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Two NumPy arrays together with a error code.
pub fn vtses_ityp<'py>(
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
            "vtses_ityp expects rank-2 or rank-3 coefficient arrays",
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

    let vt = expand_vtses_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtses_output(wt, nlat, nlon, nt, idv);
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
