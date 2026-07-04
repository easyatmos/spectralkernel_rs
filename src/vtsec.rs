use crate::gaussian_computed_vector::{vtgint_impl, wtgint_impl};
use crate::hrffti::hrffti_impl;
use crate::vtsgc::vtsgc_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn expand_vtsec_output(
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

fn equidistant_theta(nlat: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let dt = std::f64::consts::PI / ((nlat - 1) as f64);
    (0..imid).map(|i| (i as f64) * dt).collect()
}

pub fn vtseci_impl(nlat: i32, nlon: i32, lwvts: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let lzz1 = 2 * nlat * imid;
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = 3 * (0.max(mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwvts < 2 * (lzz1 + labc) + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < 2 * nlat + 2 {
        return (Vec::new(), ierror);
    }

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let theta = equidistant_theta(nlat_usize);
    let mut wvts = vtgint_impl(nlat_usize, nlon_usize, &theta);
    wvts.extend_from_slice(&wtgint_impl(nlat_usize, nlon_usize, &theta));
    wvts.extend_from_slice(&hrffti_impl(nlon));
    (wvts, 0)
}

#[pyfunction]
pub fn vtseci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lwvts: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvts, ierror) = vtseci_impl(nlat, nlon, lwvts, ldwork);
    Ok((PyArray1::from_vec(py, wvts).to_owned(), ierror))
}

#[pyfunction]
pub fn vtsec<'py>(
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
            "vtsec expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgc_impl(
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

    let vt = expand_vtsec_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsec_output(wt, nlat, nlon, nt, idv);
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
pub fn vtsec_ityp<'py>(
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
            "vtsec_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = bshape[0];
    let nt = if bshape.len() == 2 { 1 } else { bshape[2] };
    let br_arr = br.as_array().as_standard_layout().to_owned();
    let bi_arr = bi.as_array().as_standard_layout().to_owned();
    let cr_arr = cr.as_array().as_standard_layout().to_owned();
    let ci_arr = ci.as_array().as_standard_layout().to_owned();
    let wvts_arr = wvts.as_array().as_standard_layout().to_owned();
    let (vt, wt, idv, nlon, ierror) = vtsgc_impl(
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

    let vt = expand_vtsec_output(vt, nlat, nlon, nt, idv);
    let wt = expand_vtsec_output(wt, nlat, nlon, nt, idv);
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
