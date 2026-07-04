use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_ig(ig: i32) -> PyResult<()> {
    if ig == 0 || ig == 1 {
        Ok(())
    } else {
        Err(PyValueError::new_err("ig must be 0 or 1"))
    }
}

fn geo_index(nlat: usize, j: usize, i: usize) -> usize {
    j * nlat + i
}

fn math_index(nlon: usize, i: usize, j: usize) -> usize {
    i * nlon + j
}

pub fn geo2maths_impl(ig: i32, sg: &[f32], nlon: usize, nlat: usize) -> PyResult<Vec<f32>> {
    validate_ig(ig)?;
    if sg.len() != nlon * nlat {
        return Err(PyValueError::new_err("sg size mismatch"));
    }

    let mut sm = vec![0.0_f32; nlat * nlon];
    for i in 0..nlat {
        for j in 0..nlon {
            let src = geo_index(nlat, j, i);
            let dst_i = if ig == 0 { nlat - 1 - i } else { i };
            let dst = math_index(nlon, dst_i, j);
            sm[dst] = sg[src];
        }
    }
    Ok(sm)
}

pub fn math2geos_impl(ig: i32, sm: &[f32], nlat: usize, nlon: usize) -> PyResult<Vec<f32>> {
    validate_ig(ig)?;
    if sm.len() != nlat * nlon {
        return Err(PyValueError::new_err("sm size mismatch"));
    }

    let mut sg = vec![0.0_f32; nlon * nlat];
    for i in 0..nlat {
        for j in 0..nlon {
            let src = math_index(nlon, i, j);
            let dst_i = if ig == 0 { nlat - 1 - i } else { i };
            let dst = geo_index(nlat, j, dst_i);
            sg[dst] = sm[src];
        }
    }
    Ok(sg)
}

pub fn geo2mathv_impl(
    ig: i32,
    ug: &[f32],
    vg: &[f32],
    nlon: usize,
    nlat: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    validate_ig(ig)?;
    if ug.len() != nlon * nlat || vg.len() != nlon * nlat {
        return Err(PyValueError::new_err("ug/vg size mismatch"));
    }

    let mut vm = vec![0.0_f32; nlat * nlon];
    let mut wm = vec![0.0_f32; nlat * nlon];
    for i in 0..nlat {
        for j in 0..nlon {
            let src = geo_index(nlat, j, i);
            let dst_i = if ig == 0 { nlat - 1 - i } else { i };
            let dst = math_index(nlon, dst_i, j);
            vm[dst] = -vg[src];
            wm[dst] = ug[src];
        }
    }
    Ok((vm, wm))
}

pub fn math2geov_impl(
    ig: i32,
    vm: &[f32],
    wm: &[f32],
    nlat: usize,
    nlon: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    validate_ig(ig)?;
    if vm.len() != nlat * nlon || wm.len() != nlat * nlon {
        return Err(PyValueError::new_err("vm/wm size mismatch"));
    }

    let mut ug = vec![0.0_f32; nlon * nlat];
    let mut vg = vec![0.0_f32; nlon * nlat];
    for i in 0..nlat {
        for j in 0..nlon {
            let src = math_index(nlon, i, j);
            let dst_i = if ig == 0 { nlat - 1 - i } else { i };
            let dst = geo_index(nlat, j, dst_i);
            ug[dst] = wm[src];
            vg[dst] = -vm[src];
        }
    }
    Ok((ug, vg))
}

#[pyfunction]
#[pyo3(signature = (sg, ig=0))]
pub fn geo2maths<'py>(
    py: Python<'py>,
    sg: PyReadonlyArrayDyn<'py, f32>,
    ig: i32,
) -> PyResult<Py<PyAny>> {
    let shape = sg.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("geo2maths expects a rank-2 array"));
    }
    let nlon = shape[0];
    let nlat = shape[1];
    let sg_arr = sg.as_array().as_standard_layout().to_owned();
    let sm = geo2maths_impl(
        ig,
        sg_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("sg is not standard-layout after copy"))?,
        nlon,
        nlat,
    )?;
    let out = ArrayD::from_shape_vec(IxDyn(&[nlat, nlon]), sm)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (sm, ig=0))]
pub fn math2geos<'py>(
    py: Python<'py>,
    sm: PyReadonlyArrayDyn<'py, f32>,
    ig: i32,
) -> PyResult<Py<PyAny>> {
    let shape = sm.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("math2geos expects a rank-2 array"));
    }
    let nlat = shape[0];
    let nlon = shape[1];
    let sm_arr = sm.as_array().as_standard_layout().to_owned();
    let sg = math2geos_impl(
        ig,
        sm_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("sm is not standard-layout after copy"))?,
        nlat,
        nlon,
    )?;
    let out = ArrayD::from_shape_vec(IxDyn(&[nlon, nlat]), sg)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (ug, vg, ig=0))]
pub fn geo2mathv<'py>(
    py: Python<'py>,
    ug: PyReadonlyArrayDyn<'py, f32>,
    vg: PyReadonlyArrayDyn<'py, f32>,
    ig: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let ushape = ug.shape().to_vec();
    if ushape != vg.shape() {
        return Err(PyValueError::new_err(
            "ug and vg must have identical shapes",
        ));
    }
    if ushape.len() != 2 {
        return Err(PyValueError::new_err("geo2mathv expects rank-2 arrays"));
    }
    let nlon = ushape[0];
    let nlat = ushape[1];
    let ug_arr = ug.as_array().as_standard_layout().to_owned();
    let vg_arr = vg.as_array().as_standard_layout().to_owned();
    let (vm, wm) = geo2mathv_impl(
        ig,
        ug_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("ug is not standard-layout after copy"))?,
        vg_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("vg is not standard-layout after copy"))?,
        nlon,
        nlat,
    )?;
    let vm = ArrayD::from_shape_vec(IxDyn(&[nlat, nlon]), vm)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let wm = ArrayD::from_shape_vec(IxDyn(&[nlat, nlon]), wm)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    Ok((vm, wm))
}

#[pyfunction]
#[pyo3(signature = (vm, wm, ig=0))]
pub fn math2geov<'py>(
    py: Python<'py>,
    vm: PyReadonlyArrayDyn<'py, f32>,
    wm: PyReadonlyArrayDyn<'py, f32>,
    ig: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let vshape = vm.shape().to_vec();
    if vshape != wm.shape() {
        return Err(PyValueError::new_err(
            "vm and wm must have identical shapes",
        ));
    }
    if vshape.len() != 2 {
        return Err(PyValueError::new_err("math2geov expects rank-2 arrays"));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let vm_arr = vm.as_array().as_standard_layout().to_owned();
    let wm_arr = wm.as_array().as_standard_layout().to_owned();
    let (ug, vg) = math2geov_impl(
        ig,
        vm_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("vm is not standard-layout after copy"))?,
        wm_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("wm is not standard-layout after copy"))?,
        nlat,
        nlon,
    )?;
    let ug = ArrayD::from_shape_vec(IxDyn(&[nlon, nlat]), ug)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    let vg = ArrayD::from_shape_vec(IxDyn(&[nlon, nlat]), vg)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    Ok((ug, vg))
}
