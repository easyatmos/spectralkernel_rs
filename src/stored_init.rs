use crate::shaesi::shaesi_impl;
use crate::shagsi::shagsi_impl;
use crate::shsesi::shsesi_impl;
use crate::shsgsi::shsgsi_impl;
use crate::vhaesi::vhaesi_impl_parallel;
use crate::vhagsi::vhagsi_impl;
use crate::vhsesi::vhsesi_impl_parallel;
use crate::vhsgsi::vhsgsi_impl;
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::join;

type RegularStoredInit = (Vec<f32>, i32, Vec<f32>, i32, Vec<f32>, i32, Vec<f32>, i32);
type GaussianStoredInit = (Vec<f32>, i32, Vec<f32>, i32, Vec<f32>, i32, Vec<f32>, i32);

fn regular_stored_init_impl(
    nlat: i32,
    nlon: i32,
    lshaes: i32,
    lshaes_work: i32,
    lvhaes: i32,
    lvhaes_work: i32,
    ldwork_scalar: i32,
    ldwork_vector: i32,
) -> RegularStoredInit {
    let ((wshaes, shaes_err), (wshses, shses_err)) = join(
        || shaesi_impl(nlat, nlon, lshaes, lshaes_work, ldwork_scalar),
        || shsesi_impl(nlat, nlon, lshaes, lshaes_work, ldwork_scalar),
    );
    let ((wvhaes, vhaes_err), (wvhses, vhses_err)) = join(
        || vhaesi_impl_parallel(nlat, nlon, lvhaes, lvhaes_work, ldwork_vector),
        || vhsesi_impl_parallel(nlat, nlon, lvhaes, lvhaes_work, ldwork_vector),
    );
    (
        wshaes, shaes_err, wshses, shses_err, wvhaes, vhaes_err, wvhses, vhses_err,
    )
}

fn gaussian_stored_init_impl(
    nlat: i32,
    nlon: i32,
    lshags: i32,
    lshags_work: i32,
    lshags_dwork: i32,
    lvhags: i32,
    lvhags_dwork: i32,
    lvhsgs: i32,
    lvhsgs_dwork: i32,
) -> GaussianStoredInit {
    let ((wshags, shags_err), (wshsgs, shsgs_err)) = join(
        || shagsi_impl(nlat, nlon, lshags, lshags_work, lshags_dwork),
        || shsgsi_impl(nlat, nlon, lshags, lshags_work, lshags_dwork),
    );
    let ((wvhags, vhags_err), (wvhsgs, vhsgs_err)) = join(
        || vhagsi_impl(nlat, nlon, lvhags, lvhags_dwork),
        || vhsgsi_impl(nlat, nlon, lvhsgs, lvhsgs_dwork),
    );
    (
        wshags, shags_err, wshsgs, shsgs_err, wvhags, vhags_err, wvhsgs, vhsgs_err,
    )
}

#[pyfunction]
/// Build regular-grid stored workspaces for scalar and vector transforms while releasing the GIL.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshaes`: Declared length of the `wshaes` workspace.
/// - `lshaes_work`: Requested storage for the scalar regular-grid stored-analysis workspace.
/// - `lvhaes`: Declared length of the `wvhaes` workspace.
/// - `lvhaes_work`: Requested storage for the vector regular-grid stored-analysis workspace.
/// - `ldwork_scalar`: Auxiliary workspace length used by the scalar stored initializer.
/// - `ldwork_vector`: Auxiliary workspace length used by the vector stored initializer.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn regular_stored_init_nogil<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshaes: i32,
    lshaes_work: i32,
    lvhaes: i32,
    lvhaes_work: i32,
    ldwork_scalar: i32,
    ldwork_vector: i32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
)> {
    let result = py.detach(move || {
        regular_stored_init_impl(
            nlat,
            nlon,
            lshaes,
            lshaes_work,
            lvhaes,
            lvhaes_work,
            ldwork_scalar,
            ldwork_vector,
        )
    });
    Ok((
        PyArray1::from_vec(py, result.0).to_owned(),
        result.1,
        PyArray1::from_vec(py, result.2).to_owned(),
        result.3,
        PyArray1::from_vec(py, result.4).to_owned(),
        result.5,
        PyArray1::from_vec(py, result.6).to_owned(),
        result.7,
    ))
}

#[pyfunction]
/// Build Gaussian-grid stored workspaces for scalar and vector transforms while releasing the GIL.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshags`: Declared length of the `wshags` workspace.
/// - `lshags_work`: Requested storage for the scalar Gaussian stored-analysis workspace.
/// - `lshags_dwork`: Auxiliary workspace length for the scalar Gaussian stored initializer.
/// - `lvhags`: Declared length of the `wvhags` workspace.
/// - `lvhags_dwork`: Auxiliary workspace length for the Gaussian vector analysis initializer.
/// - `lvhsgs`: Declared length of the `wvhsgs` workspace.
/// - `lvhsgs_dwork`: Auxiliary workspace length for the Gaussian vector synthesis initializer.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn gaussian_stored_init_nogil<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshags: i32,
    lshags_work: i32,
    lshags_dwork: i32,
    lvhags: i32,
    lvhags_dwork: i32,
    lvhsgs: i32,
    lvhsgs_dwork: i32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
    Bound<'py, PyArray1<f32>>,
    i32,
)> {
    let result = py.detach(move || {
        gaussian_stored_init_impl(
            nlat,
            nlon,
            lshags,
            lshags_work,
            lshags_dwork,
            lvhags,
            lvhags_dwork,
            lvhsgs,
            lvhsgs_dwork,
        )
    });
    Ok((
        PyArray1::from_vec(py, result.0).to_owned(),
        result.1,
        PyArray1::from_vec(py, result.2).to_owned(),
        result.3,
        PyArray1::from_vec(py, result.4).to_owned(),
        result.5,
        PyArray1::from_vec(py, result.6).to_owned(),
        result.7,
    ))
}
