use crate::gaussian_computed_scalar::shagci_core;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Initialize the workspace required by `shagc_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshagc`: Declared length of the `wshagc` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shagci_impl(nlat: i32, nlon: i32, lshagc: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 4 {
        return (Vec::new(), ierror);
    }
    let l = ((nlon + 2) / 2).min(nlat);
    let late = (nlat + (nlat % 2)) / 2;
    ierror = 3;
    let need = nlat * (2 * late + 3 * l - 2) + 3 * l * (1 - l) / 2 + nlon + 15;
    if lshagc < need {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < nlat * (nlat + 4) {
        return (Vec::new(), ierror);
    }
    match shagci_core(nlat as usize, nlon as usize) {
        Ok(w) => (w, 0),
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
/// Python wrapper for `shagci_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshagc`: Declared length of the `wshagc` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn shagci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshagc: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshagc, ierror) = shagci_impl(nlat, nlon, lshagc, ldwork);
    Ok((PyArray1::from_vec(py, wshagc).to_owned(), ierror))
}
