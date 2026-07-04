use crate::gaussian_computed_vector::vhsgci_core;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Initialize the workspace required by `vhsgc_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsgc`: Declared length of the `wvhsgc` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhsgci_impl(nlat: i32, nlon: i32, lvhsgc: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    if lvhsgc < 2 * (lzz1 + labc) + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < 2 * nlat * (nlat + 1) + 1 {
        return (Vec::new(), ierror);
    }
    match vhsgci_core(nlat as usize, nlon as usize) {
        Ok(w) => (w, 0),
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
/// Python wrapper for `vhsgci_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsgc`: Declared length of the `wvhsgc` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vhsgci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhsgc: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhsgc, ierror) = vhsgci_impl(nlat, nlon, lvhsgc, ldwork);
    Ok((PyArray1::from_vec(py, wvhsgc).to_owned(), ierror))
}
