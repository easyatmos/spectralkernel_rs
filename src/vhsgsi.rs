use crate::gaussian_stored_vector::vhsgsi_core;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Initialize the workspace required by `vhsgs_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsgs`: Declared length of the `wvhsgs` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhsgsi_impl(nlat: i32, nlon: i32, lvhsgs: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let lmn = nlat * (nlat + 1) / 2;
    if lvhsgs < 2 * (imid * lmn) + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < (nlat * 3 * (nlat + 3) + 2) / 2 {
        return (Vec::new(), ierror);
    }

    match vhsgsi_core(nlat as usize, nlon as usize) {
        Ok(w) => {
            let mut out = vec![0.0_f32; usize::try_from(lvhsgs).unwrap_or(0)];
            let copy_len = out.len().min(w.len());
            out[..copy_len].copy_from_slice(&w[..copy_len]);
            (out, 0)
        }
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
/// Python wrapper for `vhsgsi_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsgs`: Declared length of the `wvhsgs` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vhsgsi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhsgs: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhsgs, ierror) = vhsgsi_impl(nlat, nlon, lvhsgs, ldwork);
    Ok((PyArray1::from_vec(py, wvhsgs).to_owned(), ierror))
}
