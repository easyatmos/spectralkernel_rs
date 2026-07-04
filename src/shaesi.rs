use crate::hrffti::hrffti_impl;
use crate::sea1::sea1_impl;
use numpy::PyArray1;
use pyo3::prelude::*;

/// Initialize the workspace required by `shaes_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshaes`: Declared length of the `wshaes` workspace.
/// - `lwork`: Length of the caller-provided work array.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shaesi_impl(nlat: i32, nlon: i32, lshaes: i32, lwork: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 4 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let mmax = nlat.min(nlon / 2 + 1);
    let imid = (nlat + 1) / 2;
    let lzimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lshaes < lzimn + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    let labc = 3 * ((mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lwork < 5 * nlat * imid + labc {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < nlat + 1 {
        return (Vec::new(), ierror);
    }

    ierror = 0;
    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let lshaes_usize = usize::try_from(lshaes).unwrap_or(0);
    let lzimn_usize = usize::try_from(lzimn).unwrap_or(0);

    let mut wshaes = vec![0.0_f32; lshaes_usize];
    let z = sea1_impl(nlat_usize, nlon_usize);
    for (idx, value) in z.iter().enumerate() {
        wshaes[idx] = *value as f32;
    }
    let fft = hrffti_impl(nlon);
    for (idx, value) in fft.iter().enumerate() {
        let dst = lzimn_usize + idx;
        if dst < wshaes.len() {
            wshaes[dst] = *value;
        }
    }

    (wshaes, ierror)
}

#[pyfunction]
/// Python wrapper for `shaesi_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lshaes`: Declared length of the `wshaes` workspace.
/// - `lwork`: Length of the caller-provided work array.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn shaesi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshaes: i32,
    lwork: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshaes, ierror) = shaesi_impl(nlat, nlon, lshaes, lwork, ldwork);
    Ok((PyArray1::from_vec(py, wshaes).to_owned(), ierror))
}
