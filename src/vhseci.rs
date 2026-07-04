use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{vbinit_impl, wbinit_impl};
use numpy::PyArray1;
use pyo3::prelude::*;

/// Initialize the workspace required by `vhsec_impl`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsec`: Declared length of the `wvhsec` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A tuple containing the workspace/result vector and the error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhseci_impl(nlat: i32, nlon: i32, lvhsec: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let lwvbin = lzz1 + labc;
    if lvhsec < 2 * lwvbin + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < 2 * nlat + 2 {
        return (Vec::new(), ierror);
    }

    ierror = 0;
    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let lvhsec_usize = usize::try_from(lvhsec).unwrap_or(0);
    let lwvbin_usize = usize::try_from(lwvbin).unwrap_or(0);

    let mut wvhsec = vec![0.0_f32; lvhsec_usize];
    let wvbin = vbinit_impl(nlat_usize, nlon_usize);
    for (idx, value) in wvbin.iter().enumerate() {
        if idx < wvhsec.len() {
            wvhsec[idx] = *value as f32;
        }
    }
    let wwbin = wbinit_impl(nlat_usize, nlon_usize);
    for (idx, value) in wwbin.iter().enumerate() {
        let dst = lwvbin_usize + idx;
        if dst < wvhsec.len() {
            wvhsec[dst] = *value as f32;
        }
    }
    let fft = hrffti_impl(nlon);
    for (idx, value) in fft.iter().enumerate() {
        let dst = 2 * lwvbin_usize + idx;
        if dst < wvhsec.len() {
            wvhsec[dst] = *value;
        }
    }

    (wvhsec, ierror)
}

#[pyfunction]
/// Python wrapper for `vhseci_impl` that returns the initialized workspace.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `lvhsec`: Declared length of the `wvhsec` workspace.
/// - `ldwork`: Length of the auxiliary workspace expected by the low-level interface.
///
/// # Returns
/// A one-dimensional NumPy workspace array together with a error code.
pub fn vhseci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhsec: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhsec, ierror) = vhseci_impl(nlat, nlon, lvhsec, ldwork);
    Ok((PyArray1::from_vec(py, wvhsec).to_owned(), ierror))
}
