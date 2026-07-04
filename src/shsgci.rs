use crate::gaussian_computed_scalar::shsgci_core;
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn shsgci_impl(nlat: i32, nlon: i32, lshsgc: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    if lshsgc < need {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < nlat * (nlat + 4) {
        return (Vec::new(), ierror);
    }
    match shsgci_core(nlat as usize, nlon as usize) {
        Ok(w) => (w, 0),
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
pub fn shsgci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshsgc: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshsgc, ierror) = shsgci_impl(nlat, nlon, lshsgc, ldwork);
    Ok((PyArray1::from_vec(py, wshsgc).to_owned(), ierror))
}
