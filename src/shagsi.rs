use crate::gaussian_stored_scalar::shagsi_core;
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn shagsi_impl(nlat: i32, nlon: i32, lshags: i32, lwork: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 4 {
        return (Vec::new(), ierror);
    }
    let l = ((nlon + 2) / 2).min(nlat);
    let late = (nlat + 1) / 2;
    let l1 = l;
    let l2 = late;
    ierror = 3;
    let lp =
        nlat * (3 * (l1 + l2) - 2) + (l1 - 1) * (l2 * (2 * nlat - l1) - 3 * l1) / 2 + nlon + 15;
    if lshags < lp {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if lwork < 4 * nlat * (nlat + 2) + 2 {
        return (Vec::new(), ierror);
    }
    ierror = 5;
    if ldwork < nlat * (nlat + 4) {
        return (Vec::new(), ierror);
    }

    match shagsi_core(nlat as usize, nlon as usize) {
        Ok(w) => (w, 0),
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
pub fn shagsi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshags: i32,
    lwork: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshags, ierror) = shagsi_impl(nlat, nlon, lshags, lwork, ldwork);
    Ok((PyArray1::from_vec(py, wshags).to_owned(), ierror))
}
