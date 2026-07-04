use crate::gaussian_stored_vector::vhagsi_core;
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn vhagsi_impl(nlat: i32, nlon: i32, lvhags: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    if lvhags < 2 * (imid * lmn) + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < (nlat * (3 * nlat + 9) + 2) / 2 {
        return (Vec::new(), ierror);
    }

    match vhagsi_core(nlat as usize, nlon as usize) {
        Ok(w) => {
            let mut out = vec![0.0_f32; usize::try_from(lvhags).unwrap_or(0)];
            let copy_len = out.len().min(w.len());
            out[..copy_len].copy_from_slice(&w[..copy_len]);
            (out, 0)
        }
        Err(ierr) => (Vec::new(), ierr),
    }
}

#[pyfunction]
pub fn vhagsi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhags: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhags, ierror) = vhagsi_impl(nlat, nlon, lvhags, ldwork);
    Ok((PyArray1::from_vec(py, wvhags).to_owned(), ierror))
}
