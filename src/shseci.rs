use crate::alinit::alinit_impl;
use crate::hrffti::hrffti_impl;
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn shseci_impl(nlat: i32, nlon: i32, lshsec: i32, ldwork: i32) -> (Vec<f32>, i32) {
    let mut ierror = 1;
    if nlat < 3 {
        return (Vec::new(), ierror);
    }
    ierror = 2;
    if nlon < 4 {
        return (Vec::new(), ierror);
    }
    ierror = 3;
    let imid = (nlat + 1) / 2;
    let mmax = nlat.min(nlon / 2 + 1);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * ((mmax - 2) * (nlat + nlat - mmax - 1)) / 2;
    if lshsec < lzz1 + labc + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < nlat + 1 {
        return (Vec::new(), ierror);
    }

    ierror = 0;
    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let lshsec_usize = usize::try_from(lshsec).unwrap_or(0);
    let iw1 = usize::try_from(lzz1 + labc).unwrap_or(0);

    let mut wshsec = vec![0.0_f32; lshsec_usize];
    let walin = alinit_impl(nlat_usize, nlon_usize);
    for (idx, value) in walin.iter().enumerate() {
        if idx < wshsec.len() {
            wshsec[idx] = *value as f32;
        }
    }
    let fft = hrffti_impl(nlon);
    for (idx, value) in fft.iter().enumerate() {
        let dst = iw1 + idx;
        if dst < wshsec.len() {
            wshsec[dst] = *value;
        }
    }

    (wshsec, ierror)
}

#[pyfunction]
pub fn shseci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshsec: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshsec, ierror) = shseci_impl(nlat, nlon, lshsec, ldwork);
    Ok((PyArray1::from_vec(py, wshsec).to_owned(), ierror))
}
