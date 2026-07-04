use crate::hrffti::hrffti_impl;
use crate::sphcom_vector::{zvinit_impl, zwinit_impl};
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn vhaeci_impl(nlat: i32, nlon: i32, lvhaec: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let lwzvin = lzz1 + labc;
    if lvhaec < 2 * lwzvin + nlon + 15 {
        return (Vec::new(), ierror);
    }
    ierror = 4;
    if ldwork < 2 * nlat + 2 {
        return (Vec::new(), ierror);
    }

    ierror = 0;
    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let lvhaec_usize = usize::try_from(lvhaec).unwrap_or(0);
    let lwzvin_usize = usize::try_from(lwzvin).unwrap_or(0);

    let mut wvhaec = vec![0.0_f32; lvhaec_usize];
    let wzvin = zvinit_impl(nlat_usize, nlon_usize);
    for (idx, value) in wzvin.iter().enumerate() {
        if idx < wvhaec.len() {
            wvhaec[idx] = *value as f32;
        }
    }
    let wzwin = zwinit_impl(nlat_usize, nlon_usize);
    for (idx, value) in wzwin.iter().enumerate() {
        let dst = lwzvin_usize + idx;
        if dst < wvhaec.len() {
            wvhaec[dst] = *value as f32;
        }
    }
    let fft = hrffti_impl(nlon);
    for (idx, value) in fft.iter().enumerate() {
        let dst = 2 * lwzvin_usize + idx;
        if dst < wvhaec.len() {
            wvhaec[dst] = *value;
        }
    }

    (wvhaec, ierror)
}

#[pyfunction]
pub fn vhaeci<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lvhaec: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wvhaec, ierror) = vhaeci_impl(nlat, nlon, lvhaec, ldwork);
    Ok((PyArray1::from_vec(py, wvhaec).to_owned(), ierror))
}
