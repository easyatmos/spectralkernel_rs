use crate::alinit::ses1_impl;
use crate::hrffti::hrffti_impl;
use numpy::PyArray1;
use pyo3::prelude::*;

pub fn shsesi_impl(nlat: i32, nlon: i32, lshses: i32, lwork: i32, ldwork: i32) -> (Vec<f32>, i32) {
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
    let lpimn = (imid * mmax * (nlat + nlat - mmax + 1)) / 2;
    if lshses < lpimn + nlon + 15 {
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

    let nlat_usize = usize::try_from(nlat).unwrap_or(0);
    let nlon_usize = usize::try_from(nlon).unwrap_or(0);
    let mut out = vec![0.0_f32; usize::try_from(lshses).unwrap_or(0)];
    let p = ses1_impl(nlat_usize, nlon_usize);
    for (i, v) in p.iter().enumerate() {
        if i < out.len() {
            out[i] = *v as f32;
        }
    }
    let fft = hrffti_impl(nlon);
    for (i, v) in fft.iter().enumerate() {
        let dst = usize::try_from(lpimn).unwrap_or(0) + i;
        if dst < out.len() {
            out[dst] = *v;
        }
    }
    (out, 0)
}

#[pyfunction]
pub fn shsesi<'py>(
    py: Python<'py>,
    nlat: i32,
    nlon: i32,
    lshses: i32,
    lwork: i32,
    ldwork: i32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, i32)> {
    let (wshses, ierror) = shsesi_impl(nlat, nlon, lshses, lwork, ldwork);
    Ok((PyArray1::from_vec(py, wshses).to_owned(), ierror))
}
