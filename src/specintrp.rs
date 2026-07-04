use num_complex::Complex32;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn specintrp_impl(rlon: f32, ntrunc: i32, datnm: &[Complex32], pnm: &[f32]) -> PyResult<f32> {
    if ntrunc < 0 {
        return Err(PyValueError::new_err("ntrunc must be non-negative"));
    }

    let ntrunc_usize = usize::try_from(ntrunc).unwrap_or(0);
    let expected_len = (ntrunc_usize + 1) * (ntrunc_usize + 2) / 2;
    if datnm.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "datnm length mismatch: expected {expected_len}, got {}",
            datnm.len()
        )));
    }
    if pnm.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "pnm length mismatch: expected {expected_len}, got {}",
            pnm.len()
        )));
    }

    let mwaves = ntrunc_usize + 1;
    let mut scrm = vec![Complex32::new(0.0_f32, 0.0_f32); mwaves];
    let mut nmstrt = 0_usize;

    for m in 1..=mwaves {
        scrm[m - 1] = Complex32::new(0.0_f32, 0.0_f32);
        for n in 1..=mwaves - m + 1 {
            let nm = nmstrt + n;
            scrm[m - 1] += datnm[nm - 1] * pnm[nm - 1];
        }
        nmstrt += mwaves - m + 1;
    }

    let mut ob = scrm[0].re;
    for m in 2..=mwaves {
        let angle = (m as f32 - 1.0_f32) * rlon;
        ob += 2.0_f32 * scrm[m - 1].re * angle.cos() - 2.0_f32 * scrm[m - 1].im * angle.sin();
    }

    Ok(ob)
}

#[pyfunction]
pub fn specintrp(
    rlon: f32,
    ntrunc: i32,
    datnm: PyReadonlyArray1<'_, Complex32>,
    pnm: PyReadonlyArray1<'_, f32>,
) -> PyResult<f32> {
    let datnm = datnm.as_slice()?;
    let pnm = pnm.as_slice()?;
    specintrp_impl(rlon, ntrunc, datnm, pnm)
}
