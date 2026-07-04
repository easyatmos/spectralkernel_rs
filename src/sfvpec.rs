use crate::shsec::shsec_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_coeffs(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
    match view.ndim() {
        2 => {
            let nlat = view.shape()[0];
            let mut out = Vec::with_capacity(nlat * nlat);
            for m in 0..nlat {
                for n in 0..nlat {
                    out.push(view[[m, n]]);
                }
            }
            out
        }
        3 => {
            let nlat = view.shape()[0];
            let nt = view.shape()[2];
            let mut out = Vec::with_capacity(nlat * nlat * nt);
            for m in 0..nlat {
                for n in 0..nlat {
                    for k in 0..nt {
                        out.push(view[[m, n, k]]);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

pub fn sfvpec_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wshsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let expected = nlat * nlat * nt;
    if br.len() != expected || bi.len() != expected || cr.len() != expected || ci.len() != expected
    {
        return Err(PyValueError::new_err("br/bi/cr/ci size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 2) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * mmax.saturating_sub(2) * (2 * nlat - mmax - 1) / 2;
    let lshsec_min = lzz1 + labc + nlon + 15;
    ierror = 9;
    if wshsec.len() < lshsec_min {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let ls = if isym == 0 { nlat } else { imid };
    let mab = nlat.min(nlon / 2 + 1);
    let mn = mab * nlat * nt;
    let lwkmin = nt * ls * nlon + (ls * nlon).max(3 * nlat * imid) + 2 * mn + nlat;
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut a = vec![0.0_f32; expected];
    let mut b = vec![0.0_f32; expected];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                1.0 / ((idx as f32) * ((idx + 1) as f32)).sqrt()
            }
        })
        .collect();

    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            a[idx] = -fnn[n - 1] * cr[idx];
            b[idx] = -fnn[n - 1] * ci[idx];
        }
        for m in 2..=mab.min(nlat) {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = -fnn[n - 1] * cr[idx];
                b[idx] = -fnn[n - 1] * ci[idx];
            }
        }
    }
    let sh_lwork = lwork.saturating_sub(2 * mn + nlat);
    let sh_lwork_eff = sh_lwork.max(nlat * nlon * (nt + 1));
    let (sf_raw, ierr_sf) = shsec_impl(&a, &b, nlat, nt, isym, wshsec, sh_lwork_eff)?;
    if ierr_sf != 0 {
        return Ok((Vec::new(), Vec::new(), ierr_sf));
    }

    let sf = if isym == 0 {
        sf_raw
    } else {
        let mut out = vec![0.0_f32; nlat * nlon * nt];
        let imid = nlat.div_ceil(2);
        for k in 0..nt {
            for i in 0..imid {
                for j in 0..nlon {
                    let src = (i * nlon + j) * nt + k;
                    let dst = (i * nlon + j) * nt + k;
                    out[dst] = sf_raw[src];
                }
            }
        }
        out
    };

    a.fill(0.0);
    b.fill(0.0);
    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            a[idx] = fnn[n - 1] * br[idx];
            b[idx] = fnn[n - 1] * bi[idx];
        }
        for m in 2..=mab.min(nlat) {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = fnn[n - 1] * br[idx];
                b[idx] = fnn[n - 1] * bi[idx];
            }
        }
    }
    let (vp_raw, ierr_vp) = shsec_impl(&a, &b, nlat, nt, isym, wshsec, sh_lwork_eff)?;
    if ierr_vp != 0 {
        return Ok((Vec::new(), Vec::new(), ierr_vp));
    }
    let vp = if isym == 0 {
        vp_raw
    } else {
        let mut out = vec![0.0_f32; nlat * nlon * nt];
        let imid = nlat.div_ceil(2);
        for k in 0..nt {
            for i in 0..imid {
                for j in 0..nlon {
                    let src = (i * nlon + j) * nt + k;
                    let dst = (i * nlon + j) * nt + k;
                    out[dst] = vp_raw[src];
                }
            }
        }
        out
    };
    Ok((sf, vp, 0))
}

#[pyfunction]
pub fn sfvpec<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    sfvpec_isym(py, nlon, 0, br, bi, cr, ci, wshsec, lwork)
}

#[pyfunction]
pub fn sfvpec_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    isym: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = br.shape().to_vec();
    if shape != bi.shape() || shape != cr.shape() || shape != ci.shape() {
        return Err(PyValueError::new_err(
            "br/bi/cr/ci must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "sfvpec expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let idv = nlat;
    let (sf, vp, ierror) = sfvpec_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        isym,
        wshsec.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty.clone().into_pyarray(py).into_any().unbind(),
            empty.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let out_shape = if shape.len() == 2 {
        IxDyn(&[idv, nlon])
    } else {
        IxDyn(&[idv, nlon, nt])
    };
    let sf_arr = ArrayD::from_shape_vec(out_shape.clone(), sf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vp_arr =
        ArrayD::from_shape_vec(out_shape, vp).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        sf_arr.into_pyarray(py).into_any().unbind(),
        vp_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
