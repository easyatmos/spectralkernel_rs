use crate::shsec::shsec_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn collect_logical_ab(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
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

pub fn divec_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    nlat: usize,
    isym: usize,
    nt: usize,
    wshsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), ierror));
    }
    if br.len() != bi.len() || br.len() != nlat * nlat * nt {
        return Err(PyValueError::new_err("br/bi size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 2) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * (mmax.saturating_sub(2) * (2 * nlat - mmax - 1)) / 2;
    let lwmin = lzz1 + labc + nlon + 15;
    ierror = 9;
    if wshsec.len() < lwmin {
        return Ok((Vec::new(), ierror));
    }

    let l1 = nlat.min((nlon + 2) / 2);
    let l2 = imid;
    let lwkmin = if isym == 0 {
        nlat * (nt * nlon + (3 * l2).max(nlon) + 2 * nt * l1 + 1)
    } else {
        l2 * (nt * nlon + (3 * nlat).max(nlon)) + nlat * (2 * nt * l1 + 1)
    };
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), ierror));
    }

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let fnn = idx as f32;
                (fnn * (fnn + 1.0)).sqrt()
            }
        })
        .collect();

    let mmax_coeff = nlat.min((nlon + 1) / 2);
    for k in 0..nt {
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            a[idx] = -sqnn[n - 1] * br[idx];
            b[idx] = -sqnn[n - 1] * bi[idx];
        }
        for m in 2..=mmax_coeff {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                a[idx] = -sqnn[n - 1] * br[idx];
                b[idx] = -sqnn[n - 1] * bi[idx];
            }
        }
    }

    let _ = isym;
    shsec_impl(&a, &b, nlat, nt, isym, wshsec, lwork)
}

#[pyfunction]
pub fn divec<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    wshsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
    isym: Option<usize>,
) -> PyResult<(Py<PyAny>, i32)> {
    let brshape = br.shape().to_vec();
    let bishape = bi.shape().to_vec();
    if brshape != bishape {
        return Err(PyValueError::new_err(
            "br and bi must have identical shapes",
        ));
    }
    if brshape.len() != 2 && brshape.len() != 3 {
        return Err(PyValueError::new_err(
            "divec expects rank-2 or rank-3 br/bi",
        ));
    }

    let nlat = brshape[0];
    let nt = if brshape.len() == 2 { 1 } else { brshape[2] };
    let isym = isym.unwrap_or(0);
    let brbuf = collect_logical_ab(br.as_array());
    let bibuf = collect_logical_ab(bi.as_array());
    let (dv, ierror) = divec_impl(
        nlon,
        &brbuf,
        &bibuf,
        nlat,
        isym,
        nt,
        wshsec.as_slice()?,
        lwork,
    )?;
    let idv = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let shape = if brshape.len() == 2 {
        IxDyn(&[idv, nlon])
    } else {
        IxDyn(&[idv, nlon, nt])
    };
    let dv_arr =
        ArrayD::from_shape_vec(shape, dv).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((dv_arr.into_pyarray(py).into_any().unbind(), ierror))
}
