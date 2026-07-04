use crate::vhsec::vhsec_impl;
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

fn ityp_from_isym(isym: usize) -> PyResult<usize> {
    match isym {
        0 => Ok(0),
        1 => Ok(3),
        2 => Ok(6),
        _ => Err(PyValueError::new_err("isym must be 0, 1 or 2")),
    }
}

pub fn idvtec_impl(
    nlon: usize,
    ad: &[f32],
    bd: &[f32],
    av: &[f32],
    bv: &[f32],
    nlat: usize,
    nt: usize,
    isym: usize,
    wvhsec: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 4 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if isym > 2 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let expected = nlat * nlat * nt;
    if ad.len() != expected || bd.len() != expected || av.len() != expected || bv.len() != expected
    {
        return Err(PyValueError::new_err("ad/bd/av/bv size mismatch"));
    }

    let imid = nlat.div_ceil(2);
    let mmax = nlat.min((nlon + 1) / 2);
    let l1 = nlat.min((nlon + 1) / 2);
    let l2 = imid;
    let lwmin = 4 * nlat * l2 + 3 * l1.saturating_sub(2) * (2 * nlat - l1 - 1) + nlon + 15;
    ierror = 9;
    if wvhsec.len() < lwmin {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mn = mmax * nlat * nt;
    ierror = 10;
    if isym != 0 {
        if lwork < nlat * (2 * nt * nlon + (6 * imid).max(nlon)) + 4 * mn + nlat {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
        }
    } else if lwork < imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mut br = vec![0.0_f32; expected];
    let mut bi = vec![0.0_f32; expected];
    let mut cr = vec![0.0_f32; expected];
    let mut ci = vec![0.0_f32; expected];
    let mut pertbd = vec![0.0_f32; nt];
    let mut pertbv = vec![0.0_f32; nt];
    let sqnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                (f * (f + 1.0)).sqrt()
            }
        })
        .collect();

    for k in 0..nt {
        pertbd[k] = ad[k] / (2.0 * 2.0_f32.sqrt());
        pertbv[k] = av[k] / (2.0 * 2.0_f32.sqrt());
        for n in 2..=nlat {
            let idx = (n - 1) * nt + k;
            br[idx] = -ad[idx] / sqnn[n - 1];
            bi[idx] = -bd[idx] / sqnn[n - 1];
            cr[idx] = av[idx] / sqnn[n - 1];
            ci[idx] = bv[idx] / sqnn[n - 1];
        }
        for m in 2..=mmax {
            for n in m..=nlat {
                let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                br[idx] = -ad[idx] / sqnn[n - 1];
                bi[idx] = -bd[idx] / sqnn[n - 1];
                cr[idx] = av[idx] / sqnn[n - 1];
                ci[idx] = bv[idx] / sqnn[n - 1];
            }
        }
    }

    let ityp = ityp_from_isym(isym)?;
    let lwork_vhsec = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    let (v, w, ierr) = vhsec_impl(&br, &bi, &cr, &ci, nlat, nt, ityp, wvhsec, lwork_vhsec)?;
    Ok((v, w, pertbd, pertbv, ierr))
}

#[pyfunction]
pub fn idvtec<'py>(
    py: Python<'py>,
    nlon: usize,
    ad: PyReadonlyArrayDyn<'py, f32>,
    bd: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    idvtec_isym(py, nlon, ad, bd, av, bv, 0, wvhsec, lwork)
}

#[pyfunction]
pub fn idvtec_isym<'py>(
    py: Python<'py>,
    nlon: usize,
    ad: PyReadonlyArrayDyn<'py, f32>,
    bd: PyReadonlyArrayDyn<'py, f32>,
    av: PyReadonlyArrayDyn<'py, f32>,
    bv: PyReadonlyArrayDyn<'py, f32>,
    isym: usize,
    wvhsec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let shape = ad.shape().to_vec();
    if shape != bd.shape() || shape != av.shape() || shape != bv.shape() {
        return Err(PyValueError::new_err(
            "ad/bd/av/bv must have identical shapes",
        ));
    }
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err(
            "idvtec expects rank-2 or rank-3 coefficient arrays",
        ));
    }

    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let idvw = if isym == 0 { nlat } else { nlat.div_ceil(2) };
    let (v, w, pertbd, pertbv, ierror) = idvtec_impl(
        nlon,
        &collect_logical_ab(ad.as_array()),
        &collect_logical_ab(bd.as_array()),
        &collect_logical_ab(av.as_array()),
        &collect_logical_ab(bv.as_array()),
        nlat,
        nt,
        isym,
        wvhsec.as_slice()?,
        lwork,
    )?;
    if ierror != 0 {
        let empty2 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0, 0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let empty1 = ArrayD::<f32>::from_shape_vec(IxDyn(&[0]), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((
            empty2.clone().into_pyarray(py).into_any().unbind(),
            empty2.into_pyarray(py).into_any().unbind(),
            empty1.clone().into_pyarray(py).into_any().unbind(),
            empty1.into_pyarray(py).into_any().unbind(),
            ierror,
        ));
    }
    let shape_v = if shape.len() == 2 {
        IxDyn(&[idvw, nlon])
    } else {
        IxDyn(&[idvw, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(shape_v.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(shape_v, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pd_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertbd)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pv_arr = ArrayD::from_shape_vec(IxDyn(&[nt]), pertbv)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        pd_arr.into_pyarray(py).into_any().unbind(),
        pv_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
