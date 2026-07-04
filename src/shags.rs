use crate::hrfftf::fourier_analysis_real;
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[allow(dead_code)]
fn infer_nlon_from_wshags(nlat: usize, ltotal: usize) -> Option<(usize, usize, usize, usize)> {
    for nlon in 4..=4 * nlat.max(4) {
        let l = ((nlon + 2) / 2).min(nlat);
        let late = (nlat + (nlat % 2)) / 2;
        let labc = l * (l - 1) / 2 + (nlat - l) * (l - 1);
        let prefix = nlat + 2 * nlat * late + 3 * labc + nlon + 15;
        let pmnf_count = l * (2 * nlat - l + 1) / 2;
        if prefix + late * pmnf_count == ltotal {
            return Some((nlon, l, late, prefix));
        }
    }
    None
}

pub fn shags_impl(
    g: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    wshags: &[f32],
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
    let mode = 0usize;
    let l = ((nlon + 2) / 2).min(nlat);
    let late = (nlat + (nlat % 2)) / 2;
    let lat = nlat;
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    if g.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("g size mismatch"));
    }
    ierror = 9;
    let lp_i64 = (nlat as i64) * (3 * (l + late) as i64 - 2)
        + (((l - 1) as i64) * (late as i64) * ((2 * nlat - l) as i64)
            - 3 * (l as i64) * ((l - 1) as i64))
            / 2
        + (nlon as i64)
        + 15;
    let lp = usize::try_from(lp_i64)
        .map_err(|_| PyValueError::new_err("invalid shags workspace formula"))?;
    if wshags.len() < lp {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < nlat * nlon * (nt + 1) {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let ifft = nlat + 2 * nlat * late + 3 * (l * (l - 1) / 2 + (nlat - l) * (l - 1));
    let whrfft = &wshags[ifft..ifft + nlon + 15];
    let mut grid = g.to_vec();
    let analyzed_planes: Vec<PyResult<Vec<f32>>> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut plane = vec![0.0_f32; lat * nlon];
            for i in 0..lat {
                for j in 0..nlon {
                    plane[i * nlon + j] = grid[(i * nlon + j) * nt + k];
                }
            }
            fourier_analysis_real(lat, nlon, &mut plane, whrfft)?;
            let sfn = 2.0_f32 / nlon as f32;
            for value in &mut plane {
                *value *= sfn;
            }
            Ok(plane)
        })
        .collect();

    for (k, result) in analyzed_planes.into_iter().enumerate() {
        let plane = result?;
        for i in 0..lat {
            for j in 0..nlon {
                grid[(i * nlon + j) * nt + k] = plane[i * nlon + j];
            }
        }
    }

    let iwts = 0usize;
    let ipmn = ifft + nlon + 15;
    let wts = &wshags[iwts..iwts + nlat];
    let pmnf = &wshags[ipmn..];

    let mut a = vec![0.0_f32; nlat * nlat * nt];
    let mut b = vec![0.0_f32; nlat * nlat * nt];
    let lm1 = if nlon == l + l - 2 { l - 1 } else { l };
    let nl2 = nlat / 2;

    for k in 0..nt {
        for j in 0..nlon {
            for i in 0..nl2 {
                let is = nlat - i - 1;
                let t1 = grid[(i * nlon + j) * nt + k];
                let t2 = grid[(is * nlon + j) * nt + k];
                grid[(i * nlon + j) * nt + k] = wts[i] * (t1 + t2);
                grid[(is * nlon + j) * nt + k] = wts[i] * (t1 - t2);
            }
            if nlat % 2 != 0 {
                let idx = ((late - 1) * nlon + j) * nt + k;
                grid[idx] = wts[late - 1] * grid[idx];
            }
        }
    }

    let scalar_coeffs: Vec<(Vec<f32>, Vec<f32>)> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut a_local = vec![0.0_f32; nlat * nlat];
            let mut b_local = vec![0.0_f32; nlat * nlat];

            for i in 1..=late {
                let is = nlat - i + 1;
                for np1 in (1..=nlat).step_by(2) {
                    let mn = np1;
                    a_local[np1 - 1] +=
                        grid[((i - 1) * nlon) * nt + k] * pmnf[(mn - 1) * late + (i - 1)];
                }
                for np1 in (2..=nlat).step_by(2) {
                    let mn = np1;
                    a_local[np1 - 1] +=
                        grid[((is - 1) * nlon) * nt + k] * pmnf[(mn - 1) * late + (i - 1)];
                }
            }

            for mp1 in 2..=lm1 {
                let m = mp1 - 1;
                let mml1 = m * (2 * nlat - m - 1) / 2;
                let mp2 = mp1 + 1;
                for i in 1..=late {
                    let is = nlat - i + 1;
                    for np1 in (mp1..=nlat).step_by(2) {
                        let mn = mml1 + np1;
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        a_local[aidx] += grid[((i - 1) * nlon + (2 * m - 1)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                        b_local[aidx] += grid[((i - 1) * nlon + (2 * m)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                    }
                    for np1 in (mp2..=nlat).step_by(2) {
                        let mn = mml1 + np1;
                        let aidx = ((mp1 - 1) * nlat) + (np1 - 1);
                        a_local[aidx] += grid[((is - 1) * nlon + (2 * m - 1)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                        b_local[aidx] += grid[((is - 1) * nlon + (2 * m)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                    }
                }
            }

            if nlon == l + l - 2 {
                let m = l - 1;
                let mml1 = m * (2 * nlat - m - 1) / 2;
                for i in 1..=late {
                    let is = nlat - i + 1;
                    for np1 in (l..=nlat).step_by(2) {
                        let mn = mml1 + np1;
                        let aidx = ((l - 1) * nlat) + (np1 - 1);
                        a_local[aidx] += 0.5_f32
                            * grid[((i - 1) * nlon + (nlon - 1)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                    }
                    for np1 in ((l + 1)..=nlat).step_by(2) {
                        let mn = mml1 + np1;
                        let aidx = ((l - 1) * nlat) + (np1 - 1);
                        a_local[aidx] += 0.5_f32
                            * grid[((is - 1) * nlon + (nlon - 1)) * nt + k]
                            * pmnf[(mn - 1) * late + (i - 1)];
                    }
                }
            }

            (a_local, b_local)
        })
        .collect();

    for (k, (a_local, b_local)) in scalar_coeffs.into_iter().enumerate() {
        for idx in 0..(nlat * nlat) {
            a[idx * nt + k] = a_local[idx];
            b[idx * nt + k] = b_local[idx];
        }
    }

    let _ = mode;
    Ok((a, b, 0))
}

#[pyfunction]
pub fn shags<'py>(
    py: Python<'py>,
    g: PyReadonlyArrayDyn<'py, f32>,
    wshags: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i32)> {
    let shape = g.shape().to_vec();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(PyValueError::new_err("shags expects rank-2 or rank-3 g"));
    }
    let nlat = shape[0];
    let nlon = shape[1];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let gbuf = g.as_slice()?.to_vec();
    let wbuf = wshags.as_slice()?.to_vec();
    let (a, b, ierror) = py
        .detach(|| shags_impl(&gbuf, nlat, nlon, nt, &wbuf, lwork).map_err(|err| err.to_string()))
        .map_err(PyValueError::new_err)?;
    let ashape = if shape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let aarr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), a)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let barr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&ashape), b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        aarr.into_pyarray(py).into_any().unbind(),
        barr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
