use crate::hrfftf::fourier_analysis_real;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn latitude_ranges(limit: usize) -> Vec<(usize, usize)> {
    if limit == 0 {
        return Vec::new();
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = limit.div_ceil(threads).max(1);
    let mut ranges = Vec::new();
    let mut start = 1usize;
    while start <= limit {
        let end = (start + chunk - 1).min(limit);
        ranges.push((start, end));
        start = end + 1;
    }
    ranges
}

/// Analyze vector fields on a regular grid using stored Legendre tables.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhaes`: Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhaes_impl(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhaes: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), 3));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    if v.len() != nlat * nlon * nt || w.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("v/w size mismatch"));
    }
    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    ierror = 9;
    if wvhaes.len() < lzimn + lzimn + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    if lwork < (2 * nt + 1) * nlat * nlon {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let idv = nlat;
    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];
    let mut br = vec![0.0_f32; nlat * nlat * nt];
    let mut bi = vec![0.0_f32; nlat * nlat * nt];
    let mut cr = vec![0.0_f32; nlat * nlat * nt];
    let mut ci = vec![0.0_f32; nlat * nlat * nt];

    let zv = &wvhaes[..lzimn];
    let zw = &wvhaes[lzimn..2 * lzimn];
    let whrfft = &wvhaes[2 * lzimn..2 * lzimn + nlon + 15];

    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / nlon as f32;
    let mlat = nlat % 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let mut imm1 = imid;
    if mlat != 0 {
        imm1 = imid - 1;
    }

    let fsn = 4.0_f32 / nlon as f32;
    for k in 0..nt {
        for i in 1..=imm1 {
            for j in 1..=nlon {
                let top = ((i - 1) * nlon + (j - 1)) * nt + k;
                let bot = (((nlp1 - i) - 1) * nlon + (j - 1)) * nt + k;
                if ityp <= 2 {
                    ve[top] = tsn * (v[top] + v[bot]);
                    vo[top] = tsn * (v[top] - v[bot]);
                    we[top] = tsn * (w[top] + w[bot]);
                    wo[top] = tsn * (w[top] - w[bot]);
                } else {
                    ve[top] = fsn * v[top];
                    vo[top] = fsn * v[top];
                    we[top] = fsn * w[top];
                    wo[top] = fsn * w[top];
                }
            }
        }
        if mlat != 0 {
            for j in 1..=nlon {
                let idx = ((imid - 1) * nlon + (j - 1)) * nt + k;
                ve[idx] = tsn * v[idx];
                we[idx] = tsn * w[idx];
            }
        }
    }

    let transformed_planes: Vec<PyResult<(Vec<f32>, Vec<f32>)>> = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut plane_v = vec![0.0_f32; idv * nlon];
            let mut plane_w = vec![0.0_f32; idv * nlon];
            for i in 0..imid {
                for j in 0..nlon {
                    plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                    plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                }
            }
            for i in 0..imm1 {
                for j in 0..nlon {
                    let row = imid + i;
                    plane_v[row * nlon + j] = vo[(i * nlon + j) * nt + k];
                    plane_w[row * nlon + j] = wo[(i * nlon + j) * nt + k];
                }
            }
            fourier_analysis_real(idv, nlon, &mut plane_v, whrfft)?;
            fourier_analysis_real(idv, nlon, &mut plane_w, whrfft)?;
            Ok((plane_v, plane_w))
        })
        .collect();

    for (k, result) in transformed_planes.into_iter().enumerate() {
        let (plane_v, plane_w) = result?;
        for i in 0..imid {
            for j in 0..nlon {
                ve[(i * nlon + j) * nt + k] = plane_v[i * nlon + j];
                we[(i * nlon + j) * nt + k] = plane_w[i * nlon + j];
            }
        }
        for i in 0..imm1 {
            for j in 0..nlon {
                let row = imid + i;
                vo[(i * nlon + j) * nt + k] = plane_v[row * nlon + j];
                wo[(i * nlon + j) * nt + k] = plane_w[row * nlon + j];
            }
        }
    }

    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };

    let coeff_len = nlat * nlat;
    let ranges = latitude_ranges(imid);
    for k in 0..nt {
        let (mut br_k, mut bi_k, mut cr_k, mut ci_k) = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut br_local = vec![0.0_f32; coeff_len];
                let mut bi_local = vec![0.0_f32; coeff_len];
                let mut cr_local = vec![0.0_f32; coeff_len];
                let mut ci_local = vec![0.0_f32; coeff_len];

                for i in start..=end {
                    for np1 in (2..=ndo2).step_by(2) {
                        let zidx = (np1 - 1) + (i - 1) * idz;
                        let out = np1 - 1;
                        if matches!(ityp, 0 | 1 | 3 | 4) {
                            br_local[out] += zv[zidx] * ve[((i - 1) * nlon) * nt + k];
                        }
                        if matches!(ityp, 0 | 2 | 6 | 8) {
                            cr_local[out] -= zv[zidx] * we[((i - 1) * nlon) * nt + k];
                        }
                    }

                    if i <= imm1 {
                        for np1 in (3..=ndo1).step_by(2) {
                            let zidx = (np1 - 1) + (i - 1) * idz;
                            let out = np1 - 1;
                            if matches!(ityp, 0 | 1 | 6 | 7) {
                                br_local[out] += zv[zidx] * vo[((i - 1) * nlon) * nt + k];
                            }
                            if matches!(ityp, 0 | 2 | 3 | 5) {
                                cr_local[out] -= zv[zidx] * wo[((i - 1) * nlon) * nt + k];
                            }
                        }

                        if mmax >= 2 {
                            for mp1 in 2..=mmax {
                                let m = mp1 - 1;
                                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                                let mp2 = mp1 + 1;
                                if mp1 <= ndo1 {
                                    for np1 in (mp1..=ndo1).step_by(2) {
                                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                                        let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                        let vo_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                        let vo_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                        if matches!(ityp, 0 | 1 | 6 | 7) {
                                            br_local[out] +=
                                                zv[zidx] * vo[vo_cos] + zw[zidx] * we[vo_sin];
                                            bi_local[out] +=
                                                zv[zidx] * vo[vo_sin] - zw[zidx] * we[vo_cos];
                                        }
                                        if matches!(ityp, 0 | 2 | 3 | 5) {
                                            cr_local[out] +=
                                                -zv[zidx] * wo[vo_cos] + zw[zidx] * ve[vo_sin];
                                            ci_local[out] +=
                                                -zv[zidx] * wo[vo_sin] - zw[zidx] * ve[vo_cos];
                                        }
                                    }
                                }
                                if mp2 <= ndo2 {
                                    for np1 in (mp2..=ndo2).step_by(2) {
                                        let zidx = (np1 + mb - 1) + (i - 1) * idz;
                                        let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                        let ve_cos = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                        let ve_sin = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                        if matches!(ityp, 0 | 1 | 3 | 4) {
                                            br_local[out] +=
                                                zv[zidx] * ve[ve_cos] + zw[zidx] * wo[ve_sin];
                                            bi_local[out] +=
                                                zv[zidx] * ve[ve_sin] - zw[zidx] * wo[ve_cos];
                                        }
                                        if matches!(ityp, 0 | 2 | 6 | 8) {
                                            cr_local[out] +=
                                                -zv[zidx] * we[ve_cos] + zw[zidx] * vo[ve_sin];
                                            ci_local[out] +=
                                                -zv[zidx] * we[ve_sin] - zw[zidx] * vo[ve_cos];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                (br_local, bi_local, cr_local, ci_local)
            })
            .reduce(
                || {
                    (
                        vec![0.0_f32; coeff_len],
                        vec![0.0_f32; coeff_len],
                        vec![0.0_f32; coeff_len],
                        vec![0.0_f32; coeff_len],
                    )
                },
                |mut acc, part| {
                    add_assign(&mut acc.0, &part.0);
                    add_assign(&mut acc.1, &part.1);
                    add_assign(&mut acc.2, &part.2);
                    add_assign(&mut acc.3, &part.3);
                    acc
                },
            );

        if mmax >= 2 && mlat != 0 {
            for mp1 in 2..=mmax {
                let m = mp1 - 1;
                let mb = m * (nlat - 1) - (m * (m - 1)) / 2;
                let mp2 = mp1 + 1;
                if mp1 <= ndo1 && matches!(ityp, 0 | 1 | 2 | 5 | 6 | 7) {
                    for np1 in (mp1..=ndo1).step_by(2) {
                        let zidx = (np1 + mb - 1) + (imid - 1) * idz;
                        let out = ((mp1 - 1) * nlat) + (np1 - 1);
                        let we_cos = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                        let we_sin = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                        let ve_cos_eq = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                        let ve_sin_eq = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                        if matches!(ityp, 0 | 1 | 6 | 7) {
                            br_k[out] += zw[zidx] * we[we_sin];
                            bi_k[out] -= zw[zidx] * we[we_cos];
                        }
                        if matches!(ityp, 0 | 2 | 3 | 5) {
                            cr_k[out] += zw[zidx] * ve[ve_sin_eq];
                            ci_k[out] -= zw[zidx] * ve[ve_cos_eq];
                        }
                    }
                }
                if mp2 <= ndo2 && matches!(ityp, 0 | 1 | 2 | 3 | 4 | 6 | 8) {
                    for np1 in (mp2..=ndo2).step_by(2) {
                        let zidx = (np1 + mb - 1) + (imid - 1) * idz;
                        let out = ((mp1 - 1) * nlat) + (np1 - 1);
                        let ve_cos_eq = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                        let ve_sin_eq = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                        let we_cos_eq = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                        let we_sin_eq = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                        if matches!(ityp, 0 | 1 | 3 | 4) {
                            br_k[out] += zv[zidx] * ve[ve_cos_eq];
                            bi_k[out] += zv[zidx] * ve[ve_sin_eq];
                        }
                        if matches!(ityp, 0 | 2 | 6 | 8) {
                            cr_k[out] -= zv[zidx] * we[we_cos_eq];
                            ci_k[out] -= zv[zidx] * we[we_sin_eq];
                        }
                    }
                }
            }
        }

        for out in 0..coeff_len {
            let idx = out * nt + k;
            br[idx] += br_k[out];
            bi[idx] += bi_k[out];
            cr[idx] += cr_k[out];
            ci[idx] += ci_k[out];
        }
    }

    let _ = ityp;
    Ok((br, bi, cr, ci, 0))
}

fn build_vhaes_outputs<'py>(
    py: Python<'py>,
    out_shape: &[usize],
    br: Vec<f32>,
    bi: Vec<f32>,
    cr: Vec<f32>,
    ci: Vec<f32>,
    ierror: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let br_arr = ArrayD::from_shape_vec(IxDyn(out_shape), br)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bi_arr = ArrayD::from_shape_vec(IxDyn(out_shape), bi)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr_arr = ArrayD::from_shape_vec(IxDyn(out_shape), cr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ci_arr = ArrayD::from_shape_vec(IxDyn(out_shape), ci)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        br_arr.into_pyarray(py).into_any().unbind(),
        bi_arr.into_pyarray(py).into_any().unbind(),
        cr_arr.into_pyarray(py).into_any().unbind(),
        ci_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
/// Python wrapper for `vhaes_impl` using the default vector layout.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhaes`: Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaes<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhaes: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err("vhaes expects rank-2 or rank-3 v/w"));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = v.as_slice()?.to_vec();
    let wbuf = w.as_slice()?.to_vec();
    let wvbuf = wvhaes.as_slice()?.to_vec();
    let (br, bi, cr, ci, ierror) = py
        .detach(|| {
            vhaes_impl(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaes_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhaes_impl` that releases the GIL during analysis.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhaes`: Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaes_nogil<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhaes: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhaes_nogil expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = v.as_slice()?.to_vec();
    let wbuf = w.as_slice()?.to_vec();
    let wvbuf = wvhaes.as_slice()?.to_vec();
    let result = py.detach(move || vhaes_impl(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork));
    let (br, bi, cr, ci, ierror) = result?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaes_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for the latitude-parallel `vhaes_impl` path that releases the GIL.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhaes`: Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaes_latpar_nogil<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhaes: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhaes_latpar_nogil expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = v.as_slice()?.to_vec();
    let wbuf = w.as_slice()?.to_vec();
    let wvbuf = wvhaes.as_slice()?.to_vec();
    let result = py.detach(move || vhaes_impl(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork));
    let (br, bi, cr, ci, ierror) = result?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaes_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhaes_impl` with an explicit `ityp` selector.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhaes`: Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaes_ityp<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhaes: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhaes_ityp expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let (br, bi, cr, ci, ierror) = vhaes_impl(
        v.as_slice()?,
        w.as_slice()?,
        nlat,
        nlon,
        nt,
        ityp,
        wvhaes.as_slice()?,
        lwork,
    )?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    let br_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), br)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bi_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), bi)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), cr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ci_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), ci)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        br_arr.into_pyarray(py).into_any().unbind(),
        bi_arr.into_pyarray(py).into_any().unbind(),
        cr_arr.into_pyarray(py).into_any().unbind(),
        ci_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
