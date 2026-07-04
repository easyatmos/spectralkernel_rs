use crate::hrfftf::fourier_analysis_real;
use crate::sphcom_vector::{zvin_column, zwin_column};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn mp1_ranges(start: usize, end: usize) -> Vec<(usize, usize)> {
    if start > end {
        return Vec::new();
    }
    let count = end - start + 1;
    let threads = rayon::current_num_threads().max(1);
    let chunk = count.div_ceil(threads).max(1);
    let mut ranges = Vec::new();
    let mut s = start;
    while s <= end {
        let e = (s + chunk - 1).min(end);
        ranges.push((s, e));
        s = e + 1;
    }
    ranges
}

fn computed_mode(ityp: usize) -> i32 {
    match ityp {
        4 | 8 => 1,
        5 | 7 => 2,
        _ => 0,
    }
}

fn build_vhaec_outputs(
    py: Python<'_>,
    out_shape: &[usize],
    br: Vec<f32>,
    bi: Vec<f32>,
    cr: Vec<f32>,
    ci: Vec<f32>,
    ierror: i32,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let br_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(out_shape), br)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bi_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(out_shape), bi)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(out_shape), cr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ci_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(out_shape), ci)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        br_arr.into_pyarray(py).into_any().unbind(),
        bi_arr.into_pyarray(py).into_any().unbind(),
        cr_arr.into_pyarray(py).into_any().unbind(),
        ci_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

/// Analyze vector fields on a regular grid using computed Legendre tables.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhaec`: Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhaec_impl(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhaec: &[f32],
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

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let lwzvin = lzz1 + labc;
    ierror = 9;
    if wvhaec.len() < 2 * lwzvin + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    let min_lwork = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    let whrfft = &wvhaec[2 * lwzvin..2 * lwzvin + nlon + 15];
    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];
    let mut br = vec![0.0_f32; nlat * nlat * nt];
    let mut bi = vec![0.0_f32; nlat * nlat * nt];
    let mut cr = vec![0.0_f32; nlat * nlat * nt];
    let mut ci = vec![0.0_f32; nlat * nlat * nt];

    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / nlon as f32;
    let fsn = 4.0_f32 / nlon as f32;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };

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

    for k in 0..nt {
        let mut plane_v = vec![0.0_f32; idv * nlon];
        let mut plane_w = vec![0.0_f32; idv * nlon];
        if ityp <= 2 {
            for i in 0..imid {
                for j in 0..nlon {
                    plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                    plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                }
            }
            for i in 0..imm1 {
                let row = imid + i;
                for j in 0..nlon {
                    plane_v[row * nlon + j] = vo[(i * nlon + j) * nt + k];
                    plane_w[row * nlon + j] = wo[(i * nlon + j) * nt + k];
                }
            }
        } else {
            for i in 0..idv {
                for j in 0..nlon {
                    plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                    plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                }
            }
        }
        fourier_analysis_real(idv, nlon, &mut plane_v, whrfft)?;
        fourier_analysis_real(idv, nlon, &mut plane_w, whrfft)?;
        if ityp <= 2 {
            for i in 0..imid {
                for j in 0..nlon {
                    ve[(i * nlon + j) * nt + k] = plane_v[i * nlon + j];
                    we[(i * nlon + j) * nt + k] = plane_w[i * nlon + j];
                }
            }
            for i in 0..imm1 {
                let row = imid + i;
                for j in 0..nlon {
                    vo[(i * nlon + j) * nt + k] = plane_v[row * nlon + j];
                    wo[(i * nlon + j) * nt + k] = plane_w[row * nlon + j];
                }
            }
        } else {
            for i in 0..idv {
                for j in 0..nlon {
                    let idx = (i * nlon + j) * nt + k;
                    ve[idx] = plane_v[i * nlon + j];
                    vo[idx] = plane_v[i * nlon + j];
                    we[idx] = plane_w[i * nlon + j];
                    wo[idx] = plane_w[i * nlon + j];
                }
            }
        }
    }

    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };
    let cmode = computed_mode(ityp);
    let wvbin64 = wvhaec[..lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();
    let wwbin64 = wvhaec[lwzvin..2 * lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();

    if !matches!(ityp, 2 | 5 | 8) {
        br.fill(0.0);
        bi.fill(0.0);
    }
    if !matches!(ityp, 1 | 4 | 7) {
        cr.fill(0.0);
        ci.fill(0.0);
    }

    let idx2 = |i: usize, np1: usize| (np1 - 1) * imid + (i - 1);

    let zv0 = zvin_column(nlat, nlon, cmode, 0, &wvbin64)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    match ityp {
        0 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        let z = zv0[idx2(i, np1)] as f32;
                        br[(np1 - 1) * nt + k] += z * ve[((i - 1) * nlon) * nt + k];
                        cr[(np1 - 1) * nt + k] -= z * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        let z = zv0[idx2(i, np1)] as f32;
                        br[(np1 - 1) * nt + k] += z * vo[((i - 1) * nlon) * nt + k];
                        cr[(np1 - 1) * nt + k] -= z * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        1 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * ve[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        2 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        3 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * ve[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        4 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * ve[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        5 => {
            for k in 0..nt {
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        6 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        7 => {
            for k in 0..nt {
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] +=
                            (zv0[idx2(i, np1)] as f32) * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        8 => {
            for k in 0..nt {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -=
                            (zv0[idx2(i, np1)] as f32) * we[((i - 1) * nlon) * nt + k];
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    if mmax >= 2 {
        for mp1 in 2..=mmax {
            let m = mp1 - 1;
            let mp2 = mp1 + 1;
            let zv = zvin_column(nlat, nlon, cmode, m, &wvbin64)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>();
            let zw = zwin_column(nlat, nlon, cmode, m, &wwbin64)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<_>>();
            if mp1 <= ndo1 {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        for np1 in (mp1..=ndo1).step_by(2) {
                            let zvidx = idx2(i, np1);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let c = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                            let s = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                            if matches!(ityp, 0 | 1 | 6 | 7) {
                                br[out] += (zv[zvidx] as f32) * vo[c] + (zw[zvidx] as f32) * we[s];
                                bi[out] += (zv[zvidx] as f32) * vo[s] - (zw[zvidx] as f32) * we[c];
                            }
                            if matches!(ityp, 0 | 2 | 3 | 5) {
                                cr[out] -= (zv[zvidx] as f32) * wo[c];
                                cr[out] += (zw[zvidx] as f32) * ve[s];
                                ci[out] -= (zv[zvidx] as f32) * wo[s];
                                ci[out] -= (zw[zvidx] as f32) * ve[c];
                            }
                        }
                    }
                    if mlat != 0 {
                        for np1 in (mp1..=ndo1).step_by(2) {
                            let zvidx = idx2(imid, np1);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let c = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                            let s = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                            if matches!(ityp, 0 | 1 | 6 | 7) {
                                br[out] += (zw[zvidx] as f32) * we[s];
                                bi[out] -= (zw[zvidx] as f32) * we[c];
                            }
                            if matches!(ityp, 0 | 2 | 3 | 5) {
                                cr[out] += (zw[zvidx] as f32) * ve[s];
                                ci[out] -= (zw[zvidx] as f32) * ve[c];
                            }
                        }
                    }
                }
            }
            if mp2 <= ndo2 {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        for np1 in (mp2..=ndo2).step_by(2) {
                            let zvidx = idx2(i, np1);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let c = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                            let s = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                            if matches!(ityp, 0 | 1 | 3 | 4) {
                                br[out] += (zv[zvidx] as f32) * ve[c] + (zw[zvidx] as f32) * wo[s];
                                bi[out] += (zv[zvidx] as f32) * ve[s] - (zw[zvidx] as f32) * wo[c];
                            }
                            if matches!(ityp, 0 | 2 | 6 | 8) {
                                cr[out] -= (zv[zvidx] as f32) * we[c];
                                cr[out] += (zw[zvidx] as f32) * vo[s];
                                ci[out] -= (zv[zvidx] as f32) * we[s];
                                ci[out] -= (zw[zvidx] as f32) * vo[c];
                            }
                        }
                    }
                    if mlat != 0 {
                        for np1 in (mp2..=ndo2).step_by(2) {
                            let zvidx = idx2(imid, np1);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let c = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                            let s = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                            if matches!(ityp, 0 | 1 | 3 | 4) {
                                br[out] += (zv[zvidx] as f32) * ve[c];
                                bi[out] += (zv[zvidx] as f32) * ve[s];
                            }
                            if matches!(ityp, 0 | 2 | 6 | 8) {
                                cr[out] -= (zv[zvidx] as f32) * we[c];
                                ci[out] -= (zv[zvidx] as f32) * we[s];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((br, bi, cr, ci, 0))
}

/// Parallel analysis of vector fields on a regular grid using computed Legendre tables.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhaec`: Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhaec_impl_parallel(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhaec: &[f32],
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

    let imid = (nlat + 1) / 2;
    let mmax = nlat.min((nlon + 1) / 2);
    let lzz1 = 2 * nlat * imid;
    let labc = 3 * (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let lwzvin = lzz1 + labc;
    ierror = 9;
    if wvhaec.len() < 2 * lwzvin + nlon + 15 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 10;
    let min_lwork = if ityp <= 2 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon))
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon))
    };
    if lwork < min_lwork {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    let whrfft = &wvhaec[2 * lwzvin..2 * lwzvin + nlon + 15];
    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];

    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / nlon as f32;
    let fsn = 4.0_f32 / nlon as f32;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };

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
            if ityp <= 2 {
                for i in 0..imid {
                    for j in 0..nlon {
                        plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                        plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                    }
                }
                for i in 0..imm1 {
                    let row = imid + i;
                    for j in 0..nlon {
                        plane_v[row * nlon + j] = vo[(i * nlon + j) * nt + k];
                        plane_w[row * nlon + j] = wo[(i * nlon + j) * nt + k];
                    }
                }
            } else {
                for i in 0..idv {
                    for j in 0..nlon {
                        plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                        plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                    }
                }
            }
            fourier_analysis_real(idv, nlon, &mut plane_v, whrfft)?;
            fourier_analysis_real(idv, nlon, &mut plane_w, whrfft)?;
            Ok((plane_v, plane_w))
        })
        .collect();

    for (k, result) in transformed_planes.into_iter().enumerate() {
        let (plane_v, plane_w) = result?;
        if ityp <= 2 {
            for i in 0..imid {
                for j in 0..nlon {
                    ve[(i * nlon + j) * nt + k] = plane_v[i * nlon + j];
                    we[(i * nlon + j) * nt + k] = plane_w[i * nlon + j];
                }
            }
            for i in 0..imm1 {
                let row = imid + i;
                for j in 0..nlon {
                    vo[(i * nlon + j) * nt + k] = plane_v[row * nlon + j];
                    wo[(i * nlon + j) * nt + k] = plane_w[row * nlon + j];
                }
            }
        } else {
            for i in 0..idv {
                for j in 0..nlon {
                    let idx = (i * nlon + j) * nt + k;
                    ve[idx] = plane_v[i * nlon + j];
                    vo[idx] = plane_v[i * nlon + j];
                    we[idx] = plane_w[i * nlon + j];
                    wo[idx] = plane_w[i * nlon + j];
                }
            }
        }
    }

    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };
    let idx2 = |i: usize, np1: usize| (np1 - 1) * imid + (i - 1);
    let coeff_len = nlat * nlat * nt;
    let cmode = computed_mode(ityp);
    let wvbin64 = wvhaec[..lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();
    let wwbin64 = wvhaec[lwzvin..2 * lwzvin]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();

    let zv0 = zvin_column(nlat, nlon, cmode, 0, &wvbin64)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>();
    let m_columns: Vec<(Vec<f32>, Vec<f32>)> = if mmax >= 2 {
        (1..mmax)
            .into_par_iter()
            .map(|m| {
                (
                    zvin_column(nlat, nlon, cmode, m, &wvbin64)
                        .into_iter()
                        .map(|x| x as f32)
                        .collect(),
                    zwin_column(nlat, nlon, cmode, m, &wwbin64)
                        .into_iter()
                        .map(|x| x as f32)
                        .collect(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut br = vec![0.0_f32; coeff_len];
    let mut bi = vec![0.0_f32; coeff_len];
    let mut cr = vec![0.0_f32; coeff_len];
    let mut ci = vec![0.0_f32; coeff_len];

    if matches!(ityp, 2 | 5 | 8) {
        br.fill(0.0);
        bi.fill(0.0);
    }
    if matches!(ityp, 1 | 4 | 7) {
        cr.fill(0.0);
        ci.fill(0.0);
    }

    for k in 0..nt {
        match ityp {
            0 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        let z = zv0[idx2(i, np1)];
                        br[(np1 - 1) * nt + k] += z * ve[((i - 1) * nlon) * nt + k];
                        cr[(np1 - 1) * nt + k] -= z * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        let z = zv0[idx2(i, np1)];
                        br[(np1 - 1) * nt + k] += z * vo[((i - 1) * nlon) * nt + k];
                        cr[(np1 - 1) * nt + k] -= z * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            1 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * ve[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            2 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            3 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * ve[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            4 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * ve[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            5 => {
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * wo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            6 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * we[((i - 1) * nlon) * nt + k];
                    }
                }
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            7 => {
                for i in 1..=imm1 {
                    for np1 in (3..=ndo1).step_by(2) {
                        br[(np1 - 1) * nt + k] += zv0[idx2(i, np1)] * vo[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            8 => {
                for i in 1..=imid {
                    for np1 in (2..=ndo2).step_by(2) {
                        cr[(np1 - 1) * nt + k] -= zv0[idx2(i, np1)] * we[((i - 1) * nlon) * nt + k];
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    if mmax >= 2 {
        let ranges = mp1_ranges(2, mmax);
        let (br_add, bi_add, cr_add, ci_add) = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut br_local = vec![0.0_f32; coeff_len];
                let mut bi_local = vec![0.0_f32; coeff_len];
                let mut cr_local = vec![0.0_f32; coeff_len];
                let mut ci_local = vec![0.0_f32; coeff_len];
                for mp1 in start..=end {
                    let m = mp1 - 1;
                    let mp2 = mp1 + 1;
                    let (zv, zw) = &m_columns[m - 1];
                    if mp1 <= ndo1 {
                        for k in 0..nt {
                            for i in 1..=imm1 {
                                for np1 in (mp1..=ndo1).step_by(2) {
                                    let zvidx = idx2(i, np1);
                                    let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                    let c = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                    let s = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                    if matches!(ityp, 0 | 1 | 6 | 7) {
                                        br_local[out] += zv[zvidx] * vo[c] + zw[zvidx] * we[s];
                                        bi_local[out] += zv[zvidx] * vo[s] - zw[zvidx] * we[c];
                                    }
                                    if matches!(ityp, 0 | 2 | 3 | 5) {
                                        cr_local[out] -= zv[zvidx] * wo[c];
                                        cr_local[out] += zw[zvidx] * ve[s];
                                        ci_local[out] -= zv[zvidx] * wo[s];
                                        ci_local[out] -= zw[zvidx] * ve[c];
                                    }
                                }
                            }
                            if mlat != 0 {
                                for np1 in (mp1..=ndo1).step_by(2) {
                                    let zvidx = idx2(imid, np1);
                                    let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                    let c = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                    let s = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                    if matches!(ityp, 0 | 1 | 6 | 7) {
                                        br_local[out] += zw[zvidx] * we[s];
                                        bi_local[out] -= zw[zvidx] * we[c];
                                    }
                                    if matches!(ityp, 0 | 2 | 3 | 5) {
                                        cr_local[out] += zw[zvidx] * ve[s];
                                        ci_local[out] -= zw[zvidx] * ve[c];
                                    }
                                }
                            }
                        }
                    }
                    if mp2 <= ndo2 {
                        for k in 0..nt {
                            for i in 1..=imm1 {
                                for np1 in (mp2..=ndo2).step_by(2) {
                                    let zvidx = idx2(i, np1);
                                    let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                    let c = ((i - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                    let s = ((i - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                    if matches!(ityp, 0 | 1 | 3 | 4) {
                                        br_local[out] += zv[zvidx] * ve[c] + zw[zvidx] * wo[s];
                                        bi_local[out] += zv[zvidx] * ve[s] - zw[zvidx] * wo[c];
                                    }
                                    if matches!(ityp, 0 | 2 | 6 | 8) {
                                        cr_local[out] -= zv[zvidx] * we[c];
                                        cr_local[out] += zw[zvidx] * vo[s];
                                        ci_local[out] -= zv[zvidx] * we[s];
                                        ci_local[out] -= zw[zvidx] * vo[c];
                                    }
                                }
                            }
                            if mlat != 0 {
                                for np1 in (mp2..=ndo2).step_by(2) {
                                    let zvidx = idx2(imid, np1);
                                    let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                                    let c = ((imid - 1) * nlon + (2 * mp1 - 2 - 1)) * nt + k;
                                    let s = ((imid - 1) * nlon + (2 * mp1 - 1 - 1)) * nt + k;
                                    if matches!(ityp, 0 | 1 | 3 | 4) {
                                        br_local[out] += zv[zvidx] * ve[c];
                                        bi_local[out] += zv[zvidx] * ve[s];
                                    }
                                    if matches!(ityp, 0 | 2 | 6 | 8) {
                                        cr_local[out] -= zv[zvidx] * we[c];
                                        ci_local[out] -= zv[zvidx] * we[s];
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
        add_assign(&mut br, &br_add);
        add_assign(&mut bi, &bi_add);
        add_assign(&mut cr, &cr_add);
        add_assign(&mut ci, &ci_add);
    }

    Ok((br, bi, cr, ci, 0))
}

#[pyfunction]
/// Python wrapper for `vhaec_impl` using the default vector layout.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhaec`: Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaec<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhaec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err("vhaec expects rank-2 or rank-3 v/w"));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = v.as_slice()?.to_vec();
    let wbuf = w.as_slice()?.to_vec();
    let wvbuf = wvhaec.as_slice()?.to_vec();
    let (br, bi, cr, ci, ierror) = py
        .detach(|| {
            vhaec_impl_parallel(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaec_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhaec_impl_parallel` that releases the GIL during analysis.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhaec`: Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaec_nogil<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhaec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhaec_nogil expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = v.as_slice()?.to_vec();
    let wbuf = w.as_slice()?.to_vec();
    let wvbuf = wvhaec.as_slice()?.to_vec();
    let result =
        py.detach(move || vhaec_impl_parallel(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork));
    let (br, bi, cr, ci, ierror) = result?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaec_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhaec_impl` with an explicit `ityp` selector.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhaec`: Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhaec_ityp<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhaec: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhaec_ityp expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let (br, bi, cr, ci, ierror) = vhaec_impl(
        v.as_slice()?,
        w.as_slice()?,
        nlat,
        nlon,
        nt,
        ityp,
        wvhaec.as_slice()?,
        lwork,
    )?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhaec_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}
