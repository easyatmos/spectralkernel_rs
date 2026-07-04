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

fn collect_logical_vw(view: ndarray::ArrayViewD<'_, f32>) -> Vec<f32> {
    match view.ndim() {
        2 => {
            let nlat = view.shape()[0];
            let nlon = view.shape()[1];
            let mut out = Vec::with_capacity(nlat * nlon);
            for i in 0..nlat {
                for j in 0..nlon {
                    out.push(view[[i, j]]);
                }
            }
            out
        }
        3 => {
            let nlat = view.shape()[0];
            let nlon = view.shape()[1];
            let nt = view.shape()[2];
            let mut out = Vec::with_capacity(nlat * nlon * nt);
            for i in 0..nlat {
                for j in 0..nlon {
                    for k in 0..nt {
                        out.push(view[[i, j, k]]);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Analyze vector fields on a Gaussian grid using stored Legendre tables.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhags_impl(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhags: &[f32],
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
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
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
    let lmn = nlat * (nlat + 1) / 2;
    let basis = |i: usize, mn: usize| (mn - 1) * imid + (i - 1);
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    let vb_len = imid * lmn;
    ierror = 9;
    if wvhags.len() < (2 * vb_len + nlon + 15).max(lzimn + lzimn + nlon + 15) {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    let lnl = nt * idv * nlon;
    ierror = 10;
    if lwork < lnl + lnl + idv * nlon {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mut ve = vec![0.0_f32; idv * nlon * nt];
    let mut vo = vec![0.0_f32; idv * nlon * nt];
    let mut we = vec![0.0_f32; idv * nlon * nt];
    let mut wo = vec![0.0_f32; idv * nlon * nt];
    let mut br = vec![0.0_f32; nlat * nlat * nt];
    let mut bi = vec![0.0_f32; nlat * nlat * nt];
    let mut cr = vec![0.0_f32; nlat * nlat * nt];
    let mut ci = vec![0.0_f32; nlat * nlat * nt];

    let vb = &wvhags[..vb_len];
    let wb = &wvhags[vb_len..2 * vb_len];
    let whrfft = &wvhags[2 * vb_len..2 * vb_len + nlon + 15];

    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / nlon as f32;
    let fsn = 4.0_f32 / nlon as f32;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let packed_idx = |row: usize, j: usize, k: usize| ((row - 1) * nlon + (j - 1)) * nt + k;

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

    let mut packed_v = if ityp <= 2 {
        Some(vec![0.0_f32; idv * nlon * nt])
    } else {
        None
    };
    let mut packed_w = if ityp <= 2 {
        Some(vec![0.0_f32; idv * nlon * nt])
    } else {
        None
    };

    if ityp <= 2 {
        let pv = packed_v.as_mut().unwrap();
        let pw = packed_w.as_mut().unwrap();
        for k in 0..nt {
            for i in 1..=imid {
                for j in 1..=nlon {
                    pv[packed_idx(i, j, k)] = ve[((i - 1) * nlon + (j - 1)) * nt + k];
                    pw[packed_idx(i, j, k)] = we[((i - 1) * nlon + (j - 1)) * nt + k];
                }
            }
            for i in 1..=imm1 {
                for j in 1..=nlon {
                    pv[packed_idx(imid + i, j, k)] = vo[((i - 1) * nlon + (j - 1)) * nt + k];
                    pw[packed_idx(imid + i, j, k)] = wo[((i - 1) * nlon + (j - 1)) * nt + k];
                }
            }
        }
    }

    for k in 0..nt {
        if ityp <= 2 {
            let pv = packed_v.as_mut().unwrap();
            let pw = packed_w.as_mut().unwrap();
            let mut plane_v = vec![0.0_f32; idv * nlon];
            let mut plane_w = vec![0.0_f32; idv * nlon];
            for i in 1..=idv {
                for j in 0..nlon {
                    plane_v[(i - 1) * nlon + j] = pv[packed_idx(i, j + 1, k)];
                    plane_w[(i - 1) * nlon + j] = pw[packed_idx(i, j + 1, k)];
                }
            }
            fourier_analysis_real(idv, nlon, &mut plane_v, whrfft)?;
            fourier_analysis_real(idv, nlon, &mut plane_w, whrfft)?;
            for i in 1..=idv {
                for j in 0..nlon {
                    pv[packed_idx(i, j + 1, k)] = plane_v[(i - 1) * nlon + j];
                    pw[packed_idx(i, j + 1, k)] = plane_w[(i - 1) * nlon + j];
                }
            }
        } else {
            let mut plane_v = vec![0.0_f32; idv * nlon];
            let mut plane_w = vec![0.0_f32; idv * nlon];
            for i in 0..idv {
                for j in 0..nlon {
                    plane_v[i * nlon + j] = ve[(i * nlon + j) * nt + k];
                    plane_w[i * nlon + j] = we[(i * nlon + j) * nt + k];
                }
            }
            fourier_analysis_real(idv, nlon, &mut plane_v, whrfft)?;
            fourier_analysis_real(idv, nlon, &mut plane_w, whrfft)?;
            for i in 0..idv {
                for j in 0..nlon {
                    ve[(i * nlon + j) * nt + k] = plane_v[i * nlon + j];
                    we[(i * nlon + j) * nt + k] = plane_w[i * nlon + j];
                    vo[(i * nlon + j) * nt + k] = plane_v[i * nlon + j];
                    wo[(i * nlon + j) * nt + k] = plane_w[i * nlon + j];
                }
            }
        }
    }

    let get_ve = |i: usize, j: usize, k: usize, ve: &[f32], packed_v: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_v.as_ref().unwrap()[packed_idx(i, j, k)]
        } else {
            ve[((i - 1) * nlon + (j - 1)) * nt + k]
        }
    };
    let get_vo = |i: usize, j: usize, k: usize, vo: &[f32], packed_v: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_v.as_ref().unwrap()[packed_idx(imid + i, j, k)]
        } else {
            vo[((i - 1) * nlon + (j - 1)) * nt + k]
        }
    };
    let get_we = |i: usize, j: usize, k: usize, we: &[f32], packed_w: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_w.as_ref().unwrap()[packed_idx(i, j, k)]
        } else {
            we[((i - 1) * nlon + (j - 1)) * nt + k]
        }
    };
    let get_wo = |i: usize, j: usize, k: usize, wo: &[f32], packed_w: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_w.as_ref().unwrap()[packed_idx(imid + i, j, k)]
        } else {
            wo[((i - 1) * nlon + (j - 1)) * nt + k]
        }
    };

    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };

    if !matches!(ityp, 2 | 5 | 8) {
        br.fill(0.0);
        bi.fill(0.0);
    }
    if !matches!(ityp, 1 | 4 | 7) {
        cr.fill(0.0);
        ci.fill(0.0);
    }

    for k in 0..nt {
        for i in 1..=imid {
            for np1 in (2..=ndo2).step_by(2) {
                let zidx = basis(i, np1);
                let aidx = (np1 - 1) * nt + k;
                if matches!(ityp, 0 | 1 | 3 | 4) {
                    br[aidx] += vb[zidx] * get_ve(i, 1, k, &ve, &packed_v);
                }
                if matches!(ityp, 0 | 2 | 6 | 8) {
                    cr[aidx] -= vb[zidx] * get_we(i, 1, k, &we, &packed_w);
                }
            }
        }
        for i in 1..=imm1 {
            for np1 in (3..=ndo1).step_by(2) {
                let zidx = basis(i, np1);
                let aidx = (np1 - 1) * nt + k;
                if matches!(ityp, 0 | 1 | 6 | 7) {
                    br[aidx] += vb[zidx] * get_vo(i, 1, k, &vo, &packed_v);
                }
                if matches!(ityp, 0 | 2 | 3 | 5) {
                    cr[aidx] -= vb[zidx] * get_wo(i, 1, k, &wo, &packed_w);
                }
            }
        }
    }

    if mmax >= 2 {
        for mp1 in 2..=mmax {
            let m = mp1 - 1;
            let mb = m * nlat - (m * (m + 1)) / 2;
            let mp2 = mp1 + 1;
            if mp1 <= ndo1 {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        for np1 in (mp1..=ndo1).step_by(2) {
                            let zidx = basis(i, np1 + mb);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            let vo_cos = get_vo(i, jc, k, &vo, &packed_v);
                            let vo_sin = get_vo(i, js, k, &vo, &packed_v);
                            let we_cos = get_we(i, jc, k, &we, &packed_w);
                            let we_sin = get_we(i, js, k, &we, &packed_w);
                            let wo_cos = get_wo(i, jc, k, &wo, &packed_w);
                            let wo_sin = get_wo(i, js, k, &wo, &packed_w);
                            let ve_cos = get_ve(i, jc, k, &ve, &packed_v);
                            let ve_sin = get_ve(i, js, k, &ve, &packed_v);
                            if matches!(ityp, 0 | 1 | 6 | 7) {
                                br[out] += vb[zidx] * vo_cos + wb[zidx] * we_sin;
                                bi[out] += vb[zidx] * vo_sin - wb[zidx] * we_cos;
                            }
                            if matches!(ityp, 0 | 2 | 3 | 5) {
                                cr[out] += -vb[zidx] * wo_cos + wb[zidx] * ve_sin;
                                ci[out] += -vb[zidx] * wo_sin - wb[zidx] * ve_cos;
                            }
                        }
                    }
                    if mlat != 0 {
                        for np1 in (mp1..=ndo1).step_by(2) {
                            let zidx = basis(imid, np1 + mb);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            let we_cos = get_we(imid, jc, k, &we, &packed_w);
                            let we_sin = get_we(imid, js, k, &we, &packed_w);
                            let ve_cos = get_ve(imid, jc, k, &ve, &packed_v);
                            let ve_sin = get_ve(imid, js, k, &ve, &packed_v);
                            if matches!(ityp, 0 | 1 | 6 | 7) {
                                br[out] += wb[zidx] * we_sin;
                                bi[out] -= wb[zidx] * we_cos;
                            }
                            if matches!(ityp, 0 | 2 | 3 | 5) {
                                cr[out] += wb[zidx] * ve_sin;
                                ci[out] -= wb[zidx] * ve_cos;
                            }
                        }
                    }
                }
            }
            if mp2 <= ndo2 {
                for k in 0..nt {
                    for i in 1..=imm1 {
                        for np1 in (mp2..=ndo2).step_by(2) {
                            let zidx = basis(i, np1 + mb);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            let ve_cos = get_ve(i, jc, k, &ve, &packed_v);
                            let ve_sin = get_ve(i, js, k, &ve, &packed_v);
                            let wo_cos = get_wo(i, jc, k, &wo, &packed_w);
                            let wo_sin = get_wo(i, js, k, &wo, &packed_w);
                            let we_cos = get_we(i, jc, k, &we, &packed_w);
                            let we_sin = get_we(i, js, k, &we, &packed_w);
                            let vo_cos = get_vo(i, jc, k, &vo, &packed_v);
                            let vo_sin = get_vo(i, js, k, &vo, &packed_v);
                            if matches!(ityp, 0 | 1 | 3 | 4) {
                                br[out] += vb[zidx] * ve_cos + wb[zidx] * wo_sin;
                                bi[out] += vb[zidx] * ve_sin - wb[zidx] * wo_cos;
                            }
                            if matches!(ityp, 0 | 2 | 6 | 8) {
                                cr[out] += -vb[zidx] * we_cos + wb[zidx] * vo_sin;
                                ci[out] += -vb[zidx] * we_sin - wb[zidx] * vo_cos;
                            }
                        }
                    }
                    if mlat != 0 {
                        for np1 in (mp2..=ndo2).step_by(2) {
                            let zidx = basis(imid, np1 + mb);
                            let out = (((mp1 - 1) * nlat) + (np1 - 1)) * nt + k;
                            let jc = 2 * mp1 - 2;
                            let js = 2 * mp1 - 1;
                            let ve_cos = get_ve(imid, jc, k, &ve, &packed_v);
                            let ve_sin = get_ve(imid, js, k, &ve, &packed_v);
                            let we_cos = get_we(imid, jc, k, &we, &packed_w);
                            let we_sin = get_we(imid, js, k, &we, &packed_w);
                            if matches!(ityp, 0 | 1 | 3 | 4) {
                                br[out] += vb[zidx] * ve_cos;
                                bi[out] += vb[zidx] * ve_sin;
                            }
                            if matches!(ityp, 0 | 2 | 6 | 8) {
                                cr[out] -= vb[zidx] * we_cos;
                                ci[out] -= vb[zidx] * we_sin;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((br, bi, cr, ci, 0))
}

/// Parallel implementation of `vhags_impl`.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn vhags_impl_parallel(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhags: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    if nt <= 1 {
        return vhags_impl(v, w, nlat, nlon, nt, ityp, wvhags, lwork);
    }
    if v.len() != nlat * nlon * nt || w.len() != nlat * nlon * nt {
        return Err(PyValueError::new_err("v/w size mismatch"));
    }
    let coeff_len = nlat * nlat;
    let parts = (0..nt)
        .into_par_iter()
        .map(|k| {
            let mut vk = vec![0.0_f32; nlat * nlon];
            let mut wk = vec![0.0_f32; nlat * nlon];
            for i in 0..nlat {
                for j in 0..nlon {
                    let src = (i * nlon + j) * nt + k;
                    let dst = i * nlon + j;
                    vk[dst] = v[src];
                    wk[dst] = w[src];
                }
            }
            let (br, bi, cr, ci, ierr) = vhags_impl(&vk, &wk, nlat, nlon, 1, ityp, wvhags, lwork)?;
            Ok::<_, PyErr>((br, bi, cr, ci, ierr))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut br = vec![0.0_f32; coeff_len * nt];
    let mut bi = vec![0.0_f32; coeff_len * nt];
    let mut cr = vec![0.0_f32; coeff_len * nt];
    let mut ci = vec![0.0_f32; coeff_len * nt];
    for (k, (brk, bik, crk, cik, ierr)) in parts.into_iter().enumerate() {
        if ierr != 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierr));
        }
        for idx in 0..coeff_len {
            br[idx * nt + k] = brk[idx];
            bi[idx * nt + k] = bik[idx];
            cr[idx * nt + k] = crk[idx];
            ci[idx * nt + k] = cik[idx];
        }
    }
    Ok((br, bi, cr, ci, 0))
}

/// Rust entry point for `vhags_impl_latpar`.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
/// - `nt`: Number of stacked fields processed together.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn vhags_impl_latpar(
    v: &[f32],
    w: &[f32],
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
    wvhags: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, i32)> {
    if nt > 1 {
        if v.len() != nlat * nlon * nt || w.len() != nlat * nlon * nt {
            return Err(PyValueError::new_err("v/w size mismatch"));
        }
        let coeff_len = nlat * nlat;
        let parts = (0..nt)
            .into_par_iter()
            .map(|k| {
                let mut vk = vec![0.0_f32; nlat * nlon];
                let mut wk = vec![0.0_f32; nlat * nlon];
                for i in 0..nlat {
                    for j in 0..nlon {
                        let src = (i * nlon + j) * nt + k;
                        let dst = i * nlon + j;
                        vk[dst] = v[src];
                        wk[dst] = w[src];
                    }
                }
                let (br, bi, cr, ci, ierr) =
                    vhags_impl_latpar(&vk, &wk, nlat, nlon, 1, ityp, wvhags, lwork)?;
                Ok::<_, PyErr>((br, bi, cr, ci, ierr))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut br = vec![0.0_f32; coeff_len * nt];
        let mut bi = vec![0.0_f32; coeff_len * nt];
        let mut cr = vec![0.0_f32; coeff_len * nt];
        let mut ci = vec![0.0_f32; coeff_len * nt];
        for (k, (brk, bik, crk, cik, ierr)) in parts.into_iter().enumerate() {
            if ierr != 0 {
                return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierr));
            }
            for idx in 0..coeff_len {
                br[idx * nt + k] = brk[idx];
                bi[idx * nt + k] = bik[idx];
                cr[idx * nt + k] = crk[idx];
                ci[idx * nt + k] = cik[idx];
            }
        }
        return Ok((br, bi, cr, ci, 0));
    }

    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    ierror = 4;
    if nt < 1 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }
    if v.len() != nlat * nlon || w.len() != nlat * nlon {
        return Err(PyValueError::new_err("v/w size mismatch"));
    }

    let mmax = nlat.min((nlon + 1) / 2);
    let imid = (nlat + 1) / 2;
    let lmn = nlat * (nlat + 1) / 2;
    let basis = |i: usize, mn: usize| (mn - 1) * imid + (i - 1);
    let idz = (mmax * (nlat + nlat - mmax + 1)) / 2;
    let lzimn = idz * imid;
    let vb_len = imid * lmn;
    ierror = 9;
    if wvhags.len() < (2 * vb_len + nlon + 15).max(lzimn + lzimn + nlon + 15) {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let idv = if ityp <= 2 { nlat } else { imid };
    ierror = 10;
    if lwork < 2 * idv * nlon + idv * nlon {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), ierror));
    }

    let mut ve = vec![0.0_f32; idv * nlon];
    let mut vo = vec![0.0_f32; idv * nlon];
    let mut we = vec![0.0_f32; idv * nlon];
    let mut wo = vec![0.0_f32; idv * nlon];
    let vb = &wvhags[..vb_len];
    let wb = &wvhags[vb_len..2 * vb_len];
    let whrfft = &wvhags[2 * vb_len..2 * vb_len + nlon + 15];

    let nlp1 = nlat + 1;
    let tsn = 2.0_f32 / nlon as f32;
    let fsn = 4.0_f32 / nlon as f32;
    let mlat = nlat % 2;
    let imm1 = if mlat != 0 { imid - 1 } else { imid };
    let packed_idx = |row: usize, j: usize| (row - 1) * nlon + (j - 1);

    for i in 1..=imm1 {
        for j in 1..=nlon {
            let top = (i - 1) * nlon + (j - 1);
            let bot = ((nlp1 - i) - 1) * nlon + (j - 1);
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
            let idx = (imid - 1) * nlon + (j - 1);
            ve[idx] = tsn * v[idx];
            we[idx] = tsn * w[idx];
        }
    }

    let mut packed_v = if ityp <= 2 {
        Some(vec![0.0_f32; idv * nlon])
    } else {
        None
    };
    let mut packed_w = if ityp <= 2 {
        Some(vec![0.0_f32; idv * nlon])
    } else {
        None
    };

    if ityp <= 2 {
        let pv = packed_v.as_mut().unwrap();
        let pw = packed_w.as_mut().unwrap();
        for i in 1..=imid {
            for j in 1..=nlon {
                pv[packed_idx(i, j)] = ve[(i - 1) * nlon + (j - 1)];
                pw[packed_idx(i, j)] = we[(i - 1) * nlon + (j - 1)];
            }
        }
        for i in 1..=imm1 {
            for j in 1..=nlon {
                pv[packed_idx(imid + i, j)] = vo[(i - 1) * nlon + (j - 1)];
                pw[packed_idx(imid + i, j)] = wo[(i - 1) * nlon + (j - 1)];
            }
        }
        fourier_analysis_real(idv, nlon, pv, whrfft)?;
        fourier_analysis_real(idv, nlon, pw, whrfft)?;
    } else {
        fourier_analysis_real(idv, nlon, &mut ve, whrfft)?;
        fourier_analysis_real(idv, nlon, &mut we, whrfft)?;
        vo.clone_from(&ve);
        wo.clone_from(&we);
    }

    let get_ve = |i: usize, j: usize, ve: &[f32], packed_v: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_v.as_ref().unwrap()[packed_idx(i, j)]
        } else {
            ve[(i - 1) * nlon + (j - 1)]
        }
    };
    let get_vo = |i: usize, j: usize, vo: &[f32], packed_v: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_v.as_ref().unwrap()[packed_idx(imid + i, j)]
        } else {
            vo[(i - 1) * nlon + (j - 1)]
        }
    };
    let get_we = |i: usize, j: usize, we: &[f32], packed_w: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_w.as_ref().unwrap()[packed_idx(i, j)]
        } else {
            we[(i - 1) * nlon + (j - 1)]
        }
    };
    let get_wo = |i: usize, j: usize, wo: &[f32], packed_w: &Option<Vec<f32>>| -> f32 {
        if ityp <= 2 {
            packed_w.as_ref().unwrap()[packed_idx(imid + i, j)]
        } else {
            wo[(i - 1) * nlon + (j - 1)]
        }
    };

    let ndo1 = if mlat != 0 { nlat - 1 } else { nlat };
    let ndo2 = if mlat == 0 { nlat - 1 } else { nlat };
    let coeff_len = nlat * nlat;

    let (br, bi, cr, ci) = (1..=imid)
        .into_par_iter()
        .map(|i| {
            let mut br_local = vec![0.0_f32; coeff_len];
            let mut bi_local = vec![0.0_f32; coeff_len];
            let mut cr_local = vec![0.0_f32; coeff_len];
            let mut ci_local = vec![0.0_f32; coeff_len];

            for np1 in (2..=ndo2).step_by(2) {
                let zidx = basis(i, np1);
                let out = np1 - 1;
                if matches!(ityp, 0 | 1 | 3 | 4) {
                    br_local[out] += vb[zidx] * get_ve(i, 1, &ve, &packed_v);
                }
                if matches!(ityp, 0 | 2 | 6 | 8) {
                    cr_local[out] -= vb[zidx] * get_we(i, 1, &we, &packed_w);
                }
            }

            if i <= imm1 {
                for np1 in (3..=ndo1).step_by(2) {
                    let zidx = basis(i, np1);
                    let out = np1 - 1;
                    if matches!(ityp, 0 | 1 | 6 | 7) {
                        br_local[out] += vb[zidx] * get_vo(i, 1, &vo, &packed_v);
                    }
                    if matches!(ityp, 0 | 2 | 3 | 5) {
                        cr_local[out] -= vb[zidx] * get_wo(i, 1, &wo, &packed_w);
                    }
                }
            }

            if mmax >= 2 {
                for mp1 in 2..=mmax {
                    let m = mp1 - 1;
                    let mb = m * nlat - (m * (m + 1)) / 2;
                    let mp2 = mp1 + 1;
                    if mp1 <= ndo1 {
                        if i <= imm1 {
                            for np1 in (mp1..=ndo1).step_by(2) {
                                let zidx = basis(i, np1 + mb);
                                let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                let jc = 2 * mp1 - 2;
                                let js = 2 * mp1 - 1;
                                let vo_cos = get_vo(i, jc, &vo, &packed_v);
                                let vo_sin = get_vo(i, js, &vo, &packed_v);
                                let we_cos = get_we(i, jc, &we, &packed_w);
                                let we_sin = get_we(i, js, &we, &packed_w);
                                let wo_cos = get_wo(i, jc, &wo, &packed_w);
                                let wo_sin = get_wo(i, js, &wo, &packed_w);
                                let ve_cos = get_ve(i, jc, &ve, &packed_v);
                                let ve_sin = get_ve(i, js, &ve, &packed_v);
                                if matches!(ityp, 0 | 1 | 6 | 7) {
                                    br_local[out] += vb[zidx] * vo_cos + wb[zidx] * we_sin;
                                    bi_local[out] += vb[zidx] * vo_sin - wb[zidx] * we_cos;
                                }
                                if matches!(ityp, 0 | 2 | 3 | 5) {
                                    cr_local[out] += -vb[zidx] * wo_cos + wb[zidx] * ve_sin;
                                    ci_local[out] += -vb[zidx] * wo_sin - wb[zidx] * ve_cos;
                                }
                            }
                        } else if mlat != 0 {
                            for np1 in (mp1..=ndo1).step_by(2) {
                                let zidx = basis(imid, np1 + mb);
                                let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                let jc = 2 * mp1 - 2;
                                let js = 2 * mp1 - 1;
                                let we_cos = get_we(imid, jc, &we, &packed_w);
                                let we_sin = get_we(imid, js, &we, &packed_w);
                                let ve_cos = get_ve(imid, jc, &ve, &packed_v);
                                let ve_sin = get_ve(imid, js, &ve, &packed_v);
                                if matches!(ityp, 0 | 1 | 6 | 7) {
                                    br_local[out] += wb[zidx] * we_sin;
                                    bi_local[out] -= wb[zidx] * we_cos;
                                }
                                if matches!(ityp, 0 | 2 | 3 | 5) {
                                    cr_local[out] += wb[zidx] * ve_sin;
                                    ci_local[out] -= wb[zidx] * ve_cos;
                                }
                            }
                        }
                    }
                    if mp2 <= ndo2 {
                        if i <= imm1 {
                            for np1 in (mp2..=ndo2).step_by(2) {
                                let zidx = basis(i, np1 + mb);
                                let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                let jc = 2 * mp1 - 2;
                                let js = 2 * mp1 - 1;
                                let ve_cos = get_ve(i, jc, &ve, &packed_v);
                                let ve_sin = get_ve(i, js, &ve, &packed_v);
                                let wo_cos = get_wo(i, jc, &wo, &packed_w);
                                let wo_sin = get_wo(i, js, &wo, &packed_w);
                                let we_cos = get_we(i, jc, &we, &packed_w);
                                let we_sin = get_we(i, js, &we, &packed_w);
                                let vo_cos = get_vo(i, jc, &vo, &packed_v);
                                let vo_sin = get_vo(i, js, &vo, &packed_v);
                                if matches!(ityp, 0 | 1 | 3 | 4) {
                                    br_local[out] += vb[zidx] * ve_cos + wb[zidx] * wo_sin;
                                    bi_local[out] += vb[zidx] * ve_sin - wb[zidx] * wo_cos;
                                }
                                if matches!(ityp, 0 | 2 | 6 | 8) {
                                    cr_local[out] += -vb[zidx] * we_cos + wb[zidx] * vo_sin;
                                    ci_local[out] += -vb[zidx] * we_sin - wb[zidx] * vo_cos;
                                }
                            }
                        } else if mlat != 0 {
                            for np1 in (mp2..=ndo2).step_by(2) {
                                let zidx = basis(imid, np1 + mb);
                                let out = ((mp1 - 1) * nlat) + (np1 - 1);
                                let jc = 2 * mp1 - 2;
                                let js = 2 * mp1 - 1;
                                let ve_cos = get_ve(imid, jc, &ve, &packed_v);
                                let ve_sin = get_ve(imid, js, &ve, &packed_v);
                                let we_cos = get_we(imid, jc, &we, &packed_w);
                                let we_sin = get_we(imid, js, &we, &packed_w);
                                if matches!(ityp, 0 | 1 | 3 | 4) {
                                    br_local[out] += vb[zidx] * ve_cos;
                                    bi_local[out] += vb[zidx] * ve_sin;
                                }
                                if matches!(ityp, 0 | 2 | 6 | 8) {
                                    cr_local[out] -= vb[zidx] * we_cos;
                                    ci_local[out] -= vb[zidx] * we_sin;
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

    Ok((br, bi, cr, ci, 0))
}

fn build_vhags_outputs<'py>(
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
/// Python wrapper for `vhags_impl` using the default vector layout.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhags<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhags: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err("vhags expects rank-2 or rank-3 v/w"));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());
    let wvbuf = wvhags.as_slice()?.to_vec();
    let (br, bi, cr, ci, ierror) = py
        .detach(|| {
            vhags_impl_parallel(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork)
                .map_err(|err| err.to_string())
        })
        .map_err(PyValueError::new_err)?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhags_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhags_impl` that releases the GIL during analysis.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhags_nogil<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhags: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhags_nogil expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());
    let wvbuf = wvhags.as_slice()?.to_vec();
    let result =
        py.detach(move || vhags_impl_parallel(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork));
    let (br, bi, cr, ci, ierror) = result?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhags_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for the latitude-parallel `vhags_impl` path that releases the GIL.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhags_latpar_nogil<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    wvhags: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhags_latpar_nogil expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());
    let wvbuf = wvhags.as_slice()?.to_vec();
    let result =
        py.detach(move || vhags_impl_latpar(&vbuf, &wbuf, nlat, nlon, nt, 0, &wvbuf, lwork));
    let (br, bi, cr, ci, ierror) = result?;
    let out_shape = if vshape.len() == 2 {
        vec![nlat, nlat]
    } else {
        vec![nlat, nlat, nt]
    };
    build_vhags_outputs(py, &out_shape, br, bi, cr, ci, ierror)
}

#[pyfunction]
/// Python wrapper for `vhags_impl` with an explicit `ityp` selector.
///
/// # Parameters
/// - `v`: Input vector component stored in `(nlat, nlon[, nt])` order.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `ityp`: Vector storage selector controlling the coefficient families in use.
/// - `wvhags`: Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.
/// - `lwork`: Length of the caller-provided work array.
///
/// # Returns
/// Four NumPy arrays together with a error code.
pub fn vhags_ityp<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhags: PyReadonlyArrayDyn<'py, f32>,
    lwork: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)> {
    let vshape = v.shape().to_vec();
    let wshape = w.shape().to_vec();
    if vshape != wshape {
        return Err(PyValueError::new_err("v and w must have identical shapes"));
    }
    if vshape.len() != 2 && vshape.len() != 3 {
        return Err(PyValueError::new_err(
            "vhags_ityp expects rank-2 or rank-3 v/w",
        ));
    }
    let nlat = vshape[0];
    let nlon = vshape[1];
    let nt = if vshape.len() == 2 { 1 } else { vshape[2] };
    let vbuf = collect_logical_vw(v.as_array());
    let wbuf = collect_logical_vw(w.as_array());

    let (br, bi, cr, ci, ierror) = vhags_impl_parallel(
        &vbuf,
        &wbuf,
        nlat,
        nlon,
        nt,
        ityp,
        wvhags.as_slice()?,
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
