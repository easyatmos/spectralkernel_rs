use crate::vhsgc::vhsgc_impl;
use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;

fn trace_enabled() -> bool {
    std::env::var("VLAPGC_TRACE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

fn trace_line(message: &str) {
    if !trace_enabled() {
        return;
    }
    if let Ok(path) = std::env::var("VLAPGC_TRACE_FILE") {
        if !path.is_empty() {
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(file, "{message}");
                return;
            }
        }
    }
    eprintln!("{message}");
}

fn trace_coeff_sample(name: &str, data: &[f32], nlat: usize, nt: usize) {
    if !trace_enabled() {
        return;
    }
    let take_n = nlat.min(4);
    let take_k = nt.min(2);
    trace_line(&format!("[vlapgc trace] {name} sample:"));
    for n in 0..take_n {
        for k in 0..take_k {
            let idx = n * nt + k;
            trace_line(&format!("  n={}, k={}, val={}", n + 1, k + 1, data[idx]));
        }
    }
}

fn trace_mn_sample(name: &str, data: &[f32], nlat: usize, nt: usize, samples: &[(usize, usize)]) {
    if !trace_enabled() {
        return;
    }
    trace_line(&format!("[vlapgc trace] {name} targeted sample:"));
    for &(m, n) in samples {
        if m == 0 || n == 0 || m > n || n > nlat {
            continue;
        }
        for k in 0..nt.min(2) {
            let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
            trace_line(&format!(
                "  m={}, n={}, k={}, val={}",
                m,
                n,
                k + 1,
                data[idx]
            ));
        }
    }
}

fn trace_flat_window(name: &str, data: &[f32], start: usize, len: usize) {
    if !trace_enabled() {
        return;
    }
    let end = (start + len).min(data.len());
    trace_line(&format!(
        "[vlapgc trace] {name} flat window [{}..{}):",
        start, end
    ));
    for (idx, val) in data[start..end].iter().enumerate() {
        trace_line(&format!("  idx={}, val={}", start + idx, val));
    }
}

fn trace_layout_probe(name: &str, data: &[f32], nlat: usize, nt: usize) {
    if !trace_enabled() {
        return;
    }
    let probes = [(1usize, 1usize), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)];
    trace_line(&format!("[vlapgc trace] {name} layout probe:"));
    for &(m, n) in &probes {
        if m > n || n > nlat {
            continue;
        }
        for k in 0..nt.min(2) {
            let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
            trace_line(&format!(
                "  logical(m={}, n={}, k={}) -> flat idx {} val={}",
                m,
                n,
                k + 1,
                idx,
                data[idx]
            ));
        }
    }
}

fn trace_workspace_layout(ityp: usize, mn: usize, nlat: usize, lwork: usize) {
    if !trace_enabled() {
        return;
    }
    let (ibr, ibi, icr, ici) = if matches!(ityp, 0 | 3 | 6) {
        (1usize, 1 + mn, 1 + 2 * mn, 1 + 3 * mn)
    } else if matches!(ityp, 1 | 4 | 7) {
        (1usize, 1 + mn, 1 + 2 * mn, 1 + 2 * mn)
    } else {
        (1usize, 1usize, 1 + mn, 1 + 2 * mn)
    };
    let ifn = ici + mn;
    let iwk = ifn + nlat;
    let liwk = lwork.saturating_sub(4 * mn + nlat);
    trace_line(&format!(
        "[vlapgc trace] fortran packed layout ityp={} ibr={} ibi={} icr={} ici={} ifn={} iwk={} liwk={}",
        ityp, ibr, ibi, icr, ici, ifn, iwk, liwk
    ));
    trace_line(&format!(
        "[vlapgc trace] rust uses fully split buffers: brlap=0..{} bilap=0..{} crlap=0..{} cilap=0..{}",
        mn, mn, mn, mn
    ));
    if ibi == 1 || ici == icr {
        trace_line("[vlapgc trace] packed alias detected in Fortran workspace for this ityp");
    }
}

fn trace_ityp_profile(ityp: usize) {
    if !trace_enabled() {
        return;
    }
    let profile = match ityp {
        0 | 3 | 6 => "all coefficient groups active (br/bi/cr/ci)",
        1 | 4 | 7 => "br/bi active; cr/ci expected to stay zero",
        2 | 5 | 8 => "cr/ci active; br/bi expected to stay zero",
        _ => unreachable!(),
    };
    trace_line(&format!(
        "[vlapgc trace] ityp={} profile: {}",
        ityp, profile
    ));
}

fn trace_region_stats(name: &str, data: &[f32], nlat: usize, nt: usize, mmax: usize) {
    if !trace_enabled() {
        return;
    }
    let mut valid_nnz = 0usize;
    let mut invalid_nnz = 0usize;
    let mut truncated_nnz = 0usize;
    let mut valid_max = 0.0_f32;
    let mut invalid_max = 0.0_f32;
    let mut truncated_max = 0.0_f32;

    let mmax_eff = mmax.min(nlat);
    for k in 0..nt {
        for m in 0..nlat {
            for n in 0..nlat {
                let idx = ((m * nlat + n) * nt) + k;
                let val = data[idx].abs();
                if m >= mmax_eff {
                    if val != 0.0 {
                        truncated_nnz += 1;
                        truncated_max = truncated_max.max(val);
                    }
                } else if m > n {
                    if val != 0.0 {
                        invalid_nnz += 1;
                        invalid_max = invalid_max.max(val);
                    }
                } else if val != 0.0 {
                    valid_nnz += 1;
                    valid_max = valid_max.max(val);
                }
            }
        }
    }

    trace_line(&format!(
        "[vlapgc trace] {name} region stats: valid_nnz={} valid_maxabs={} invalid_nnz={} invalid_maxabs={} truncated_nnz={} truncated_maxabs={}",
        valid_nnz, valid_max, invalid_nnz, invalid_max, truncated_nnz, truncated_max
    ));
}

fn trace_fnn(name: &str, fnn: &[f32]) {
    if !trace_enabled() {
        return;
    }
    let take = fnn.len().min(8);
    trace_line(&format!("[vlapgc trace] {name} head: {:?}", &fnn[..take]));
}

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

fn expand_vlapgc_output(
    data: Vec<f32>,
    nlat: usize,
    nlon: usize,
    nt: usize,
    ityp: usize,
) -> Vec<f32> {
    let idv = if ityp <= 2 { nlat } else { nlat.div_ceil(2) };
    if idv == nlat {
        return data;
    }
    let mut full = vec![0.0_f32; nlat * nlon * nt];
    for i in 0..idv {
        for j in 0..nlon {
            for k in 0..nt {
                full[(i * nlon + j) * nt + k] = data[(i * nlon + j) * nt + k];
            }
        }
    }
    full
}

pub fn vlapgc_impl(
    nlon: usize,
    br: &[f32],
    bi: &[f32],
    cr: &[f32],
    ci: &[f32],
    nlat: usize,
    nt: usize,
    ityp: usize,
    wvhsgc: &[f32],
    lwork: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, i32)> {
    if trace_enabled() {
        trace_line(&format!(
            "[vlapgc trace] enter nlat={} nlon={} nt={} ityp={} br_len={} wvhsgc_len={} lwork={}",
            nlat,
            nlon,
            nt,
            ityp,
            br.len(),
            wvhsgc.len(),
            lwork
        ));
    }
    let mut ierror = 1;
    if nlat < 3 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 2;
    if nlon < 1 {
        return Ok((Vec::new(), Vec::new(), ierror));
    }
    ierror = 3;
    if ityp > 8 {
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
    let mmax = nlat.min((nlon + 1) / 2);
    let mn = mmax * nlat * nt;
    let l1 = nlat.min((nlon + 1) / 2);
    let l2 = imid;
    let lwmin = 4 * nlat * l2 + 3 * l1.saturating_sub(2) * (2 * nlat - l1 - 1) + nlon + 15;
    if trace_enabled() {
        trace_line(&format!(
            "[vlapgc trace] dims imid={} mmax={} l1={} l2={} lwmin={} mn={}",
            imid, mmax, l1, l2, lwmin, mn
        ));
    }
    ierror = 9;
    if wvhsgc.len() < lwmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let lwkmin = if ityp < 3 {
        nlat * (2 * nt * nlon + (6 * imid).max(nlon) + 1) + 4 * mn
    } else {
        imid * (2 * nt * nlon + (6 * nlat).max(nlon)) + 4 * mn + nlat
    };
    if trace_enabled() {
        trace_line(&format!("[vlapgc trace] lwkmin={}", lwkmin));
        trace_workspace_layout(ityp, mn, nlat, lwork);
        trace_ityp_profile(ityp);
        trace_coeff_sample("br(in)", br, nlat, nt);
        trace_coeff_sample("bi(in)", bi, nlat, nt);
        trace_coeff_sample("cr(in)", cr, nlat, nt);
        trace_coeff_sample("ci(in)", ci, nlat, nt);
        trace_layout_probe("br(in)", br, nlat, nt);
        trace_layout_probe("bi(in)", bi, nlat, nt);
        trace_layout_probe("cr(in)", cr, nlat, nt);
        trace_layout_probe("ci(in)", ci, nlat, nt);
        trace_mn_sample("br(in)", br, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("bi(in)", bi, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("cr(in)", cr, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("ci(in)", ci, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_flat_window("br(in)", br, 0, (6 * nt).max(8));
        trace_flat_window("bi(in)", bi, 0, (6 * nt).max(8));
        trace_flat_window("cr(in)", cr, 0, (6 * nt).max(8));
        trace_flat_window("ci(in)", ci, 0, (6 * nt).max(8));
    }
    ierror = 10;
    if lwork < lwkmin {
        return Ok((Vec::new(), Vec::new(), ierror));
    }

    let mut brlap = vec![0.0_f32; expected];
    let mut bilap = vec![0.0_f32; expected];
    let mut crlap = vec![0.0_f32; expected];
    let mut cilap = vec![0.0_f32; expected];
    let fnn: Vec<f32> = (0..nlat)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else {
                let f = idx as f32;
                -f * (f + 1.0)
            }
        })
        .collect();
    trace_fnn("fnn", &fnn);

    match ityp {
        0 | 3 | 6 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brlap[idx] = fnn[n - 1] * br[idx];
                    bilap[idx] = fnn[n - 1] * bi[idx];
                    crlap[idx] = fnn[n - 1] * cr[idx];
                    cilap[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brlap[idx] = fnn[n - 1] * br[idx];
                        bilap[idx] = fnn[n - 1] * bi[idx];
                        crlap[idx] = fnn[n - 1] * cr[idx];
                        cilap[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
        1 | 4 | 7 => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    brlap[idx] = fnn[n - 1] * br[idx];
                    bilap[idx] = fnn[n - 1] * bi[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        brlap[idx] = fnn[n - 1] * br[idx];
                        bilap[idx] = fnn[n - 1] * bi[idx];
                    }
                }
            }
        }
        _ => {
            for k in 0..nt {
                for n in 2..=nlat {
                    let idx = (n - 1) * nt + k;
                    crlap[idx] = fnn[n - 1] * cr[idx];
                    cilap[idx] = fnn[n - 1] * ci[idx];
                }
                for m in 2..=mmax {
                    for n in m..=nlat {
                        let idx = (((m - 1) * nlat) + (n - 1)) * nt + k;
                        crlap[idx] = fnn[n - 1] * cr[idx];
                        cilap[idx] = fnn[n - 1] * ci[idx];
                    }
                }
            }
        }
    }

    if trace_enabled() {
        trace_coeff_sample("brlap", &brlap, nlat, nt);
        trace_coeff_sample("bilap", &bilap, nlat, nt);
        trace_coeff_sample("crlap", &crlap, nlat, nt);
        trace_coeff_sample("cilap", &cilap, nlat, nt);
        trace_layout_probe("brlap", &brlap, nlat, nt);
        trace_layout_probe("bilap", &bilap, nlat, nt);
        trace_layout_probe("crlap", &crlap, nlat, nt);
        trace_layout_probe("cilap", &cilap, nlat, nt);
        trace_mn_sample("brlap", &brlap, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("bilap", &bilap, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("crlap", &crlap, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_mn_sample("cilap", &cilap, nlat, nt, &[(2, 2), (2, 3), (3, 3), (3, 4)]);
        trace_flat_window("brlap", &brlap, 0, (6 * nt).max(8));
        trace_flat_window("bilap", &bilap, 0, (6 * nt).max(8));
        trace_flat_window("crlap", &crlap, 0, (6 * nt).max(8));
        trace_flat_window("cilap", &cilap, 0, (6 * nt).max(8));
        trace_region_stats("brlap", &brlap, nlat, nt, mmax);
        trace_region_stats("bilap", &bilap, nlat, nt, mmax);
        trace_region_stats("crlap", &crlap, nlat, nt, mmax);
        trace_region_stats("cilap", &cilap, nlat, nt, mmax);
    }

    let lwork_vhsgc = lwork.saturating_sub(4 * mn + nlat);
    if trace_enabled() {
        trace_line(&format!(
            "[vlapgc trace] delegated lwork_vhsgc={}",
            lwork_vhsgc
        ));
    }
    let (vlap, wlap, _idvw, _nlon2, ierr) = vhsgc_impl(
        &brlap,
        &bilap,
        &crlap,
        &cilap,
        nlat,
        nt,
        ityp,
        wvhsgc,
        lwork_vhsgc,
    )?;
    if trace_enabled() {
        trace_line(&format!(
            "[vlapgc trace] vhsgc returned ierr={} vlap_len={} wlap_len={}",
            ierr,
            vlap.len(),
            wlap.len()
        ));
    }
    Ok((vlap, wlap, ierr))
}

#[pyfunction]
pub fn vlapgc<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
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
            "vlapgc expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = vlapgc_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        0,
        wvhsgc.as_slice()?,
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
    let v = expand_vlapgc_output(v, nlat, nlon, nt, 0);
    let w = expand_vlapgc_output(w, nlat, nlon, nt, 0);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}

#[pyfunction]
pub fn vlapgc_ityp<'py>(
    py: Python<'py>,
    nlon: usize,
    br: PyReadonlyArrayDyn<'py, f32>,
    bi: PyReadonlyArrayDyn<'py, f32>,
    cr: PyReadonlyArrayDyn<'py, f32>,
    ci: PyReadonlyArrayDyn<'py, f32>,
    ityp: usize,
    wvhsgc: PyReadonlyArrayDyn<'py, f32>,
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
            "vlapgc_ityp expects rank-2 or rank-3 coefficient arrays",
        ));
    }
    let nlat = shape[0];
    let nt = if shape.len() == 2 { 1 } else { shape[2] };
    let (v, w, ierror) = vlapgc_impl(
        nlon,
        &collect_logical_coeffs(br.as_array()),
        &collect_logical_coeffs(bi.as_array()),
        &collect_logical_coeffs(cr.as_array()),
        &collect_logical_coeffs(ci.as_array()),
        nlat,
        nt,
        ityp,
        wvhsgc.as_slice()?,
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
    let v = expand_vlapgc_output(v, nlat, nlon, nt, ityp);
    let w = expand_vlapgc_output(w, nlat, nlon, nt, ityp);
    let out_shape = if shape.len() == 2 {
        IxDyn(&[nlat, nlon])
    } else {
        IxDyn(&[nlat, nlon, nt])
    };
    let v_arr = ArrayD::from_shape_vec(out_shape.clone(), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let w_arr =
        ArrayD::from_shape_vec(out_shape, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        v_arr.into_pyarray(py).into_any().unbind(),
        w_arr.into_pyarray(py).into_any().unbind(),
        ierror,
    ))
}
