use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HrfftfMode {
    Auto,
    ReferenceOnly,
    KernelOnly,
}

#[inline]
fn idx4(a1: usize, a2: usize, a3: usize, a4: usize, d1: usize, d2: usize, d3: usize) -> usize {
    (a1 - 1) + d1 * ((a2 - 1) + d2 * ((a3 - 1) + d3 * (a4 - 1)))
}

#[inline]
fn idx3(a1: usize, a2: usize, a3: usize, d1: usize, d2: usize) -> usize {
    (a1 - 1) + d1 * ((a2 - 1) + d2 * (a3 - 1))
}

#[inline]
fn pimach() -> f32 {
    std::f32::consts::PI
}

fn fourier_analysis_real_row_major(rows: usize, nlon: usize, data: &mut [f32]) {
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut out = vec![0.0_f32; rows * nlon];

    for i in 0..rows {
        let row = &data[i * nlon..(i + 1) * nlon];
        let out_row = &mut out[i * nlon..(i + 1) * nlon];

        for k in 0..nlon {
            let mut sum = 0.0_f32;
            for j in 0..nlon {
                let value = row[j];
                if k == 0 {
                    sum += value;
                } else if k == nlon - 1 && nlon.is_multiple_of(2) {
                    sum += if j % 2 == 0 { value } else { -value };
                } else if k % 2 == 1 {
                    let mode = k.div_ceil(2);
                    let angle = two_pi * (mode as f32) * (j as f32) / (nlon as f32);
                    sum += value * angle.cos();
                } else {
                    let mode = k / 2;
                    let angle = two_pi * (mode as f32) * (j as f32) / (nlon as f32);
                    sum += -value * angle.sin();
                }
            }
            out_row[k] = sum;
        }
    }

    data.copy_from_slice(&out);
}

fn unpack_col_major_to_rows(m: usize, n: usize, r: &[f32]) -> Vec<f32> {
    let mut row_major = vec![0.0_f32; m * n];
    for j in 0..n {
        for i in 0..m {
            row_major[i * n + j] = r[j * m + i];
        }
    }
    row_major
}

fn pack_rows_to_col_major(m: usize, n: usize, row_major: &[f32], r: &mut [f32]) {
    for j in 0..n {
        for i in 0..m {
            r[j * m + i] = row_major[i * n + j];
        }
    }
}

fn hrfftf_can_use_kernel(n: usize, whrfft: &[f32]) -> bool {
    if n == 1 || whrfft.len() < n + 15 {
        return false;
    }
    let fac = &whrfft[n..];
    let nf = fac[1] as usize;
    nf > 0 && (1..=nf).all(|k| fac[k + 1] >= 2.0)
}

fn hrfftf_impl_with_mode(
    m: usize,
    n: usize,
    r: &mut [f32],
    whrfft: &[f32],
    mode: HrfftfMode,
) -> PyResult<()> {
    if n == 1 {
        return Ok(());
    }
    if r.len() != m * n {
        return Err(PyValueError::new_err("hrfftf_impl: r length mismatch"));
    }
    if whrfft.len() < n + 15 {
        return Err(PyValueError::new_err(
            "hrfftf_impl: whrfft length too small",
        ));
    }

    match mode {
        HrfftfMode::ReferenceOnly => {
            let mut row_major = unpack_col_major_to_rows(m, n, r);
            fourier_analysis_real_row_major(m, n, &mut row_major);
            pack_rows_to_col_major(m, n, &row_major, r);
        }
        HrfftfMode::KernelOnly => {
            let mut work = vec![0.0_f32; m * n];
            let wa = &whrfft[..n];
            let fac = &whrfft[n..];
            hrftf1(m, n, r, m, &mut work, wa, fac)?;
        }
        HrfftfMode::Auto => {
            if hrfftf_can_use_kernel(n, whrfft) {
                let mut work = vec![0.0_f32; m * n];
                let wa = &whrfft[..n];
                let fac = &whrfft[n..];
                hrftf1(m, n, r, m, &mut work, wa, fac)?;
            } else {
                let mut row_major = unpack_col_major_to_rows(m, n, r);
                fourier_analysis_real_row_major(m, n, &mut row_major);
                pack_rows_to_col_major(m, n, &row_major, r);
            }
        }
    }
    Ok(())
}

pub fn hrfftf_impl(m: usize, n: usize, r: &mut [f32], whrfft: &[f32]) -> PyResult<()> {
    hrfftf_impl_with_mode(m, n, r, whrfft, HrfftfMode::Auto)
}

pub(crate) fn fourier_analysis_real(
    rows: usize,
    nlon: usize,
    data: &mut [f32],
    whrfft: &[f32],
) -> PyResult<()> {
    if data.len() != rows * nlon {
        return Err(PyValueError::new_err(
            "fourier_analysis_real: data length mismatch",
        ));
    }

    let mut packed = vec![0.0_f32; rows * nlon];
    for j in 0..nlon {
        for i in 0..rows {
            packed[j * rows + i] = data[i * nlon + j];
        }
    }
    hrfftf_impl(rows, nlon, &mut packed, whrfft)?;
    for j in 0..nlon {
        for i in 0..rows {
            data[i * nlon + j] = packed[j * rows + i];
        }
    }
    Ok(())
}

fn hrfftf_with_mode_py<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
    mode: HrfftfMode,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = r.shape();
    let mut data = if shape.len() == 2 {
        if shape[0] != m || shape[1] != n {
            return Err(PyValueError::new_err(
                "hrfftf: input shape does not match m,n",
            ));
        }
        let arr = r.as_array();
        let mut packed = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                packed.push(arr[[i, j]]);
            }
        }
        packed
    } else if shape.len() == 1 {
        let flat = r.as_slice()?;
        if flat.len() != m * n {
            return Err(PyValueError::new_err(
                "hrfftf: flat input length does not match m*n",
            ));
        }
        flat.to_vec()
    } else {
        return Err(PyValueError::new_err(
            "hrfftf expects rank-1 or rank-2 input",
        ));
    };

    let wsave = whrfft.as_slice()?;
    hrfftf_impl_with_mode(m, n, &mut data, wsave, mode)?;

    if shape.len() == 2 {
        let mut c_flat = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                c_flat.push(data[j * m + i]);
            }
        }
        Ok(PyArray1::from_vec(py, c_flat))
    } else {
        Ok(PyArray1::from_vec(py, data))
    }
}

#[pyfunction]
pub fn hrfftf_reference_only<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    hrfftf_with_mode_py(py, r, whrfft, m, n, HrfftfMode::ReferenceOnly)
}

#[pyfunction]
pub fn hrfftf_kernel_only<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    hrfftf_with_mode_py(py, r, whrfft, m, n, HrfftfMode::KernelOnly)
}

#[pyfunction]
pub fn hrfftf<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = r.shape();
    let mut data = if shape.len() == 2 {
        if shape[0] != m || shape[1] != n {
            return Err(PyValueError::new_err(
                "hrfftf: input shape does not match m,n",
            ));
        }
        let arr = r.as_array();
        let mut packed = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                packed.push(arr[[i, j]]);
            }
        }
        packed
    } else if shape.len() == 1 {
        let flat = r.as_slice()?;
        if flat.len() != m * n {
            return Err(PyValueError::new_err(
                "hrfftf: flat input length does not match m*n",
            ));
        }
        flat.to_vec()
    } else {
        return Err(PyValueError::new_err(
            "hrfftf expects rank-1 or rank-2 input",
        ));
    };

    let wsave = whrfft.as_slice()?;
    hrfftf_impl(m, n, &mut data, wsave)?;

    if shape.len() == 2 {
        let mut c_flat = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                c_flat.push(data[j * m + i]);
            }
        }
        Ok(PyArray1::from_vec(py, c_flat))
    } else {
        Ok(PyArray1::from_vec(py, data))
    }
}

fn hrftf1(
    m: usize,
    n: usize,
    c: &mut [f32],
    mdimc: usize,
    ch: &mut [f32],
    wa: &[f32],
    fac: &[f32],
) -> PyResult<()> {
    let nf = fac[1] as usize;
    let mut na = 1usize;
    let mut l2 = n;
    let mut iw = n;

    for k1 in 1..=nf {
        let kh = nf - k1;
        let ip = fac[kh + 2] as usize;
        let l1 = l2 / ip;
        let ido = n / l2;
        let idl1 = ido * l1;
        iw -= (ip - 1) * ido;
        na = 1 - na;

        match ip {
            4 => {
                let ix2 = iw + ido;
                let ix3 = ix2 + ido;
                if na == 0 {
                    hradf4(
                        m,
                        ido,
                        l1,
                        c,
                        mdimc,
                        ch,
                        m,
                        &wa[iw - 1..],
                        &wa[ix2 - 1..],
                        &wa[ix3 - 1..],
                    );
                } else {
                    hradf4(
                        m,
                        ido,
                        l1,
                        ch,
                        m,
                        c,
                        mdimc,
                        &wa[iw - 1..],
                        &wa[ix2 - 1..],
                        &wa[ix3 - 1..],
                    );
                }
            }
            2 => {
                if na == 0 {
                    hradf2(m, ido, l1, c, mdimc, ch, m, &wa[iw - 1..]);
                } else {
                    hradf2(m, ido, l1, ch, m, c, mdimc, &wa[iw - 1..]);
                }
            }
            3 => {
                let ix2 = iw + ido;
                if na == 0 {
                    hradf3(m, ido, l1, c, mdimc, ch, m, &wa[iw - 1..], &wa[ix2 - 1..]);
                } else {
                    hradf3(m, ido, l1, ch, m, c, mdimc, &wa[iw - 1..], &wa[ix2 - 1..]);
                }
            }
            5 => {
                let ix2 = iw + ido;
                let ix3 = ix2 + ido;
                let ix4 = ix3 + ido;
                if na == 0 {
                    hradf5(
                        m,
                        ido,
                        l1,
                        c,
                        mdimc,
                        ch,
                        m,
                        &wa[iw - 1..],
                        &wa[ix2 - 1..],
                        &wa[ix3 - 1..],
                        &wa[ix4 - 1..],
                    );
                } else {
                    hradf5(
                        m,
                        ido,
                        l1,
                        ch,
                        m,
                        c,
                        mdimc,
                        &wa[iw - 1..],
                        &wa[ix2 - 1..],
                        &wa[ix3 - 1..],
                        &wa[ix4 - 1..],
                    );
                }
            }
            _ => {
                if ido == 1 {
                    na = 1 - na;
                }
                if na == 0 {
                    hradfg(m, ido, ip, l1, idl1, c, mdimc, ch, m, &wa[iw - 1..]);
                    na = 1;
                } else {
                    hradfg(m, ido, ip, l1, idl1, ch, m, c, mdimc, &wa[iw - 1..]);
                    na = 0;
                }
            }
        }
        l2 = l1;
    }

    if na == 1 {
        return Ok(());
    }
    c.copy_from_slice(ch);
    Ok(())
}

fn hradf2(
    mp: usize,
    ido: usize,
    l1: usize,
    cc: &[f32],
    mdimcc: usize,
    ch: &mut [f32],
    mdimch: usize,
    wa1: &[f32],
) {
    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, 1, k, mdimch, ido, 2)] =
                cc[idx4(m, 1, k, 1, mdimcc, ido, l1)] + cc[idx4(m, 1, k, 2, mdimcc, ido, l1)];
            ch[idx4(m, ido, 2, k, mdimch, ido, 2)] =
                cc[idx4(m, 1, k, 1, mdimcc, ido, l1)] - cc[idx4(m, 1, k, 2, mdimcc, ido, l1)];
        }
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        let idp2 = ido + 2;
        for k in 1..=l1 {
            let mut i = 3;
            while i <= ido {
                let ic = idp2 - i;
                for m in 1..=mp {
                    ch[idx4(m, i, 1, k, mdimch, ido, 2)] = cc[idx4(m, i, k, 1, mdimcc, ido, l1)]
                        + (wa1[i - 3] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]
                            - wa1[i - 2] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]);
                    ch[idx4(m, ic, 2, k, mdimch, ido, 2)] = (wa1[i - 3]
                        * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]
                        - wa1[i - 2] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)])
                        - cc[idx4(m, i, k, 1, mdimcc, ido, l1)];
                    ch[idx4(m, i - 1, 1, k, mdimch, ido, 2)] = cc
                        [idx4(m, i - 1, k, 1, mdimcc, ido, l1)]
                        + (wa1[i - 3] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]
                            + wa1[i - 2] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]);
                    ch[idx4(m, ic - 1, 2, k, mdimch, ido, 2)] = cc
                        [idx4(m, i - 1, k, 1, mdimcc, ido, l1)]
                        - (wa1[i - 3] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]
                            + wa1[i - 2] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]);
                }
                i += 2;
            }
        }
    }
    if ido % 2 == 1 {
        return;
    }
    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, 2, k, mdimch, ido, 2)] = -cc[idx4(m, ido, k, 2, mdimcc, ido, l1)];
            ch[idx4(m, ido, 1, k, mdimch, ido, 2)] = cc[idx4(m, ido, k, 1, mdimcc, ido, l1)];
        }
    }
}

fn hradfg(
    mp: usize,
    ido: usize,
    ip: usize,
    l1: usize,
    idl1: usize,
    cc: &mut [f32],
    mdimcc: usize,
    ch: &mut [f32],
    mdimch: usize,
    wa: &[f32],
) {
    let tpi = 2.0 * pimach();
    let arg = tpi / ip as f32;
    let dcp = arg.cos();
    let dsp = arg.sin();
    let ipph = (ip + 1) / 2;
    let ipp2 = ip + 2;
    let idp2 = ido + 2;
    let nbd = (ido - 1) / 2;

    if ido != 1 {
        for ik in 1..=idl1 {
            for m in 1..=mp {
                ch[idx3(m, ik, 1, mdimch, idl1)] = cc[idx3(m, ik, 1, mdimcc, idl1)];
            }
        }

        for j in 2..=ip {
            for k in 1..=l1 {
                for m in 1..=mp {
                    ch[idx4(m, 1, k, j, mdimch, ido, l1)] = cc[idx4(m, 1, k, j, mdimcc, ido, l1)];
                }
            }
        }

        if nbd <= l1 {
            let mut is = 0usize;
            for j in 2..=ip {
                let mut idij = is;
                let mut i = 3;
                while i <= ido {
                    idij += 2;
                    for k in 1..=l1 {
                        for m in 1..=mp {
                            let c_im1 = cc[idx4(m, i - 1, k, j, mdimcc, ido, l1)];
                            let c_i = cc[idx4(m, i, k, j, mdimcc, ido, l1)];
                            ch[idx4(m, i - 1, k, j, mdimch, ido, l1)] =
                                wa[idij - 2] * c_im1 + wa[idij - 1] * c_i;
                            ch[idx4(m, i, k, j, mdimch, ido, l1)] =
                                wa[idij - 2] * c_i - wa[idij - 1] * c_im1;
                        }
                    }
                    i += 2;
                }
                is += ido;
            }
        } else {
            let mut is = 0usize;
            for j in 2..=ip {
                for k in 1..=l1 {
                    let mut idij = is;
                    let mut i = 3;
                    while i <= ido {
                        idij += 2;
                        for m in 1..=mp {
                            let c_im1 = cc[idx4(m, i - 1, k, j, mdimcc, ido, l1)];
                            let c_i = cc[idx4(m, i, k, j, mdimcc, ido, l1)];
                            ch[idx4(m, i - 1, k, j, mdimch, ido, l1)] =
                                wa[idij - 2] * c_im1 + wa[idij - 1] * c_i;
                            ch[idx4(m, i, k, j, mdimch, ido, l1)] =
                                wa[idij - 2] * c_i - wa[idij - 1] * c_im1;
                        }
                        i += 2;
                    }
                }
                is += ido;
            }
        }

        if nbd >= l1 {
            for j in 2..=ipph {
                let jc = ipp2 - j;
                for k in 1..=l1 {
                    let mut i = 3;
                    while i <= ido {
                        for m in 1..=mp {
                            let hj_im1 = ch[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                            let hj_i = ch[idx4(m, i, k, j, mdimch, ido, l1)];
                            let hc_im1 = ch[idx4(m, i - 1, k, jc, mdimch, ido, l1)];
                            let hc_i = ch[idx4(m, i, k, jc, mdimch, ido, l1)];
                            cc[idx4(m, i - 1, k, j, mdimcc, ido, l1)] = hj_im1 + hc_im1;
                            cc[idx4(m, i - 1, k, jc, mdimcc, ido, l1)] = hj_i - hc_i;
                            cc[idx4(m, i, k, j, mdimcc, ido, l1)] = hj_i + hc_i;
                            cc[idx4(m, i, k, jc, mdimcc, ido, l1)] = hc_im1 - hj_im1;
                        }
                        i += 2;
                    }
                }
            }
        } else {
            for j in 2..=ipph {
                let jc = ipp2 - j;
                let mut i = 3;
                while i <= ido {
                    for k in 1..=l1 {
                        for m in 1..=mp {
                            let hj_im1 = ch[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                            let hj_i = ch[idx4(m, i, k, j, mdimch, ido, l1)];
                            let hc_im1 = ch[idx4(m, i - 1, k, jc, mdimch, ido, l1)];
                            let hc_i = ch[idx4(m, i, k, jc, mdimch, ido, l1)];
                            cc[idx4(m, i - 1, k, j, mdimcc, ido, l1)] = hj_im1 + hc_im1;
                            cc[idx4(m, i - 1, k, jc, mdimcc, ido, l1)] = hj_i - hc_i;
                            cc[idx4(m, i, k, j, mdimcc, ido, l1)] = hj_i + hc_i;
                            cc[idx4(m, i, k, jc, mdimcc, ido, l1)] = hc_im1 - hj_im1;
                        }
                    }
                    i += 2;
                }
            }
        }
    } else {
        for ik in 1..=idl1 {
            for m in 1..=mp {
                cc[idx3(m, ik, 1, mdimcc, idl1)] = ch[idx3(m, ik, 1, mdimch, idl1)];
            }
        }
    }

    for j in 2..=ipph {
        let jc = ipp2 - j;
        for k in 1..=l1 {
            for m in 1..=mp {
                let hj = ch[idx4(m, 1, k, j, mdimch, ido, l1)];
                let hjc = ch[idx4(m, 1, k, jc, mdimch, ido, l1)];
                cc[idx4(m, 1, k, j, mdimcc, ido, l1)] = hj + hjc;
                cc[idx4(m, 1, k, jc, mdimcc, ido, l1)] = hjc - hj;
            }
        }
    }

    let mut ar1 = 1.0_f32;
    let mut ai1 = 0.0_f32;
    for l in 2..=ipph {
        let lc = ipp2 - l;
        let ar1h = dcp * ar1 - dsp * ai1;
        ai1 = dcp * ai1 + dsp * ar1;
        ar1 = ar1h;
        for ik in 1..=idl1 {
            for m in 1..=mp {
                ch[idx3(m, ik, l, mdimch, idl1)] =
                    cc[idx3(m, ik, 1, mdimcc, idl1)] + ar1 * cc[idx3(m, ik, 2, mdimcc, idl1)];
                ch[idx3(m, ik, lc, mdimch, idl1)] = ai1 * cc[idx3(m, ik, ip, mdimcc, idl1)];
            }
        }
        let dc2 = ar1;
        let ds2 = ai1;
        let mut ar2 = ar1;
        let mut ai2 = ai1;
        for j in 3..=ipph {
            let jc = ipp2 - j;
            let ar2h = dc2 * ar2 - ds2 * ai2;
            ai2 = dc2 * ai2 + ds2 * ar2;
            ar2 = ar2h;
            for ik in 1..=idl1 {
                for m in 1..=mp {
                    ch[idx3(m, ik, l, mdimch, idl1)] += ar2 * cc[idx3(m, ik, j, mdimcc, idl1)];
                    ch[idx3(m, ik, lc, mdimch, idl1)] += ai2 * cc[idx3(m, ik, jc, mdimcc, idl1)];
                }
            }
        }
    }

    for j in 2..=ipph {
        for ik in 1..=idl1 {
            for m in 1..=mp {
                ch[idx3(m, ik, 1, mdimch, idl1)] += cc[idx3(m, ik, j, mdimcc, idl1)];
            }
        }
    }

    if ido >= l1 {
        for k in 1..=l1 {
            for i in 1..=ido {
                for m in 1..=mp {
                    cc[idx4(m, i, 1, k, mdimcc, ido, ip)] = ch[idx4(m, i, k, 1, mdimch, ido, l1)];
                }
            }
        }
    } else {
        for i in 1..=ido {
            for k in 1..=l1 {
                for m in 1..=mp {
                    cc[idx4(m, i, 1, k, mdimcc, ido, ip)] = ch[idx4(m, i, k, 1, mdimch, ido, l1)];
                }
            }
        }
    }

    for j in 2..=ipph {
        let jc = ipp2 - j;
        let j2 = j + j;
        for k in 1..=l1 {
            for m in 1..=mp {
                cc[idx4(m, ido, j2 - 2, k, mdimcc, ido, ip)] =
                    ch[idx4(m, 1, k, j, mdimch, ido, l1)];
                cc[idx4(m, 1, j2 - 1, k, mdimcc, ido, ip)] = ch[idx4(m, 1, k, jc, mdimch, ido, l1)];
            }
        }
    }

    if ido == 1 {
        return;
    }

    if nbd >= l1 {
        for j in 2..=ipph {
            let jc = ipp2 - j;
            let j2 = j + j;
            for k in 1..=l1 {
                let mut i = 3;
                while i <= ido {
                    let ic = idp2 - i;
                    for m in 1..=mp {
                        let hj_im1 = ch[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                        let hj_i = ch[idx4(m, i, k, j, mdimch, ido, l1)];
                        let hc_im1 = ch[idx4(m, i - 1, k, jc, mdimch, ido, l1)];
                        let hc_i = ch[idx4(m, i, k, jc, mdimch, ido, l1)];
                        cc[idx4(m, i - 1, j2 - 1, k, mdimcc, ido, ip)] = hj_im1 + hc_im1;
                        cc[idx4(m, ic - 1, j2 - 2, k, mdimcc, ido, ip)] = hj_im1 - hc_im1;
                        cc[idx4(m, i, j2 - 1, k, mdimcc, ido, ip)] = hj_i + hc_i;
                        cc[idx4(m, ic, j2 - 2, k, mdimcc, ido, ip)] = hc_i - hj_i;
                    }
                    i += 2;
                }
            }
        }
    } else {
        for j in 2..=ipph {
            let jc = ipp2 - j;
            let j2 = j + j;
            let mut i = 3;
            while i <= ido {
                let ic = idp2 - i;
                for k in 1..=l1 {
                    for m in 1..=mp {
                        let hj_im1 = ch[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                        let hj_i = ch[idx4(m, i, k, j, mdimch, ido, l1)];
                        let hc_im1 = ch[idx4(m, i - 1, k, jc, mdimch, ido, l1)];
                        let hc_i = ch[idx4(m, i, k, jc, mdimch, ido, l1)];
                        cc[idx4(m, i - 1, j2 - 1, k, mdimcc, ido, ip)] = hj_im1 + hc_im1;
                        cc[idx4(m, ic - 1, j2 - 2, k, mdimcc, ido, ip)] = hj_im1 - hc_im1;
                        cc[idx4(m, i, j2 - 1, k, mdimcc, ido, ip)] = hj_i + hc_i;
                        cc[idx4(m, ic, j2 - 2, k, mdimcc, ido, ip)] = hc_i - hj_i;
                    }
                }
                i += 2;
            }
        }
    }
}

fn hradf3(
    mp: usize,
    ido: usize,
    l1: usize,
    cc: &[f32],
    mdimcc: usize,
    ch: &mut [f32],
    mdimch: usize,
    wa1: &[f32],
    wa2: &[f32],
) {
    let arg = 2.0 * pimach() / 3.0;
    let taur = arg.cos();
    let taui = arg.sin();
    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, 1, k, mdimch, ido, 3)] = cc[idx4(m, 1, k, 1, mdimcc, ido, l1)]
                + (cc[idx4(m, 1, k, 2, mdimcc, ido, l1)] + cc[idx4(m, 1, k, 3, mdimcc, ido, l1)]);
            ch[idx4(m, 1, 3, k, mdimch, ido, 3)] = taui
                * (cc[idx4(m, 1, k, 3, mdimcc, ido, l1)] - cc[idx4(m, 1, k, 2, mdimcc, ido, l1)]);
            ch[idx4(m, ido, 2, k, mdimch, ido, 3)] = cc[idx4(m, 1, k, 1, mdimcc, ido, l1)]
                + taur
                    * (cc[idx4(m, 1, k, 2, mdimcc, ido, l1)]
                        + cc[idx4(m, 1, k, 3, mdimcc, ido, l1)]);
        }
    }
    if ido == 1 {
        return;
    }
    let idp2 = ido + 2;
    for k in 1..=l1 {
        let mut i = 3;
        while i <= ido {
            let ic = idp2 - i;
            for m in 1..=mp {
                let a1 = wa1[i - 3] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]
                    + wa1[i - 2] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)];
                let a2 = wa1[i - 3] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]
                    - wa1[i - 2] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)];
                let b1 = wa2[i - 3] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)]
                    + wa2[i - 2] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)];
                let b2 = wa2[i - 3] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)]
                    - wa2[i - 2] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)];

                ch[idx4(m, i - 1, 1, k, mdimch, ido, 3)] =
                    cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] + a1 + b1;
                ch[idx4(m, i, 1, k, mdimch, ido, 3)] =
                    cc[idx4(m, i, k, 1, mdimcc, ido, l1)] + a2 + b2;

                ch[idx4(m, i - 1, 3, k, mdimch, ido, 3)] =
                    (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] + taur * (a1 + b1))
                        + taui * (a2 - b2);
                ch[idx4(m, ic - 1, 2, k, mdimch, ido, 3)] =
                    (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] + taur * (a1 + b1))
                        - taui * (a2 - b2);
                ch[idx4(m, i, 3, k, mdimch, ido, 3)] =
                    (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] + taur * (a2 + b2)) + taui * (b1 - a1);
                ch[idx4(m, ic, 2, k, mdimch, ido, 3)] =
                    taui * (b1 - a1) - (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] + taur * (a2 + b2));
            }
            i += 2;
        }
    }
}

fn hradf4(
    mp: usize,
    ido: usize,
    l1: usize,
    cc: &[f32],
    mdimcc: usize,
    ch: &mut [f32],
    mdimch: usize,
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    let hsqt2 = 2.0_f32.sqrt() * 0.5_f32;
    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, 1, k, mdimch, ido, 4)] = (cc[idx4(m, 1, k, 2, mdimcc, ido, l1)]
                + cc[idx4(m, 1, k, 4, mdimcc, ido, l1)])
                + (cc[idx4(m, 1, k, 1, mdimcc, ido, l1)] + cc[idx4(m, 1, k, 3, mdimcc, ido, l1)]);
            ch[idx4(m, ido, 4, k, mdimch, ido, 4)] = (cc[idx4(m, 1, k, 1, mdimcc, ido, l1)]
                + cc[idx4(m, 1, k, 3, mdimcc, ido, l1)])
                - (cc[idx4(m, 1, k, 2, mdimcc, ido, l1)] + cc[idx4(m, 1, k, 4, mdimcc, ido, l1)]);
            ch[idx4(m, ido, 2, k, mdimch, ido, 4)] =
                cc[idx4(m, 1, k, 1, mdimcc, ido, l1)] - cc[idx4(m, 1, k, 3, mdimcc, ido, l1)];
            ch[idx4(m, 1, 3, k, mdimch, ido, 4)] =
                cc[idx4(m, 1, k, 4, mdimcc, ido, l1)] - cc[idx4(m, 1, k, 2, mdimcc, ido, l1)];
        }
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        let idp2 = ido + 2;
        for k in 1..=l1 {
            let mut i = 3;
            while i <= ido {
                let ic = idp2 - i;
                for m in 1..=mp {
                    let a1 = wa1[i - 3] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]
                        + wa1[i - 2] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)];
                    let a2 = wa1[i - 3] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]
                        - wa1[i - 2] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)];
                    let b1 = wa2[i - 3] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)]
                        + wa2[i - 2] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)];
                    let b2 = wa2[i - 3] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)]
                        - wa2[i - 2] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)];
                    let c1 = wa3[i - 3] * cc[idx4(m, i - 1, k, 4, mdimcc, ido, l1)]
                        + wa3[i - 2] * cc[idx4(m, i, k, 4, mdimcc, ido, l1)];
                    let c2 = wa3[i - 3] * cc[idx4(m, i, k, 4, mdimcc, ido, l1)]
                        - wa3[i - 2] * cc[idx4(m, i - 1, k, 4, mdimcc, ido, l1)];

                    ch[idx4(m, i - 1, 1, k, mdimch, ido, 4)] =
                        (a1 + c1) + (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] + b1);
                    ch[idx4(m, ic - 1, 4, k, mdimch, ido, 4)] =
                        (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] + b1) - (a1 + c1);
                    ch[idx4(m, i, 1, k, mdimch, ido, 4)] =
                        (a2 + c2) + (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] + b2);
                    ch[idx4(m, ic, 4, k, mdimch, ido, 4)] =
                        (a2 + c2) - (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] + b2);

                    ch[idx4(m, i - 1, 3, k, mdimch, ido, 4)] =
                        (a2 - c2) + (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] - b1);
                    ch[idx4(m, ic - 1, 2, k, mdimch, ido, 4)] =
                        (cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)] - b1) - (a2 - c2);
                    ch[idx4(m, i, 3, k, mdimch, ido, 4)] =
                        (c1 - a1) + (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] - b2);
                    ch[idx4(m, ic, 2, k, mdimch, ido, 4)] =
                        (c1 - a1) - (cc[idx4(m, i, k, 1, mdimcc, ido, l1)] - b2);
                }
                i += 2;
            }
        }
    }
    if ido % 2 == 1 {
        return;
    }
    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, ido, 1, k, mdimch, ido, 4)] = hsqt2
                * (cc[idx4(m, ido, k, 2, mdimcc, ido, l1)]
                    - cc[idx4(m, ido, k, 4, mdimcc, ido, l1)])
                + cc[idx4(m, ido, k, 1, mdimcc, ido, l1)];
            ch[idx4(m, ido, 3, k, mdimch, ido, 4)] = cc[idx4(m, ido, k, 1, mdimcc, ido, l1)]
                - hsqt2
                    * (cc[idx4(m, ido, k, 2, mdimcc, ido, l1)]
                        - cc[idx4(m, ido, k, 4, mdimcc, ido, l1)]);
            ch[idx4(m, 1, 2, k, mdimch, ido, 4)] = -hsqt2
                * (cc[idx4(m, ido, k, 2, mdimcc, ido, l1)]
                    + cc[idx4(m, ido, k, 4, mdimcc, ido, l1)])
                - cc[idx4(m, ido, k, 3, mdimcc, ido, l1)];
            ch[idx4(m, 1, 4, k, mdimch, ido, 4)] = -hsqt2
                * (cc[idx4(m, ido, k, 2, mdimcc, ido, l1)]
                    + cc[idx4(m, ido, k, 4, mdimcc, ido, l1)])
                + cc[idx4(m, ido, k, 3, mdimcc, ido, l1)];
        }
    }
}

fn hradf5(
    mp: usize,
    ido: usize,
    l1: usize,
    cc: &[f32],
    mdimcc: usize,
    ch: &mut [f32],
    mdimch: usize,
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
    wa4: &[f32],
) {
    let arg = 2.0 * pimach() / 5.0;
    let tr11 = arg.cos();
    let ti11 = arg.sin();
    let tr12 = (2.0 * arg).cos();
    let ti12 = (2.0 * arg).sin();

    for k in 1..=l1 {
        for m in 1..=mp {
            let c1 = cc[idx4(m, 1, k, 1, mdimcc, ido, l1)];
            let c2 = cc[idx4(m, 1, k, 2, mdimcc, ido, l1)];
            let c3 = cc[idx4(m, 1, k, 3, mdimcc, ido, l1)];
            let c4 = cc[idx4(m, 1, k, 4, mdimcc, ido, l1)];
            let c5 = cc[idx4(m, 1, k, 5, mdimcc, ido, l1)];
            ch[idx4(m, 1, 1, k, mdimch, ido, 5)] = c1 + (c5 + c2) + (c4 + c3);
            ch[idx4(m, ido, 2, k, mdimch, ido, 5)] = c1 + tr11 * (c5 + c2) + tr12 * (c4 + c3);
            ch[idx4(m, 1, 3, k, mdimch, ido, 5)] = ti11 * (c5 - c2) + ti12 * (c4 - c3);
            ch[idx4(m, ido, 4, k, mdimch, ido, 5)] = c1 + tr12 * (c5 + c2) + tr11 * (c4 + c3);
            ch[idx4(m, 1, 5, k, mdimch, ido, 5)] = ti12 * (c5 - c2) - ti11 * (c4 - c3);
        }
    }
    if ido == 1 {
        return;
    }
    let idp2 = ido + 2;
    for k in 1..=l1 {
        let mut i = 3;
        while i <= ido {
            let ic = idp2 - i;
            for m in 1..=mp {
                let a1 = wa1[i - 3] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)]
                    + wa1[i - 2] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)];
                let a2 = wa1[i - 3] * cc[idx4(m, i, k, 2, mdimcc, ido, l1)]
                    - wa1[i - 2] * cc[idx4(m, i - 1, k, 2, mdimcc, ido, l1)];
                let b1 = wa2[i - 3] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)]
                    + wa2[i - 2] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)];
                let b2 = wa2[i - 3] * cc[idx4(m, i, k, 3, mdimcc, ido, l1)]
                    - wa2[i - 2] * cc[idx4(m, i - 1, k, 3, mdimcc, ido, l1)];
                let c1 = wa3[i - 3] * cc[idx4(m, i - 1, k, 4, mdimcc, ido, l1)]
                    + wa3[i - 2] * cc[idx4(m, i, k, 4, mdimcc, ido, l1)];
                let c2 = wa3[i - 3] * cc[idx4(m, i, k, 4, mdimcc, ido, l1)]
                    - wa3[i - 2] * cc[idx4(m, i - 1, k, 4, mdimcc, ido, l1)];
                let d1 = wa4[i - 3] * cc[idx4(m, i - 1, k, 5, mdimcc, ido, l1)]
                    + wa4[i - 2] * cc[idx4(m, i, k, 5, mdimcc, ido, l1)];
                let d2 = wa4[i - 3] * cc[idx4(m, i, k, 5, mdimcc, ido, l1)]
                    - wa4[i - 2] * cc[idx4(m, i - 1, k, 5, mdimcc, ido, l1)];
                let e1 = cc[idx4(m, i - 1, k, 1, mdimcc, ido, l1)];
                let e2 = cc[idx4(m, i, k, 1, mdimcc, ido, l1)];

                ch[idx4(m, i - 1, 1, k, mdimch, ido, 5)] = e1 + (a1 + d1) + (b1 + c1);
                ch[idx4(m, i, 1, k, mdimch, ido, 5)] = e2 + (a2 + d2) + (b2 + c2);

                let t31 =
                    e1 + tr11 * (a1 + d1) + tr12 * (b1 + c1) + ti11 * (a2 - d2) + ti12 * (b2 - c2);
                let t32 = e1 + tr11 * (a1 + d1) + tr12 * (b1 + c1)
                    - (ti11 * (a2 - d2) + ti12 * (b2 - c2));
                ch[idx4(m, i - 1, 3, k, mdimch, ido, 5)] = t31;
                ch[idx4(m, ic - 1, 2, k, mdimch, ido, 5)] = t32;

                let u31 =
                    e2 + tr11 * (a2 + d2) + tr12 * (b2 + c2) + ti11 * (d1 - a1) + ti12 * (c1 - b1);
                let u32 = (ti11 * (d1 - a1) + ti12 * (c1 - b1))
                    - (e2 + tr11 * (a2 + d2) + tr12 * (b2 + c2));
                ch[idx4(m, i, 3, k, mdimch, ido, 5)] = u31;
                ch[idx4(m, ic, 2, k, mdimch, ido, 5)] = u32;

                let t51 =
                    e1 + tr12 * (a1 + d1) + tr11 * (b1 + c1) + ti12 * (a2 - d2) - ti11 * (b2 - c2);
                let t52 = e1 + tr12 * (a1 + d1) + tr11 * (b1 + c1)
                    - (ti12 * (a2 - d2) - ti11 * (b2 - c2));
                ch[idx4(m, i - 1, 5, k, mdimch, ido, 5)] = t51;
                ch[idx4(m, ic - 1, 4, k, mdimch, ido, 5)] = t52;

                let u51 =
                    e2 + tr12 * (a2 + d2) + tr11 * (b2 + c2) + ti12 * (d1 - a1) - ti11 * (c1 - b1);
                let u52 = (ti12 * (d1 - a1) - ti11 * (c1 - b1))
                    - (e2 + tr12 * (a2 + d2) + tr11 * (b2 + c2));
                ch[idx4(m, i, 5, k, mdimch, ido, 5)] = u51;
                ch[idx4(m, ic, 4, k, mdimch, ido, 5)] = u52;
            }
            i += 2;
        }
    }
}
