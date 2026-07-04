use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

fn hrfftb_reference_rowwise(m: usize, n: usize, r: &mut [f32], row_offset: usize) {
    let two_pi = 2.0_f64 * std::f64::consts::PI;
    let l = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    let input = r.to_vec();

    let mut out = vec![0.0_f32; m * n];

    for row in 0..m {
        let global_row = row_offset + row;

        let coeff0 = input[row];
        let nyquist = if n % 2 == 0 {
            Some(input[(n - 1) * m + row] as f64)
        } else {
            None
        };

        let debug_coeffs: Vec<f32> = (0..n).map(|j| input[j * m + row]).collect();

        for j in 0..n {
            let mut sum = coeff0 as f64;

            if let Some(nyquist) = nyquist {
                sum += if j % 2 == 0 { nyquist } else { -nyquist };
            }

            for k in 2..=l {
                let angle = two_pi * ((k - 1) as f64) * (j as f64) / (n as f64);
                let cos_coeff = input[(2 * k - 3) * m + row] as f64;
                let sin_coeff = input[(2 * k - 2) * m + row] as f64;
                sum += 2.0 * cos_coeff * angle.cos();
                sum -= 2.0 * sin_coeff * angle.sin();
            }

            out[j * m + row] = sum as f32;
        }

        let _ = global_row;
        let _ = debug_coeffs;
    }

    r.copy_from_slice(&out);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HrfftbMode {
    Auto,
    KernelOnly,
    ReferenceOnly,
}

fn hrfftb_auto_should_fallback(n: usize, nf: usize) -> bool {
    let _ = (n, nf);
    false
}

/// Rust entry point for `hrfftb_impl_with_mode`.
///
/// # Parameters
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
/// - `r`: Parameter `r` passed through to the routine.
/// - `whrfft`: Parameter `whrfft` passed through to the routine.
/// - `mode`: Selector controlling which stored Legendre recurrence table is generated.
///
/// # Returns
/// A Python result containing the values produced by this routine.
pub fn hrfftb_impl_with_mode(
    m: usize,
    n: usize,
    r: &mut [f32],
    whrfft: &[f32],
    mode: HrfftbMode,
) -> PyResult<()> {
    if n == 1 {
        return Ok(());
    }
    if r.len() != m * n {
        return Err(PyValueError::new_err("hrfftb_impl: r length mismatch"));
    }

    if whrfft.len() < n + 15 {
        return Err(PyValueError::new_err(
            "hrfftb_impl: whrfft length too small",
        ));
    }

    let mut work = vec![0.0_f32; m * n];
    let wa = &whrfft[..n];
    let fac = &whrfft[n..];

    let nf = fac[1] as usize;
    let should_fallback = hrfftb_auto_should_fallback(n, nf);

    match mode {
        HrfftbMode::ReferenceOnly => {
            hrfftb_reference_rowwise(m, n, r, 0);
            Ok(())
        }
        HrfftbMode::KernelOnly => hrftb1(m, n, r, m, &mut work, wa, fac),
        HrfftbMode::Auto => {
            if should_fallback {
                hrfftb_reference_rowwise(m, n, r, 0);
                Ok(())
            } else {
                hrftb1(m, n, r, m, &mut work, wa, fac)
            }
        }
    }
}

/// Core Rust implementation of `hrfftb`.
///
/// # Parameters
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
/// - `r`: Parameter `r` passed through to the routine.
/// - `_whrfft`: Parameter `_whrfft` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn hrfftb_impl(m: usize, n: usize, r: &mut [f32], _whrfft: &[f32]) -> PyResult<()> {
    hrfftb_impl_with_mode(m, n, r, _whrfft, HrfftbMode::Auto)
}

/// Core Rust implementation of `hrfftb_kernel_only`.
///
/// # Parameters
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
/// - `r`: Parameter `r` passed through to the routine.
/// - `whrfft`: Parameter `whrfft` passed through to the routine.
///
/// # Returns
/// A Python result containing the values produced by this routine.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn hrfftb_kernel_only_impl(m: usize, n: usize, r: &mut [f32], whrfft: &[f32]) -> PyResult<()> {
    hrfftb_impl_with_mode(m, n, r, whrfft, HrfftbMode::KernelOnly)
}

#[pyfunction]
/// Rust entry point for `hrfftb`.
///
/// # Parameters
/// - `r`: Parameter `r` passed through to the routine.
/// - `whrfft`: Parameter `whrfft` passed through to the routine.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
///
/// # Returns
/// A one-dimensional NumPy array containing the computed workspace.
pub fn hrfftb<'py>(
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
                "hrfftb: input shape does not match m,n",
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
                "hrfftb: flat input length does not match m*n",
            ));
        }
        flat.to_vec()
    } else {
        return Err(PyValueError::new_err(
            "hrfftb expects rank-1 or rank-2 input",
        ));
    };
    let wsave = whrfft.as_slice()?;
    hrfftb_impl(m, n, &mut data, wsave)?;
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
/// Rust entry point for `hrfftb_kernel_only`.
///
/// # Parameters
/// - `r`: Parameter `r` passed through to the routine.
/// - `whrfft`: Parameter `whrfft` passed through to the routine.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
///
/// # Returns
/// A one-dimensional NumPy array containing the computed workspace.
pub fn hrfftb_kernel_only<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    hrfftb_with_mode_py(py, r, whrfft, m, n, HrfftbMode::KernelOnly)
}

#[pyfunction]
/// Rust entry point for `hrfftb_reference_only`.
///
/// # Parameters
/// - `r`: Parameter `r` passed through to the routine.
/// - `whrfft`: Parameter `whrfft` passed through to the routine.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `n`: Total spherical harmonic degree.
///
/// # Returns
/// A one-dimensional NumPy array containing the computed workspace.
pub fn hrfftb_reference_only<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    hrfftb_with_mode_py(py, r, whrfft, m, n, HrfftbMode::ReferenceOnly)
}

fn hrfftb_with_mode_py<'py>(
    py: Python<'py>,
    r: PyReadonlyArrayDyn<'py, f32>,
    whrfft: PyReadonlyArrayDyn<'py, f32>,
    m: usize,
    n: usize,
    mode: HrfftbMode,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = r.shape();
    let mut data = if shape.len() == 2 {
        if shape[0] != m || shape[1] != n {
            return Err(PyValueError::new_err(
                "hrfftb: input shape does not match m,n",
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
                "hrfftb: flat input length does not match m*n",
            ));
        }
        flat.to_vec()
    } else {
        return Err(PyValueError::new_err(
            "hrfftb expects rank-1 or rank-2 input",
        ));
    };
    let wsave = whrfft.as_slice()?;
    hrfftb_impl_with_mode(m, n, &mut data, wsave, mode)?;
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

fn hrftb1(
    m: usize,
    n: usize,
    c: &mut [f32],
    mdimc: usize,
    ch: &mut [f32],
    wa: &[f32],
    fac: &[f32],
) -> PyResult<()> {
    let nf = fac[1] as usize;
    let mut na = 0usize;
    let mut l1 = 1usize;
    let mut iw = 1usize; // Fortran 1-based offset into wa

    for k1 in 1..=nf {
        let ip = fac[k1 + 1] as usize;
        let l2 = ip * l1;
        let ido = n / l2;
        let idl1 = ido * l1;
        match ip {
            4 => {
                let ix2 = iw + ido;
                let ix3 = ix2 + ido;
                if na == 0 {
                    hradb4(
                        m,
                        ido,
                        l1,
                        c,
                        mdimc,
                        ch,
                        m,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                        &wa[(ix3 - 1)..],
                    );
                } else {
                    hradb4(
                        m,
                        ido,
                        l1,
                        ch,
                        m,
                        c,
                        mdimc,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                        &wa[(ix3 - 1)..],
                    );
                }
                na = 1 - na;
            }
            2 => {
                if na == 0 {
                    hradb2(m, ido, l1, c, mdimc, ch, m, &wa[(iw - 1)..]);
                } else {
                    hradb2(m, ido, l1, ch, m, c, mdimc, &wa[(iw - 1)..]);
                }
                na = 1 - na;
            }
            3 => {
                let ix2 = iw + ido;
                if na == 0 {
                    hradb3(
                        m,
                        ido,
                        l1,
                        c,
                        mdimc,
                        ch,
                        m,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                    );
                } else {
                    hradb3(
                        m,
                        ido,
                        l1,
                        ch,
                        m,
                        c,
                        mdimc,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                    );
                }
                na = 1 - na;
            }
            5 => {
                let ix2 = iw + ido;
                let ix3 = ix2 + ido;
                let ix4 = ix3 + ido;
                if na == 0 {
                    hradb5(
                        m,
                        ido,
                        l1,
                        c,
                        mdimc,
                        ch,
                        m,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                        &wa[(ix3 - 1)..],
                        &wa[(ix4 - 1)..],
                    );
                } else {
                    hradb5(
                        m,
                        ido,
                        l1,
                        ch,
                        m,
                        c,
                        mdimc,
                        &wa[(iw - 1)..],
                        &wa[(ix2 - 1)..],
                        &wa[(ix3 - 1)..],
                        &wa[(ix4 - 1)..],
                    );
                }
                na = 1 - na;
            }
            _ => {
                if na == 0 {
                    hradbg(m, ido, ip, l1, idl1, c, mdimc, ch, m, &wa[(iw - 1)..]);
                } else {
                    hradbg(m, ido, ip, l1, idl1, ch, m, c, mdimc, &wa[(iw - 1)..]);
                }
                if ido == 1 {
                    na = 1 - na;
                }
            }
        }
        l1 = l2;
        iw += (ip - 1) * ido;
        let _ = idl1;
    }

    if na != 0 {
        c.copy_from_slice(ch);
    }
    Ok(())
}

fn hradb2(
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
            ch[idx4(m, 1, k, 1, mdimch, ido, l1)] =
                cc[idx4(m, 1, 1, k, mdimcc, ido, 2)] + cc[idx4(m, ido, 2, k, mdimcc, ido, 2)];
            ch[idx4(m, 1, k, 2, mdimch, ido, l1)] =
                cc[idx4(m, 1, 1, k, mdimcc, ido, 2)] - cc[idx4(m, ido, 2, k, mdimcc, ido, 2)];
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
                    ch[idx4(m, i - 1, k, 1, mdimch, ido, l1)] = cc
                        [idx4(m, i - 1, 1, k, mdimcc, ido, 2)]
                        + cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 2)];
                    ch[idx4(m, i, k, 1, mdimch, ido, l1)] = cc[idx4(m, i, 1, k, mdimcc, ido, 2)]
                        - cc[idx4(m, ic, 2, k, mdimcc, ido, 2)];
                    ch[idx4(m, i - 1, k, 2, mdimch, ido, l1)] = wa1[i - 3]
                        * (cc[idx4(m, i - 1, 1, k, mdimcc, ido, 2)]
                            - cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 2)])
                        - wa1[i - 2]
                            * (cc[idx4(m, i, 1, k, mdimcc, ido, 2)]
                                + cc[idx4(m, ic, 2, k, mdimcc, ido, 2)]);
                    ch[idx4(m, i, k, 2, mdimch, ido, l1)] = wa1[i - 3]
                        * (cc[idx4(m, i, 1, k, mdimcc, ido, 2)]
                            + cc[idx4(m, ic, 2, k, mdimcc, ido, 2)])
                        + wa1[i - 2]
                            * (cc[idx4(m, i - 1, 1, k, mdimcc, ido, 2)]
                                - cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 2)]);
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
            ch[idx4(m, ido, k, 1, mdimch, ido, l1)] = 2.0 * cc[idx4(m, ido, 1, k, mdimcc, ido, 2)];
            ch[idx4(m, ido, k, 2, mdimch, ido, l1)] = -2.0 * cc[idx4(m, 1, 2, k, mdimcc, ido, 2)];
        }
    }
}

fn hradb3(
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

    if ido == 1 {
        for k in 1..=l1 {
            for m in 1..=mp {
                let cc11 = cc[idx4(m, 1, 1, k, mdimcc, ido, 3)];
                let cc12n = cc[idx4(m, ido, 2, k, mdimcc, ido, 3)];
                let cc13 = cc[idx4(m, 1, 3, k, mdimcc, ido, 3)];
                ch[idx4(m, 1, k, 1, mdimch, ido, l1)] = cc11 + 2.0 * cc12n;
                ch[idx4(m, 1, k, 2, mdimch, ido, l1)] =
                    cc11 + (2.0 * taur) * cc12n - (2.0 * taui) * cc13;
                ch[idx4(m, 1, k, 3, mdimch, ido, l1)] =
                    cc11 + (2.0 * taur) * cc12n + (2.0 * taui) * cc13;
            }
        }
        return;
    }

    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, k, 1, mdimch, ido, l1)] =
                cc[idx4(m, 1, 1, k, mdimcc, ido, 3)] + 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 3)];
            ch[idx4(m, 1, k, 2, mdimch, ido, l1)] = cc[idx4(m, 1, 1, k, mdimcc, ido, 3)]
                + (2.0 * taur) * cc[idx4(m, ido, 2, k, mdimcc, ido, 3)]
                - (2.0 * taui) * cc[idx4(m, 1, 3, k, mdimcc, ido, 3)];
            ch[idx4(m, 1, k, 3, mdimch, ido, l1)] = cc[idx4(m, 1, 1, k, mdimcc, ido, 3)]
                + (2.0 * taur) * cc[idx4(m, ido, 2, k, mdimcc, ido, 3)]
                + (2.0 * taui) * cc[idx4(m, 1, 3, k, mdimcc, ido, 3)];
        }
    }
    let idp2 = ido + 2;
    for k in 1..=l1 {
        let mut i = 3;
        while i <= ido {
            let ic = idp2 - i;
            for m in 1..=mp {
                let t1 = cc[idx4(m, i - 1, 1, k, mdimcc, ido, 3)]
                    + taur
                        * (cc[idx4(m, i - 1, 3, k, mdimcc, ido, 3)]
                            + cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 3)]);
                let t2 = taui
                    * (cc[idx4(m, i, 3, k, mdimcc, ido, 3)]
                        + cc[idx4(m, ic, 2, k, mdimcc, ido, 3)]);
                let t3 = cc[idx4(m, i, 1, k, mdimcc, ido, 3)]
                    + taur
                        * (cc[idx4(m, i, 3, k, mdimcc, ido, 3)]
                            - cc[idx4(m, ic, 2, k, mdimcc, ido, 3)]);
                let t4 = taui
                    * (cc[idx4(m, i - 1, 3, k, mdimcc, ido, 3)]
                        - cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 3)]);

                ch[idx4(m, i - 1, k, 1, mdimch, ido, l1)] = cc
                    [idx4(m, i - 1, 1, k, mdimcc, ido, 3)]
                    + (cc[idx4(m, i - 1, 3, k, mdimcc, ido, 3)]
                        + cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 3)]);
                ch[idx4(m, i, k, 1, mdimch, ido, l1)] = cc[idx4(m, i, 1, k, mdimcc, ido, 3)]
                    + (cc[idx4(m, i, 3, k, mdimcc, ido, 3)]
                        - cc[idx4(m, ic, 2, k, mdimcc, ido, 3)]);

                ch[idx4(m, i - 1, k, 2, mdimch, ido, l1)] =
                    wa1[i - 3] * (t1 - t2) - wa1[i - 2] * (t3 + t4);
                ch[idx4(m, i, k, 2, mdimch, ido, l1)] =
                    wa1[i - 3] * (t3 + t4) + wa1[i - 2] * (t1 - t2);

                ch[idx4(m, i - 1, k, 3, mdimch, ido, l1)] =
                    wa2[i - 3] * (t1 + t2) - wa2[i - 2] * (t3 - t4);
                ch[idx4(m, i, k, 3, mdimch, ido, l1)] =
                    wa2[i - 3] * (t3 - t4) + wa2[i - 2] * (t1 + t2);
            }
            i += 2;
        }
    }
}

fn hradb4(
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
    let sqrt2 = 2.0_f32.sqrt();

    if ido == 1 {
        for k in 1..=l1 {
            for m in 1..=mp {
                let cc11 = cc[idx4(m, 1, 1, k, mdimcc, ido, 4)];
                let cc24 = cc[idx4(m, ido, 4, k, mdimcc, ido, 4)];
                let cc22 = cc[idx4(m, ido, 2, k, mdimcc, ido, 4)];
                let cc13 = cc[idx4(m, 1, 3, k, mdimcc, ido, 4)];
                ch[idx4(m, 1, k, 3, mdimch, ido, l1)] = (cc11 + cc24) - 2.0 * cc22;
                ch[idx4(m, 1, k, 1, mdimch, ido, l1)] = (cc11 + cc24) + 2.0 * cc22;
                ch[idx4(m, 1, k, 4, mdimch, ido, l1)] = (cc11 - cc24) + 2.0 * cc13;
                ch[idx4(m, 1, k, 2, mdimch, ido, l1)] = (cc11 - cc24) - 2.0 * cc13;
            }
        }
        return;
    }

    for k in 1..=l1 {
        for m in 1..=mp {
            ch[idx4(m, 1, k, 3, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 4)]
                + cc[idx4(m, ido, 4, k, mdimcc, ido, 4)])
                - 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 4)];
            ch[idx4(m, 1, k, 1, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 4)]
                + cc[idx4(m, ido, 4, k, mdimcc, ido, 4)])
                + 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 4)];
            ch[idx4(m, 1, k, 4, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 4)]
                - cc[idx4(m, ido, 4, k, mdimcc, ido, 4)])
                + 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 4)];
            ch[idx4(m, 1, k, 2, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 4)]
                - cc[idx4(m, ido, 4, k, mdimcc, ido, 4)])
                - 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 4)];
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
                    let c11 = cc[idx4(m, i - 1, 1, k, mdimcc, ido, 4)];
                    let c14 = cc[idx4(m, ic - 1, 4, k, mdimcc, ido, 4)];
                    let c13 = cc[idx4(m, i - 1, 3, k, mdimcc, ido, 4)];
                    let c12 = cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 4)];
                    let s11 = cc[idx4(m, i, 1, k, mdimcc, ido, 4)];
                    let s14 = cc[idx4(m, ic, 4, k, mdimcc, ido, 4)];
                    let s13 = cc[idx4(m, i, 3, k, mdimcc, ido, 4)];
                    let s12 = cc[idx4(m, ic, 2, k, mdimcc, ido, 4)];

                    ch[idx4(m, i - 1, k, 1, mdimch, ido, l1)] = (c11 + c14) + (c13 + c12);
                    ch[idx4(m, i, k, 1, mdimch, ido, l1)] = (s11 - s14) + (s13 - s12);

                    let x21 = (c11 - c14) - (s13 + s12);
                    let y21 = (s11 + s14) + (c13 - c12);
                    ch[idx4(m, i - 1, k, 2, mdimch, ido, l1)] = wa1[i - 3] * x21 - wa1[i - 2] * y21;
                    ch[idx4(m, i, k, 2, mdimch, ido, l1)] = wa1[i - 3] * y21 + wa1[i - 2] * x21;

                    let x31 = (c11 + c14) - (c13 + c12);
                    let y31 = (s11 - s14) - (s13 - s12);
                    ch[idx4(m, i - 1, k, 3, mdimch, ido, l1)] = wa2[i - 3] * x31 - wa2[i - 2] * y31;
                    ch[idx4(m, i, k, 3, mdimch, ido, l1)] = wa2[i - 3] * y31 + wa2[i - 2] * x31;

                    let x41 = (c11 - c14) + (s13 + s12);
                    let y41 = (s11 + s14) - (c13 - c12);
                    ch[idx4(m, i - 1, k, 4, mdimch, ido, l1)] = wa3[i - 3] * x41 - wa3[i - 2] * y41;
                    ch[idx4(m, i, k, 4, mdimch, ido, l1)] = wa3[i - 3] * y41 + wa3[i - 2] * x41;
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
            ch[idx4(m, ido, k, 1, mdimch, ido, l1)] = 2.0
                * (cc[idx4(m, ido, 1, k, mdimcc, ido, 4)] + cc[idx4(m, ido, 3, k, mdimcc, ido, 4)]);
            ch[idx4(m, ido, k, 2, mdimch, ido, l1)] = sqrt2
                * ((cc[idx4(m, ido, 1, k, mdimcc, ido, 4)]
                    - cc[idx4(m, ido, 3, k, mdimcc, ido, 4)])
                    - (cc[idx4(m, 1, 2, k, mdimcc, ido, 4)]
                        + cc[idx4(m, 1, 4, k, mdimcc, ido, 4)]));
            ch[idx4(m, ido, k, 3, mdimch, ido, l1)] =
                2.0 * (cc[idx4(m, 1, 4, k, mdimcc, ido, 4)] - cc[idx4(m, 1, 2, k, mdimcc, ido, 4)]);
            ch[idx4(m, ido, k, 4, mdimch, ido, l1)] = -sqrt2
                * ((cc[idx4(m, ido, 1, k, mdimcc, ido, 4)]
                    - cc[idx4(m, ido, 3, k, mdimcc, ido, 4)])
                    + (cc[idx4(m, 1, 2, k, mdimcc, ido, 4)]
                        + cc[idx4(m, 1, 4, k, mdimcc, ido, 4)]));
        }
    }
}

fn hradb5(
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
            ch[idx4(m, 1, k, 1, mdimch, ido, l1)] = cc[idx4(m, 1, 1, k, mdimcc, ido, 5)]
                + 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 5)]
                + 2.0 * cc[idx4(m, ido, 4, k, mdimcc, ido, 5)];
            ch[idx4(m, 1, k, 2, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 5)]
                + tr11 * 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 5)]
                + tr12 * 2.0 * cc[idx4(m, ido, 4, k, mdimcc, ido, 5)])
                - (ti11 * 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 5)]
                    + ti12 * 2.0 * cc[idx4(m, 1, 5, k, mdimcc, ido, 5)]);
            ch[idx4(m, 1, k, 3, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 5)]
                + tr12 * 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 5)]
                + tr11 * 2.0 * cc[idx4(m, ido, 4, k, mdimcc, ido, 5)])
                - (ti12 * 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 5)]
                    - ti11 * 2.0 * cc[idx4(m, 1, 5, k, mdimcc, ido, 5)]);
            ch[idx4(m, 1, k, 4, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 5)]
                + tr12 * 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 5)]
                + tr11 * 2.0 * cc[idx4(m, ido, 4, k, mdimcc, ido, 5)])
                + (ti12 * 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 5)]
                    - ti11 * 2.0 * cc[idx4(m, 1, 5, k, mdimcc, ido, 5)]);
            ch[idx4(m, 1, k, 5, mdimch, ido, l1)] = (cc[idx4(m, 1, 1, k, mdimcc, ido, 5)]
                + tr11 * 2.0 * cc[idx4(m, ido, 2, k, mdimcc, ido, 5)]
                + tr12 * 2.0 * cc[idx4(m, ido, 4, k, mdimcc, ido, 5)])
                + (ti11 * 2.0 * cc[idx4(m, 1, 3, k, mdimcc, ido, 5)]
                    + ti12 * 2.0 * cc[idx4(m, 1, 5, k, mdimcc, ido, 5)]);
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
                let a = cc[idx4(m, i - 1, 1, k, mdimcc, ido, 5)];
                let b = cc[idx4(m, i, 1, k, mdimcc, ido, 5)];
                let c = cc[idx4(m, i - 1, 3, k, mdimcc, ido, 5)]
                    + cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 5)];
                let d =
                    cc[idx4(m, i, 3, k, mdimcc, ido, 5)] - cc[idx4(m, ic, 2, k, mdimcc, ido, 5)];
                let e = cc[idx4(m, i - 1, 5, k, mdimcc, ido, 5)]
                    + cc[idx4(m, ic - 1, 4, k, mdimcc, ido, 5)];
                let f =
                    cc[idx4(m, i, 5, k, mdimcc, ido, 5)] - cc[idx4(m, ic, 4, k, mdimcc, ido, 5)];
                let g =
                    cc[idx4(m, i, 3, k, mdimcc, ido, 5)] + cc[idx4(m, ic, 2, k, mdimcc, ido, 5)];
                let h = cc[idx4(m, i - 1, 3, k, mdimcc, ido, 5)]
                    - cc[idx4(m, ic - 1, 2, k, mdimcc, ido, 5)];
                let p =
                    cc[idx4(m, i, 5, k, mdimcc, ido, 5)] + cc[idx4(m, ic, 4, k, mdimcc, ido, 5)];
                let q = cc[idx4(m, i - 1, 5, k, mdimcc, ido, 5)]
                    - cc[idx4(m, ic - 1, 4, k, mdimcc, ido, 5)];

                ch[idx4(m, i - 1, k, 1, mdimch, ido, l1)] = a + c + e;
                ch[idx4(m, i, k, 1, mdimch, ido, l1)] = b + d + f;

                let x2 = (a + tr11 * c + tr12 * e) - (ti11 * g + ti12 * p);
                let y2 = (b + tr11 * d + tr12 * f) + (ti11 * h + ti12 * q);
                ch[idx4(m, i - 1, k, 2, mdimch, ido, l1)] = wa1[i - 3] * x2 - wa1[i - 2] * y2;
                ch[idx4(m, i, k, 2, mdimch, ido, l1)] = wa1[i - 3] * y2 + wa1[i - 2] * x2;

                let x3 = (a + tr12 * c + tr11 * e) - (ti12 * g - ti11 * p);
                let y3 = (b + tr12 * d + tr11 * f) + (ti12 * h - ti11 * q);
                ch[idx4(m, i - 1, k, 3, mdimch, ido, l1)] = wa2[i - 3] * x3 - wa2[i - 2] * y3;
                ch[idx4(m, i, k, 3, mdimch, ido, l1)] = wa2[i - 3] * y3 + wa2[i - 2] * x3;

                let x4 = (a + tr12 * c + tr11 * e) + (ti12 * g - ti11 * p);
                let y4 = (b + tr12 * d + tr11 * f) - (ti12 * h - ti11 * q);
                ch[idx4(m, i - 1, k, 4, mdimch, ido, l1)] = wa3[i - 3] * x4 - wa3[i - 2] * y4;
                ch[idx4(m, i, k, 4, mdimch, ido, l1)] = wa3[i - 3] * y4 + wa3[i - 2] * x4;

                let x5 = (a + tr11 * c + tr12 * e) + (ti11 * g + ti12 * p);
                let y5 = (b + tr11 * d + tr12 * f) - (ti11 * h + ti12 * q);
                ch[idx4(m, i - 1, k, 5, mdimch, ido, l1)] = wa4[i - 3] * x5 - wa4[i - 2] * y5;
                ch[idx4(m, i, k, 5, mdimch, ido, l1)] = wa4[i - 3] * y5 + wa4[i - 2] * x5;
            }
            i += 2;
        }
    }
}

fn hradbg(
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
    let idp2 = ido + 2;
    let nbd = (ido - 1) / 2;
    let ipp2 = ip + 2;
    let ipph = (ip + 1) / 2;

    let mut cbuf = cc.to_vec();
    let mut hbuf = ch.to_vec();

    if ido == 1 {
        for k in 1..=l1 {
            for m in 1..=mp {
                hbuf[idx4(m, 1, k, 1, mdimch, ido, l1)] = cbuf[idx4(m, 1, 1, k, mdimcc, ido, ip)];
            }
        }

        for j in 2..=ipph {
            let jc = ipp2 - j;
            let j2 = j + j;
            for k in 1..=l1 {
                for m in 1..=mp {
                    hbuf[idx4(m, 1, k, j, mdimch, ido, l1)] =
                        2.0 * cbuf[idx4(m, ido, j2 - 2, k, mdimcc, ido, ip)];
                    hbuf[idx4(m, 1, k, jc, mdimch, ido, l1)] =
                        2.0 * cbuf[idx4(m, 1, j2 - 1, k, mdimcc, ido, ip)];
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
                    cbuf[idx3(m, ik, l, mdimcc, idl1)] = hbuf[idx3(m, ik, 1, mdimch, idl1)]
                        + ar1 * hbuf[idx3(m, ik, 2, mdimch, idl1)];
                    cbuf[idx3(m, ik, lc, mdimcc, idl1)] = ai1 * hbuf[idx3(m, ik, ip, mdimch, idl1)];
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
                        cbuf[idx3(m, ik, l, mdimcc, idl1)] +=
                            ar2 * hbuf[idx3(m, ik, j, mdimch, idl1)];
                        cbuf[idx3(m, ik, lc, mdimcc, idl1)] +=
                            ai2 * hbuf[idx3(m, ik, jc, mdimch, idl1)];
                    }
                }
            }
        }

        for j in 2..=ipph {
            for ik in 1..=idl1 {
                for m in 1..=mp {
                    hbuf[idx3(m, ik, 1, mdimch, idl1)] += hbuf[idx3(m, ik, j, mdimch, idl1)];
                }
            }
        }

        for j in 2..=ipph {
            let jc = ipp2 - j;
            for k in 1..=l1 {
                for m in 1..=mp {
                    hbuf[idx4(m, 1, k, j, mdimch, ido, l1)] = cbuf
                        [idx4(m, 1, k, j, mdimcc, ido, l1)]
                        - cbuf[idx4(m, 1, k, jc, mdimcc, ido, l1)];
                    hbuf[idx4(m, 1, k, jc, mdimch, ido, l1)] = cbuf
                        [idx4(m, 1, k, j, mdimcc, ido, l1)]
                        + cbuf[idx4(m, 1, k, jc, mdimcc, ido, l1)];
                }
            }
        }

        ch.copy_from_slice(&hbuf);
        return;
    }

    if ido >= l1 {
        for k in 1..=l1 {
            for i in 1..=ido {
                for m in 1..=mp {
                    hbuf[idx4(m, i, k, 1, mdimch, ido, l1)] =
                        cbuf[idx4(m, i, 1, k, mdimcc, ido, ip)];
                }
            }
        }
    } else {
        for i in 1..=ido {
            for k in 1..=l1 {
                for m in 1..=mp {
                    hbuf[idx4(m, i, k, 1, mdimch, ido, l1)] =
                        cbuf[idx4(m, i, 1, k, mdimcc, ido, ip)];
                }
            }
        }
    }

    for j in 2..=ipph {
        let jc = ipp2 - j;
        let j2 = j + j;
        for k in 1..=l1 {
            for m in 1..=mp {
                hbuf[idx4(m, 1, k, j, mdimch, ido, l1)] =
                    2.0 * cbuf[idx4(m, ido, j2 - 2, k, mdimcc, ido, ip)];
                hbuf[idx4(m, 1, k, jc, mdimch, ido, l1)] =
                    2.0 * cbuf[idx4(m, 1, j2 - 1, k, mdimcc, ido, ip)];
            }
        }
    }

    if ido != 1 {
        if nbd >= l1 {
            for j in 2..=ipph {
                let jc = ipp2 - j;
                for k in 1..=l1 {
                    let mut i = 3;
                    while i <= ido {
                        let ic = idp2 - i;
                        for m in 1..=mp {
                            hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, 2 * j - 1, k, mdimcc, ido, ip)]
                                + cbuf[idx4(m, ic - 1, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i - 1, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, 2 * j - 1, k, mdimcc, ido, ip)]
                                - cbuf[idx4(m, ic - 1, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, 2 * j - 1, k, mdimcc, ido, ip)]
                                - cbuf[idx4(m, ic, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, 2 * j - 1, k, mdimcc, ido, ip)]
                                + cbuf[idx4(m, ic, 2 * j - 2, k, mdimcc, ido, ip)];
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
                    let ic = idp2 - i;
                    for k in 1..=l1 {
                        for m in 1..=mp {
                            hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, 2 * j - 1, k, mdimcc, ido, ip)]
                                + cbuf[idx4(m, ic - 1, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i - 1, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, 2 * j - 1, k, mdimcc, ido, ip)]
                                - cbuf[idx4(m, ic - 1, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, 2 * j - 1, k, mdimcc, ido, ip)]
                                - cbuf[idx4(m, ic, 2 * j - 2, k, mdimcc, ido, ip)];
                            hbuf[idx4(m, i, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, 2 * j - 1, k, mdimcc, ido, ip)]
                                + cbuf[idx4(m, ic, 2 * j - 2, k, mdimcc, ido, ip)];
                        }
                    }
                    i += 2;
                }
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
                cbuf[idx3(m, ik, l, mdimcc, idl1)] =
                    hbuf[idx3(m, ik, 1, mdimch, idl1)] + ar1 * hbuf[idx3(m, ik, 2, mdimch, idl1)];
                cbuf[idx3(m, ik, lc, mdimcc, idl1)] = ai1 * hbuf[idx3(m, ik, ip, mdimch, idl1)];
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
                    cbuf[idx3(m, ik, l, mdimcc, idl1)] += ar2 * hbuf[idx3(m, ik, j, mdimch, idl1)];
                    cbuf[idx3(m, ik, lc, mdimcc, idl1)] +=
                        ai2 * hbuf[idx3(m, ik, jc, mdimch, idl1)];
                }
            }
        }
    }

    for j in 2..=ipph {
        for ik in 1..=idl1 {
            for m in 1..=mp {
                hbuf[idx3(m, ik, 1, mdimch, idl1)] += hbuf[idx3(m, ik, j, mdimch, idl1)];
            }
        }
    }

    for j in 2..=ipph {
        let jc = ipp2 - j;
        for k in 1..=l1 {
            for m in 1..=mp {
                hbuf[idx4(m, 1, k, j, mdimch, ido, l1)] = cbuf[idx4(m, 1, k, j, mdimcc, ido, l1)]
                    - cbuf[idx4(m, 1, k, jc, mdimcc, ido, l1)];
                hbuf[idx4(m, 1, k, jc, mdimch, ido, l1)] = cbuf[idx4(m, 1, k, j, mdimcc, ido, l1)]
                    + cbuf[idx4(m, 1, k, jc, mdimcc, ido, l1)];
            }
        }
    }

    if ido != 1 {
        if nbd >= l1 {
            for j in 2..=ipph {
                let jc = ipp2 - j;
                for k in 1..=l1 {
                    let mut i = 3;
                    while i <= ido {
                        for m in 1..=mp {
                            hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, k, j, mdimcc, ido, l1)]
                                - cbuf[idx4(m, i, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i - 1, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, k, j, mdimcc, ido, l1)]
                                + cbuf[idx4(m, i, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, k, j, mdimcc, ido, l1)]
                                + cbuf[idx4(m, i - 1, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, k, j, mdimcc, ido, l1)]
                                - cbuf[idx4(m, i - 1, k, jc, mdimcc, ido, l1)];
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
                            hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, k, j, mdimcc, ido, l1)]
                                - cbuf[idx4(m, i, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i - 1, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i - 1, k, j, mdimcc, ido, l1)]
                                + cbuf[idx4(m, i, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i, k, j, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, k, j, mdimcc, ido, l1)]
                                + cbuf[idx4(m, i - 1, k, jc, mdimcc, ido, l1)];
                            hbuf[idx4(m, i, k, jc, mdimch, ido, l1)] = cbuf
                                [idx4(m, i, k, j, mdimcc, ido, l1)]
                                - cbuf[idx4(m, i - 1, k, jc, mdimcc, ido, l1)];
                        }
                    }
                    i += 2;
                }
            }
        }

        for ik in 1..=idl1 {
            for m in 1..=mp {
                cbuf[idx3(m, ik, 1, mdimcc, idl1)] = hbuf[idx3(m, ik, 1, mdimch, idl1)];
            }
        }
        for j in 2..=ip {
            for k in 1..=l1 {
                for m in 1..=mp {
                    cbuf[idx4(m, 1, k, j, mdimcc, ido, l1)] =
                        hbuf[idx4(m, 1, k, j, mdimch, ido, l1)];
                }
            }
        }

        if nbd <= l1 {
            for j in 2..=ip {
                let mut idij = (j - 2) * ido;
                let mut i = 3;
                while i <= ido {
                    idij += 2;
                    for k in 1..=l1 {
                        for m in 1..=mp {
                            cbuf[idx4(m, i - 1, k, j, mdimcc, ido, l1)] = wa[idij - 2]
                                * hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)]
                                - wa[idij - 1] * hbuf[idx4(m, i, k, j, mdimch, ido, l1)];
                            cbuf[idx4(m, i, k, j, mdimcc, ido, l1)] = wa[idij - 2]
                                * hbuf[idx4(m, i, k, j, mdimch, ido, l1)]
                                + wa[idij - 1] * hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                        }
                    }
                    i += 2;
                }
            }
        } else {
            for j in 2..=ip {
                let is = (j - 2) * ido;
                for k in 1..=l1 {
                    let mut idij = is;
                    let mut i = 3;
                    while i <= ido {
                        idij += 2;
                        for m in 1..=mp {
                            cbuf[idx4(m, i - 1, k, j, mdimcc, ido, l1)] = wa[idij - 2]
                                * hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)]
                                - wa[idij - 1] * hbuf[idx4(m, i, k, j, mdimch, ido, l1)];
                            cbuf[idx4(m, i, k, j, mdimcc, ido, l1)] = wa[idij - 2]
                                * hbuf[idx4(m, i, k, j, mdimch, ido, l1)]
                                + wa[idij - 1] * hbuf[idx4(m, i - 1, k, j, mdimch, ido, l1)];
                        }
                        i += 2;
                    }
                }
            }
        }
    }
    cc.copy_from_slice(&cbuf);
}
