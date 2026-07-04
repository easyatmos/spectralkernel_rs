use numpy::PyArray1;
use pyo3::prelude::*;

/// Build the trigonometric workspace used by the real Fourier transform routines.
///
/// # Parameters
/// - `n`: Total spherical harmonic degree.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn hrffti_impl(n: i32) -> Vec<f32> {
    let n_usize = usize::try_from(n.max(0)).unwrap_or(0);
    let mut wsave = vec![0.0_f32; n_usize + 15];
    if n == 1 {
        return wsave;
    }
    let (wa, fac) = wsave.split_at_mut(n_usize);
    hrfti1(n_usize, wa, fac);
    wsave
}

#[pyfunction]
/// Python wrapper for `hrffti_impl` that exposes the FFT workspace as a NumPy array.
///
/// # Parameters
/// - `n`: Total spherical harmonic degree.
///
/// # Returns
/// A one-dimensional NumPy array containing the computed workspace.
pub fn hrffti<'py>(py: Python<'py>, n: i32) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let wsave = hrffti_impl(n);
    Ok(PyArray1::from_vec(py, wsave))
}

fn hrfti1(n: usize, wa: &mut [f32], fac: &mut [f32]) {
    let ntryh = [4_i32, 2_i32, 3_i32, 5_i32];
    let mut nl = n as i32;
    let mut nf = 0_i32;
    let mut j = 0_i32;
    let mut ntry = 0_i32;

    'outer: loop {
        j += 1;
        if j <= 4 {
            ntry = ntryh[(j - 1) as usize];
        } else {
            ntry += 2;
        }

        loop {
            let nq = nl / ntry;
            let nr = nl - ntry * nq;
            if nr != 0 {
                continue 'outer;
            }
            nf += 1;
            fac[(nf + 1) as usize] = ntry as f32;
            nl = nq;
            if ntry == 2 && nf != 1 {
                for i in 2..=nf {
                    let ib = nf - i + 2;
                    fac[(ib + 1) as usize] = fac[ib as usize];
                }
                fac[2] = 2.0_f32;
            }
            if nl == 1 {
                break 'outer;
            }
        }
    }

    fac[0] = n as f32;
    fac[1] = nf as f32;
    let tpi = 8.0_f64 * (1.0_f64).atan();
    let argh = tpi / n as f64;
    let mut is = 0_i32;
    let nfm1 = nf - 1;
    let mut l1 = 1_i32;
    if nfm1 == 0 {
        return;
    }

    for k1 in 1..=nfm1 {
        let ip = fac[(k1 + 1) as usize] as i32;
        let mut ld = 0_i32;
        let l2 = l1 * ip;
        let ido = n as i32 / l2;
        let ipm = ip - 1;
        for _j in 1..=ipm {
            ld += l1;
            let mut i = is;
            let argld = ld as f64 * argh;
            let mut fi = 0.0_f64;
            let mut ii = 3_i32;
            while ii <= ido {
                i += 2;
                fi += 1.0_f64;
                let arg = fi * argld;
                wa[(i - 2) as usize] = arg.cos() as f32;
                wa[(i - 1) as usize] = arg.sin() as f32;
                ii += 2;
            }
            is += ido;
        }
        l1 = l2;
    }
}
