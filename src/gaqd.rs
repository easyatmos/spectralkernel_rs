use numpy::PyArray1;
use pyo3::prelude::*;

fn dzeps(x: f64) -> f64 {
    let a = 4.0_f64 / 3.0_f64;
    loop {
        let b = a - 1.0_f64;
        let c = b + b + b;
        let eps = (c - 1.0_f64).abs();
        if eps != 0.0_f64 {
            return eps * x.abs();
        }
    }
}

fn cpdp(n: usize) -> (f64, Vec<f64>, Vec<f64>) {
    let ncp = (n + 1) / 2;
    let mut cp = vec![0.0_f64; ncp + 1];
    let mut dcp = vec![0.0_f64; ncp + 1];
    let mut t1 = -1.0_f64;
    let mut t2 = n as f64 + 1.0_f64;
    let mut t3 = 0.0_f64;
    let mut t4 = (n + n) as f64 + 1.0_f64;
    let mut cz = 0.0_f64;

    if n % 2 == 0 {
        cp[ncp] = 1.0_f64;
        for j in (2..=ncp).rev() {
            t1 += 2.0_f64;
            t2 -= 1.0_f64;
            t3 += 1.0_f64;
            t4 -= 2.0_f64;
            cp[j - 1] = (t1 * t2) / (t3 * t4) * cp[j];
        }
        t1 += 2.0_f64;
        t2 -= 1.0_f64;
        t3 += 1.0_f64;
        t4 -= 2.0_f64;
        cz = (t1 * t2) / (t3 * t4) * cp[1];
        for j in 1..=ncp {
            dcp[j] = (j + j) as f64 * cp[j];
        }
    } else {
        cp[ncp] = 1.0_f64;
        for j in (1..=ncp.saturating_sub(1)).rev() {
            t1 += 2.0_f64;
            t2 -= 1.0_f64;
            t3 += 1.0_f64;
            t4 -= 2.0_f64;
            cp[j] = (t1 * t2) / (t3 * t4) * cp[j + 1];
        }
        for j in 1..=ncp {
            dcp[j] = (j + j - 1) as f64 * cp[j];
        }
    }

    (cz, cp, dcp)
}

fn tpdp(n: usize, theta: f64, cz: f64, cp: &[f64], dcp: &[f64]) -> (f64, f64) {
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();

    if n % 2 == 0 {
        let kdo = n / 2;
        let mut pb = 0.5_f64 * cz;
        let mut dpb = 0.0_f64;
        if n > 0 {
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 1..=kdo {
                pb += cp[k] * cth;
                dpb -= dcp[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
        (pb, dpb)
    } else {
        let kdo = (n + 1) / 2;
        let mut pb = 0.0_f64;
        let mut dpb = 0.0_f64;
        let mut cth = theta.cos();
        let mut sth = theta.sin();
        for k in 1..=kdo {
            pb += cp[k] * cth;
            dpb -= dcp[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
        (pb, dpb)
    }
}

pub fn gaqd_impl(nlat: i32) -> (Vec<f64>, Vec<f64>, i32) {
    let nlat_usize = match usize::try_from(nlat) {
        Ok(value) if value > 0 => value,
        _ => return (Vec::new(), Vec::new(), 1),
    };

    let mut theta = vec![0.0_f64; nlat_usize + 1];
    let mut wts = vec![0.0_f64; nlat_usize + 1];

    if nlat_usize == 1 {
        theta[1] = 0.0_f64.acos();
        wts[1] = 2.0_f64;
        return (theta[1..].to_vec(), wts[1..].to_vec(), 0);
    }

    if nlat_usize == 2 {
        let x = (1.0_f64 / 3.0_f64).sqrt();
        theta[1] = x.acos();
        theta[2] = (-x).acos();
        wts[1] = 1.0_f64;
        wts[2] = 1.0_f64;
        return (theta[1..].to_vec(), wts[1..].to_vec(), 0);
    }

    let mut eps = dzeps(1.0_f64).sqrt();
    eps *= eps.sqrt();
    let pis2 = 2.0_f64 * 1.0_f64.atan();
    let pi = pis2 + pis2;
    let mnlat = nlat_usize % 2;
    let ns2 = nlat_usize / 2;
    let nhalf = (nlat_usize + 1) / 2;

    let (cz, cp, dcp) = cpdp(nlat_usize);

    let dtheta = pis2 / nhalf as f64;
    let dthalf = dtheta / 2.0_f64;
    let cmax = 0.2_f64 * dtheta;

    let mut zero;
    let mut zprev = pis2;
    let mut nix;
    if mnlat != 0 {
        zero = pis2 - dtheta;
        nix = nhalf - 1;
    } else {
        zero = pis2 - dthalf;
        nix = nhalf;
    }

    loop {
        let mut zlast;
        loop {
            zlast = zero;
            let (pb, dpb) = tpdp(nlat_usize, zero, cz, &cp, &dcp);
            let mut dcor = pb / dpb;
            let mut sgnd = 1.0_f64;
            if dcor != 0.0_f64 {
                sgnd = dcor / dcor.abs();
            }
            dcor = sgnd * dcor.abs().min(cmax);
            zero -= dcor;
            if (zero - zlast).abs() <= eps * zero.abs() {
                theta[nix] = zero;
                wts[nix] = ((nlat_usize + nlat_usize + 1) as f64)
                    / (dpb + pb * zlast.cos() / zlast.sin()).powi(2);
                break;
            }
        }

        let zhold = zero;
        if nix == 1 {
            break;
        }
        nix -= 1;
        if nix == nhalf - 1 {
            zero = 3.0_f64 * zero - pi;
        } else if nix < nhalf - 1 {
            zero = zero + zero - zprev;
        }
        zprev = zhold;
    }

    if mnlat != 0 {
        theta[nhalf] = pis2;
        let (_pb, dpb) = tpdp(nlat_usize, pis2, cz, &cp, &dcp);
        wts[nhalf] = ((nlat_usize + nlat_usize + 1) as f64) / (dpb * dpb);
    }

    for i in 1..=ns2 {
        wts[nlat_usize - i + 1] = wts[i];
        theta[nlat_usize - i + 1] = pi - theta[i];
    }

    let sum: f64 = wts[1..=nlat_usize].iter().sum();
    for value in &mut wts[1..=nlat_usize] {
        *value = 2.0_f64 * *value / sum;
    }

    (theta[1..].to_vec(), wts[1..].to_vec(), 0)
}

#[pyfunction]
pub fn gaqd<'py>(
    py: Python<'py>,
    nlat: i32,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, i32)> {
    let (theta, wts, ierror) = gaqd_impl(nlat);
    let theta = PyArray1::from_vec(py, theta);
    let wts = PyArray1::from_vec(py, wts);
    Ok((theta.to_owned(), wts.to_owned(), ierror))
}
