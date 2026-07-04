use numpy::PyArray1;
use pyo3::prelude::*;

fn to_usize(value: i32) -> usize {
    usize::try_from(value).unwrap_or(0)
}

fn alfk(n: i32, m: i32) -> Vec<f32> {
    let n_usize = to_usize(n.max(0));
    let mut cp = vec![0.0_f32; (n_usize / 2) + 2];
    let sc10 = 1024.0_f32;
    let sc20 = sc10 * sc10;
    let sc40 = sc20 * sc20;

    cp[1] = 0.0_f32;
    let ma = m.abs();
    if ma > n {
        return cp;
    }

    if n == 0 {
        cp[1] = (2.0_f32).sqrt();
        return cp;
    }

    if n == 1 {
        if ma == 0 {
            cp[1] = (1.5_f32).sqrt();
        } else {
            cp[1] = (0.75_f32).sqrt();
            if m == -1 {
                cp[1] = -cp[1];
            }
        }
        return cp;
    }

    let (nmms2, mut fnum, mut fnmh, pm1) = if (n + ma) % 2 == 0 {
        (
            (n - ma) / 2,
            (n + ma + 1) as f32,
            (n - ma + 1) as f32,
            1.0_f32,
        )
    } else {
        (
            (n - ma - 1) / 2,
            (n + ma + 2) as f32,
            (n - ma + 2) as f32,
            -1.0_f32,
        )
    };

    let mut t1 = 1.0_f32 / sc20;
    let mut nex = 20_i32;
    let mut fden = 2.0_f32;
    if nmms2 >= 1 {
        for _ in 1..=nmms2 {
            t1 = fnum * t1 / fden;
            if t1 > sc20 {
                t1 /= sc40;
                nex += 40;
            }
            fnum += 2.0_f32;
            fden += 2.0_f32;
        }
    }

    t1 /= 2.0_f32.powi(n as i32 - 1 - nex);
    if (ma / 2) % 2 != 0 {
        t1 = -t1;
    }

    let mut t2 = 1.0_f32;
    if ma != 0 {
        for _ in 1..=to_usize(ma) {
            t2 = fnmh * t2 / (fnmh + pm1);
            fnmh += 2.0_f32;
        }
    }

    let cp2 = t1 * (((n as f32) + 0.5_f32) * t2).sqrt();
    let fnnp1 = (n * (n + 1)) as f32;
    let fnmsq = fnnp1 - 2.0_f32 * (ma * ma) as f32;
    let mut l = (n + 1) / 2;
    if n % 2 == 0 && ma % 2 == 0 {
        l += 1;
    }
    let l_usize = to_usize(l);
    cp[l_usize] = cp2;
    if m < 0 && ma % 2 != 0 {
        cp[l_usize] = -cp[l_usize];
    }

    if l <= 1 {
        return cp;
    }

    let mut fk = n as f32;
    let mut a1 = (fk - 2.0_f32) * (fk - 1.0_f32) - fnnp1;
    let mut b1 = 2.0_f32 * (fk * fk - fnmsq);
    cp[to_usize(l - 1)] = b1 * cp[l_usize] / a1;

    loop {
        l -= 1;
        if l <= 1 {
            return cp;
        }
        fk -= 2.0_f32;
        a1 = (fk - 2.0_f32) * (fk - 1.0_f32) - fnnp1;
        b1 = -2.0_f32 * (fk * fk - fnmsq);
        let c1 = (fk + 1.0_f32) * (fk + 2.0_f32) - fnnp1;
        let l0 = to_usize(l - 1);
        let l1 = to_usize(l);
        let l2 = to_usize(l + 1);
        cp[l0] = -(b1 * cp[l1] + c1 * cp[l2]) / a1;
    }
}

fn lfpt(n: i32, m: i32, theta: f32, cp: &[f32]) -> f32 {
    let ma = m.abs();
    if ma > n {
        return 0.0_f32;
    }

    if n == 0 && ma == 0 {
        return (0.5_f32).sqrt();
    }

    let nmod = n % 2;
    let mmod = ma % 2;

    if nmod == 0 {
        if mmod == 0 {
            let kdo = to_usize((n / 2) + 1);
            let cdt = (theta + theta).cos();
            let sdt = (theta + theta).sin();
            let mut ct = 1.0_f32;
            let mut st = 0.0_f32;
            let mut sum = 0.5_f32 * cp[1];
            for kp1 in 2..=kdo {
                let cth = cdt * ct - sdt * st;
                st = sdt * ct + cdt * st;
                ct = cth;
                sum += cp[kp1] * ct;
            }
            return sum;
        }

        let kdo = to_usize(n / 2);
        let cdt = (theta + theta).cos();
        let sdt = (theta + theta).sin();
        let mut ct = 1.0_f32;
        let mut st = 0.0_f32;
        let mut sum = 0.0_f32;
        for k in 1..=kdo {
            let cth = cdt * ct - sdt * st;
            st = sdt * ct + cdt * st;
            ct = cth;
            sum += cp[k] * st;
        }
        return sum;
    }

    let kdo = to_usize((n + 1) / 2);
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();
    let mut ct = theta.cos();
    let mut st = -theta.sin();
    let mut sum = 0.0_f32;

    if mmod == 0 {
        for k in 1..=kdo {
            let cth = cdt * ct - sdt * st;
            st = sdt * ct + cdt * st;
            ct = cth;
            sum += cp[k] * ct;
        }
    } else {
        for k in 1..=kdo {
            let cth = cdt * ct - sdt * st;
            st = sdt * ct + cdt * st;
            ct = cth;
            sum += cp[k] * st;
        }
    }

    sum
}

pub fn getlegfunc_impl(lat: f32, ntrunc: i32) -> Vec<f32> {
    let ntrunc = ntrunc.max(0);
    let ntrunc_usize = to_usize(ntrunc);
    let mut legfunc = vec![0.0_f32; ((ntrunc_usize + 1) * (ntrunc_usize + 2) / 2) + 1];
    let pi = 4.0_f32 * 1.0_f32.atan();
    let theta = 0.5_f32 * pi - (pi / 180.0_f32) * lat;
    let mut nmstrt = 0_i32;

    for m in 1..=ntrunc + 1 {
        for n in m..=ntrunc + 1 {
            let nm = nmstrt + n - m + 1;
            let cp = alfk(n - 1, m - 1);
            legfunc[to_usize(nm)] = lfpt(n - 1, m - 1, theta, &cp);
        }
        nmstrt += ntrunc - m + 2;
    }

    legfunc[1..].to_vec()
}

#[pyfunction]
pub fn getlegfunc<'py>(
    py: Python<'py>,
    lat: f32,
    ntrunc: i32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let legfunc = getlegfunc_impl(lat, ntrunc);
    let legfunc = PyArray1::from_vec(py, legfunc);
    Ok(legfunc.to_owned())
}
