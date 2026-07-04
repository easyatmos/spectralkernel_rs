pub fn dnlfk(m: i32, n: i32) -> Vec<f64> {
    let n_usize = usize::try_from(n.max(0)).unwrap_or(0);
    let mut cp = vec![0.0_f64; (n_usize / 2) + 2];
    let sc10 = 1024.0_f64;
    let sc20 = sc10 * sc10;
    let sc40 = sc20 * sc20;

    cp[1] = 0.0_f64;
    let ma = m.abs();
    if ma > n {
        return cp;
    }
    if n == 0 {
        cp[1] = (2.0_f64).sqrt();
        return cp;
    }
    if n == 1 {
        if ma != 0 {
            cp[1] = (0.75_f64).sqrt();
            if m == -1 {
                cp[1] = -cp[1];
            }
        } else {
            cp[1] = (1.5_f64).sqrt();
        }
        return cp;
    }

    let (nmms2, mut fnum, mut fnmh, pm1) = if (n + ma) % 2 == 0 {
        (
            (n - ma) / 2,
            (n + ma + 1) as f64,
            (n - ma + 1) as f64,
            1.0_f64,
        )
    } else {
        (
            (n - ma - 1) / 2,
            (n + ma + 2) as f64,
            (n - ma + 2) as f64,
            -1.0_f64,
        )
    };

    let mut t1 = 1.0_f64 / sc20;
    let mut nex = 20_i32;
    let mut fden = 2.0_f64;
    for _ in 1..=usize::try_from(nmms2.max(0)).unwrap_or(0) {
        t1 = fnum * t1 / fden;
        if t1 > sc20 {
            t1 /= sc40;
            nex += 40;
        }
        fnum += 2.0_f64;
        fden += 2.0_f64;
    }

    t1 /= 2.0_f64.powi(n - 1 - nex);
    if ((ma / 2) % 2) != 0 {
        t1 = -t1;
    }

    let mut t2 = 1.0_f64;
    for _ in 1..=usize::try_from(ma.max(0)).unwrap_or(0) {
        if ma == 0 {
            break;
        }
        t2 = fnmh * t2 / (fnmh + pm1);
        fnmh += 2.0_f64;
    }

    let cp2 = t1 * (((n as f64) + 0.5_f64) * t2).sqrt();
    let fnnp1 = (n * (n + 1)) as f64;
    let fnmsq = fnnp1 - 2.0_f64 * (ma * ma) as f64;
    let mut l = (n + 1) / 2;
    if n % 2 == 0 && ma % 2 == 0 {
        l += 1;
    }
    let mut lu = usize::try_from(l).unwrap_or(0);
    cp[lu] = cp2;
    if m < 0 && ma % 2 != 0 {
        cp[lu] = -cp[lu];
    }
    if l <= 1 {
        return cp;
    }

    let mut fk = n as f64;
    let mut a1 = (fk - 2.0_f64) * (fk - 1.0_f64) - fnnp1;
    let mut b1 = 2.0_f64 * (fk * fk - fnmsq);
    cp[usize::try_from(l - 1).unwrap_or(0)] = b1 * cp[lu] / a1;

    loop {
        l -= 1;
        if l <= 1 {
            return cp;
        }
        fk -= 2.0_f64;
        a1 = (fk - 2.0_f64) * (fk - 1.0_f64) - fnnp1;
        b1 = -2.0_f64 * (fk * fk - fnmsq);
        let c1 = (fk + 1.0_f64) * (fk + 2.0_f64) - fnnp1;
        lu = usize::try_from(l).unwrap_or(0);
        cp[usize::try_from(l - 1).unwrap_or(0)] =
            -(b1 * cp[lu] + c1 * cp[usize::try_from(l + 1).unwrap_or(0)]) / a1;
    }
}

pub fn dnlft(m: i32, n: i32, theta: f64, cp: &[f64]) -> f64 {
    let cdt = (theta + theta).cos();
    let sdt = (theta + theta).sin();
    let nmod = n % 2;
    let mmod = m % 2;

    if nmod == 0 {
        if mmod == 0 {
            let kdo = n / 2;
            let mut pb = 0.5_f64 * cp[1];
            if n == 0 {
                return pb;
            }
            let mut cth = cdt;
            let mut sth = sdt;
            for k in 1..=usize::try_from(kdo).unwrap_or(0) {
                pb += cp[k + 1] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
            return pb;
        }
        let kdo = n / 2;
        let mut pb = 0.0_f64;
        let mut cth = cdt;
        let mut sth = sdt;
        for k in 1..=usize::try_from(kdo).unwrap_or(0) {
            pb += cp[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
        return pb;
    }

    let kdo = (n + 1) / 2;
    let mut pb = 0.0_f64;
    let mut cth = theta.cos();
    let mut sth = theta.sin();
    if mmod == 0 {
        for k in 1..=usize::try_from(kdo).unwrap_or(0) {
            pb += cp[k] * cth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    } else {
        for k in 1..=usize::try_from(kdo).unwrap_or(0) {
            pb += cp[k] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    }
    pb
}

pub fn rabcp1(nlat: usize, nlon: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mmax = nlat.min(nlon / 2 + 1);
    let labc = ((mmax.saturating_sub(2)) * (nlat + nlat - mmax - 1)) / 2;
    let mut a = vec![0.0_f64; labc + 1];
    let mut b = vec![0.0_f64; labc + 1];
    let mut c = vec![0.0_f64; labc + 1];

    for mp1 in 3..=mmax {
        let m = mp1 - 1;
        let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
        let fm = m as f64;
        let tm = fm + fm;
        let mut temp = tm * (tm - 1.0_f64);
        a[ns] = ((tm + 1.0_f64) * (tm - 2.0_f64) / temp).sqrt();
        c[ns] = (2.0_f64 / temp).sqrt();
        if m == nlat - 1 {
            continue;
        }
        ns += 1;
        temp = tm * (tm + 1.0_f64);
        a[ns] = ((tm + 3.0_f64) * (tm - 2.0_f64) / temp).sqrt();
        c[ns] = (6.0_f64 / temp).sqrt();
        let mp3 = m + 3;
        if mp3 > nlat {
            continue;
        }
        for np1 in mp3..=nlat {
            let n = np1 - 1;
            ns += 1;
            let fnn = n as f64;
            let tn = fnn + fnn;
            let cn = (tn + 1.0_f64) / (tn - 3.0_f64);
            let fnpm = fnn + fm;
            let fnmm = fnn - fm;
            temp = fnpm * (fnpm - 1.0_f64);
            a[ns] = (cn * (fnpm - 3.0_f64) * (fnpm - 2.0_f64) / temp).sqrt();
            b[ns] = (cn * fnmm * (fnmm - 1.0_f64) / temp).sqrt();
            c[ns] = ((fnmm + 1.0_f64) * (fnmm + 2.0_f64) / temp).sqrt();
        }
    }

    (a, b, c)
}
