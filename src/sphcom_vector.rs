use crate::sphcom_scalar::dnlfk;

pub fn dvbk(m: i32, n: i32) -> Vec<f64> {
    let len = usize::try_from(n.max(0) / 2 + 2).unwrap_or(0);
    let mut cv = vec![0.0_f64; len];
    if n <= 0 {
        return cv;
    }

    let work = dnlfk(m, n);
    let fnn = n as f64;
    let srnp1 = (fnn * (fnn + 1.0_f64)).sqrt();
    let modn = n % 2;
    let modm = m % 2;

    if modn == 0 {
        let ncv = usize::try_from(n / 2).unwrap_or(0);
        let mut fk = 0.0_f64;
        if modm == 0 {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = -fk * work[l + 1] / srnp1;
            }
        } else {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = fk * work[l] / srnp1;
            }
        }
    } else {
        let ncv = usize::try_from((n + 1) / 2).unwrap_or(0);
        let mut fk = -1.0_f64;
        if modm == 0 {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = -fk * work[l] / srnp1;
            }
        } else {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = fk * work[l] / srnp1;
            }
        }
    }

    cv
}

pub fn dvtk(m: i32, n: i32) -> Vec<f64> {
    let len = usize::try_from(n.max(0) / 2 + 2).unwrap_or(0);
    let mut cv = vec![0.0_f64; len];
    if n <= 0 {
        return cv;
    }

    let work = dnlfk(m, n);
    let fnn = n as f64;
    let srnp1 = (fnn * (fnn + 1.0_f64)).sqrt();
    let modn = n % 2;
    let modm = m % 2;

    if modn == 0 {
        let ncv = usize::try_from(n / 2).unwrap_or(0);
        if ncv == 0 {
            return cv;
        }
        let mut fk = 0.0_f64;
        if modm == 0 {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = -fk * fk * work[l + 1] / srnp1;
            }
        } else {
            for l in 1..=ncv {
                fk += 2.0_f64;
                cv[l] = -fk * fk * work[l] / srnp1;
            }
        }
    } else {
        let ncv = usize::try_from((n + 1) / 2).unwrap_or(0);
        let mut fk = -1.0_f64;
        for l in 1..=ncv {
            fk += 2.0_f64;
            cv[l] = -fk * fk * work[l] / srnp1;
        }
    }

    cv
}

pub fn dwbk(m: i32, n: i32) -> Vec<f64> {
    let len = usize::try_from(n.max(0) / 2 + 2).unwrap_or(0);
    let mut cw = vec![0.0_f64; len];
    if n <= 0 || m <= 0 {
        return cw;
    }

    let work = dnlfk(m, n);
    let fnn = n as f64;
    let srnp1 = (fnn * (fnn + 1.0_f64)).sqrt();
    let cf = 2.0_f64 * (m as f64) / srnp1;
    let modn = n % 2;
    let modm = m % 2;

    if modn == 0 {
        let mut l = usize::try_from(n / 2).unwrap_or(0);
        if l == 0 {
            return cw;
        }
        if modm == 0 {
            cw[l] = -cf * work[l + 1];
            while l > 1 {
                cw[l - 1] = cw[l] - cf * work[l];
                l -= 1;
            }
        } else {
            cw[l] = cf * work[l];
            while l > 1 {
                cw[l - 1] = cw[l] + cf * work[l - 1];
                l -= 1;
            }
        }
    } else if modm == 0 {
        let mut l = usize::try_from((n - 1) / 2).unwrap_or(0);
        if l == 0 {
            return cw;
        }
        cw[l] = -cf * work[l + 1];
        while l > 1 {
            cw[l - 1] = cw[l] - cf * work[l];
            l -= 1;
        }
    } else {
        let mut l = usize::try_from((n + 1) / 2).unwrap_or(0);
        cw[l] = cf * work[l];
        while l > 1 {
            cw[l - 1] = cw[l] + cf * work[l - 1];
            l -= 1;
        }
    }
    cw
}

pub fn dwtk(m: i32, n: i32) -> Vec<f64> {
    let len = usize::try_from(n.max(0) / 2 + 2).unwrap_or(0);
    let mut cw = vec![0.0_f64; len];
    if n <= 0 || m <= 0 {
        return cw;
    }

    let work = dnlfk(m, n);
    let fnn = n as f64;
    let srnp1 = (fnn * (fnn + 1.0_f64)).sqrt();
    let cf = 2.0_f64 * (m as f64) / srnp1;
    let modn = n % 2;
    let modm = m % 2;

    if modn == 0 {
        let mut l = usize::try_from(n / 2).unwrap_or(0);
        if l == 0 {
            return cw;
        }
        if modm == 0 {
            cw[l] = -cf * work[l + 1];
            loop {
                l -= 1;
                if l == 0 {
                    break;
                }
                cw[l] = cw[l + 1] - cf * work[l + 1];
                cw[l + 1] *= (2 * l + 1) as f64;
            }
        } else {
            cw[l] = cf * work[l];
            loop {
                l -= 1;
                if l == 0 {
                    cw[1] = -cw[1];
                    break;
                }
                cw[l] = cw[l + 1] + cf * work[l];
                cw[l + 1] *= -((2 * l + 1) as f64);
            }
        }
    } else if modm == 0 {
        let mut l = usize::try_from((n - 1) / 2).unwrap_or(0);
        if l == 0 {
            return cw;
        }
        cw[l] = -cf * work[l + 1];
        loop {
            l -= 1;
            if l == 0 {
                cw[1] *= 2.0_f64;
                break;
            }
            cw[l] = cw[l + 1] - cf * work[l + 1];
            cw[l + 1] *= (2 * l + 2) as f64;
        }
    } else {
        let mut l = usize::try_from((n + 1) / 2).unwrap_or(0);
        cw[l] = cf * work[l];
        loop {
            l -= 1;
            if l == 0 {
                cw[1] = 0.0_f64;
                break;
            }
            cw[l] = cw[l + 1] + cf * work[l];
            cw[l + 1] *= -((2 * l) as f64);
        }
    }

    cw
}

pub fn dvbt(m: i32, n: i32, theta: f64, cv: &[f64]) -> f64 {
    if n == 0 {
        return 0.0_f64;
    }
    let mut vh = 0.0_f64;
    let mut cth = theta.cos();
    let mut sth = theta.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let mmod = m % 2;
    let nmod = n % 2;

    if nmod == 0 {
        cth = cdt;
        sth = sdt;
        let ncv = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncv {
                vh += cv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncv {
                vh += cv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        let ncv = usize::try_from((n + 1) / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncv {
                vh += cv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncv {
                vh += cv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }

    vh
}

pub fn dvtt(m: i32, n: i32, theta: f64, cv: &[f64]) -> f64 {
    if n == 0 {
        return 0.0_f64;
    }
    let mut vh = 0.0_f64;
    let mut cth = theta.cos();
    let mut sth = theta.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let mmod = m % 2;
    let nmod = n % 2;

    if nmod == 0 {
        cth = cdt;
        sth = sdt;
        let ncv = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncv {
                vh += cv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncv {
                vh += cv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        let ncv = usize::try_from((n + 1) / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncv {
                vh += cv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncv {
                vh += cv[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }

    vh
}

pub fn dwbt(m: i32, n: i32, theta: f64, cw: &[f64]) -> f64 {
    if n <= 0 || m <= 0 {
        return 0.0_f64;
    }
    let mut wh = 0.0_f64;
    let mut cth = theta.cos();
    let mut sth = theta.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let mmod = m % 2;
    let nmod = n % 2;

    if nmod == 0 {
        let ncw = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncw {
                wh += cw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncw {
                wh += cw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        cth = cdt;
        sth = sdt;
        if mmod == 0 {
            let ncw = usize::try_from((n - 1) / 2).unwrap_or(0);
            for k in 1..=ncw {
                wh += cw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            let ncw = usize::try_from((n + 1) / 2).unwrap_or(0);
            wh = 0.5_f64 * cw[1];
            for k in 2..=ncw {
                wh += cw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }
    wh
}

pub fn dwtt(m: i32, n: i32, theta: f64, cw: &[f64]) -> f64 {
    if n <= 0 || m <= 0 {
        return 0.0_f64;
    }
    let mut wh = 0.0_f64;
    let mut cth = theta.cos();
    let mut sth = theta.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let mmod = m % 2;
    let nmod = n % 2;

    if nmod == 0 {
        let ncw = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for k in 1..=ncw {
                wh += cw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=ncw {
                wh += cw[k] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        cth = cdt;
        sth = sdt;
        if mmod == 0 {
            let ncw = usize::try_from((n - 1) / 2).unwrap_or(0);
            for k in 1..=ncw {
                wh += cw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            let ncw = usize::try_from((n + 1) / 2).unwrap_or(0);
            if ncw >= 2 {
                for k in 2..=ncw {
                    wh += cw[k] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        }
    }
    wh
}

pub fn dzvk(nlat: i32, m: i32, n: i32) -> Vec<f64> {
    let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
    let mut czv = vec![0.0_f64; lc + 1];
    if n <= 0 {
        return czv;
    }
    let work = dvbk(m, n);
    let sc1 = 2.0_f64 / ((nlat - 1) as f64);
    let nmod = n % 2;
    let mmod = m % 2;

    if nmod == 0 {
        let kdo = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for id in 1..=lc {
                let i = (id + id - 2) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - i).powi(2);
                    sum += work[k] * (t1 - t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        } else {
            for id in 1..=lc {
                let i = (id + id - 2) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - i).powi(2);
                    sum += work[k] * (t1 + t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        }
    } else {
        let kdo = usize::try_from((n + 1) / 2).unwrap_or(0);
        if mmod == 0 {
            for id in 1..=lc {
                let i = (2_i32 * id as i32 - 3_i32) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                    sum += work[k] * (t1 - t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        } else {
            for id in 1..=lc {
                let i = (id + id - 1) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                    sum += work[k] * (t1 + t2) / (t1 * t2);
                }
                czv[id] = sc1 * sum;
            }
        }
    }

    czv
}

pub fn dzvt(nlat: i32, m: i32, n: i32, th: f64, czv: &[f64]) -> f64 {
    if n <= 0 {
        return 0.0_f64;
    }
    let mut zvh = 0.0_f64;
    let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
    let lq = lc.saturating_sub(1);
    let ls = lc.saturating_sub(2);
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let lmod = nlat % 2;
    let mmod = m % 2;
    let nmod = n % 2;

    if lmod != 0 {
        if nmod == 0 {
            cth = cdt;
            sth = sdt;
            if mmod == 0 {
                for k in 1..=ls {
                    zvh += czv[k + 1] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                zvh = 0.5_f64 * czv[1];
                for k in 2..=lq {
                    zvh += czv[k] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
                zvh += 0.5_f64 * czv[lc] * (((nlat - 1) as f64) * th).cos();
            }
        } else if mmod == 0 {
            for k in 1..=lq {
                zvh += czv[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            for k in 1..=lq {
                zvh += czv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if nmod == 0 {
        cth = cdt;
        sth = sdt;
        if mmod == 0 {
            for k in 1..=lq {
                zvh += czv[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            zvh = 0.5_f64 * czv[1];
            for k in 2..=lc {
                zvh += czv[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else if mmod == 0 {
        for k in 1..=lq {
            zvh += czv[k + 1] * sth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    } else {
        zvh = 0.5_f64 * czv[lc] * (((nlat - 1) as f64) * th).cos();
        for k in 1..=lq {
            zvh += czv[k] * cth;
            let chh = cdt * cth - sdt * sth;
            sth = sdt * cth + cdt * sth;
            cth = chh;
        }
    }

    zvh
}

pub fn dzwk(nlat: i32, m: i32, n: i32) -> Vec<f64> {
    let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
    let mut czw = vec![0.0_f64; lc + 1];
    if n <= 0 {
        return czw;
    }
    let work = dwbk(m, n);
    let sc1 = 2.0_f64 / ((nlat - 1) as f64);
    let nmod = n % 2;
    let mmod = m % 2;

    if nmod == 0 {
        let kdo = usize::try_from(n / 2).unwrap_or(0);
        if mmod == 0 {
            for id in 1..=lc {
                let i = (2_i32 * id as i32 - 3_i32) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                    sum += work[k] * (t1 - t2) / (t1 * t2);
                }
                czw[id] = sc1 * sum;
            }
        } else {
            for id in 1..=lc {
                let i = (id + id - 1) as f64;
                let mut sum = 0.0_f64;
                for k in 1..=kdo {
                    let kf = k as f64;
                    let t1 = 1.0_f64 - (kf + kf - 1.0_f64 + i).powi(2);
                    let t2 = 1.0_f64 - (kf + kf - 1.0_f64 - i).powi(2);
                    sum += work[k] * (t1 + t2) / (t1 * t2);
                }
                czw[id] = sc1 * sum;
            }
        }
    } else if mmod == 0 {
        let kdo = usize::try_from((n - 1) / 2).unwrap_or(0);
        for id in 1..=lc {
            let i = (id + id - 2) as f64;
            let mut sum = 0.0_f64;
            for k in 1..=kdo {
                let kf = k as f64;
                let t1 = 1.0_f64 - (kf + kf + i).powi(2);
                let t2 = 1.0_f64 - (kf + kf - i).powi(2);
                sum += work[k] * (t1 - t2) / (t1 * t2);
            }
            czw[id] = sc1 * sum;
        }
    } else {
        let kdo = usize::try_from((n + 1) / 2).unwrap_or(0);
        for id in 1..=lc {
            let i = (id + id - 2) as f64;
            let mut sum = work[1] / (1.0_f64 - i * i);
            if kdo >= 2 {
                for kp1 in 2..=kdo {
                    let k = (kp1 - 1) as f64;
                    let t1 = 1.0_f64 - (k + k + i).powi(2);
                    let t2 = 1.0_f64 - (k + k - i).powi(2);
                    sum += work[kp1] * (t1 + t2) / (t1 * t2);
                }
            }
            czw[id] = sc1 * sum;
        }
    }

    czw
}

pub fn dzwt(nlat: i32, m: i32, n: i32, th: f64, czw: &[f64]) -> f64 {
    if n <= 0 {
        return 0.0_f64;
    }
    let mut zwh = 0.0_f64;
    let lc = usize::try_from((nlat + 1) / 2).unwrap_or(0);
    let lq = lc.saturating_sub(1);
    let ls = lc.saturating_sub(2);
    let mut cth = th.cos();
    let mut sth = th.sin();
    let cdt = cth * cth - sth * sth;
    let sdt = 2.0_f64 * sth * cth;
    let lmod = nlat % 2;
    let mmod = m % 2;
    let nmod = n % 2;

    if lmod != 0 {
        if nmod == 0 {
            if mmod == 0 {
                for k in 1..=lq {
                    zwh += czw[k + 1] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                for k in 1..=lq {
                    zwh += czw[k] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            }
        } else {
            cth = cdt;
            sth = sdt;
            if mmod == 0 {
                for k in 1..=ls {
                    zwh += czw[k + 1] * sth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
            } else {
                zwh = 0.5_f64 * czw[1];
                for k in 2..=lq {
                    zwh += czw[k] * cth;
                    let chh = cdt * cth - sdt * sth;
                    sth = sdt * cth + cdt * sth;
                    cth = chh;
                }
                zwh += 0.5_f64 * czw[lc] * (((nlat - 1) as f64) * th).cos();
            }
        }
    } else if nmod == 0 {
        if mmod == 0 {
            for k in 1..=lq {
                zwh += czw[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            zwh = 0.5_f64 * czw[lc] * (((nlat - 1) as f64) * th).cos();
            for k in 1..=lq {
                zwh += czw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    } else {
        cth = cdt;
        sth = sdt;
        if mmod == 0 {
            for k in 1..=lq {
                zwh += czw[k + 1] * sth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        } else {
            zwh = 0.5_f64 * czw[1];
            for k in 2..=lc {
                zwh += czw[k] * cth;
                let chh = cdt * cth - sdt * sth;
                sth = sdt * cth + cdt * sth;
                cth = chh;
            }
        }
    }

    zwh
}

pub fn rabcv1(nlat: usize, nlon: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let mut a = vec![0.0_f64; labc + 1];
    let mut b = vec![0.0_f64; labc + 1];
    let mut c = vec![0.0_f64; labc + 1];

    if mmax < 3 {
        return (a, b, c);
    }

    for mp1 in 3..=mmax {
        let m = mp1 - 1;
        let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
        let fm = m as f64;
        let tm = fm + fm;
        let mut temp = tm * (tm - 1.0_f64);
        let mut tpn = (fm - 2.0_f64) * (fm - 1.0_f64) / (fm * (fm + 1.0_f64));
        a[ns] = (tpn * (tm + 1.0_f64) * (tm - 2.0_f64) / temp).sqrt();
        c[ns] = (2.0_f64 / temp).sqrt();
        if m == nlat - 1 {
            continue;
        }
        ns += 1;
        temp = tm * (tm + 1.0_f64);
        tpn = (fm - 1.0_f64) * fm / ((fm + 1.0_f64) * (fm + 2.0_f64));
        a[ns] = (tpn * (tm + 3.0_f64) * (tm - 2.0_f64) / temp).sqrt();
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
            tpn = (fnn - 2.0_f64) * (fnn - 1.0_f64) / (fnn * (fnn + 1.0_f64));
            let fnpm = fnn + fm;
            let fnmm = fnn - fm;
            temp = fnpm * (fnpm - 1.0_f64);
            a[ns] = (tpn * cn * (fnpm - 3.0_f64) * (fnpm - 2.0_f64) / temp).sqrt();
            b[ns] = (tpn * cn * fnmm * (fnmm - 1.0_f64) / temp).sqrt();
            c[ns] = (((fnmm + 1.0_f64) * (fnmm + 2.0_f64)) / temp).sqrt();
        }
    }

    (a, b, c)
}

pub fn rabcw1(nlat: usize, nlon: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let mut a = vec![0.0_f64; labc + 1];
    let mut b = vec![0.0_f64; labc + 1];
    let mut c = vec![0.0_f64; labc + 1];

    if mmax < 4 {
        return (a, b, c);
    }

    for mp1 in 4..=mmax {
        let m = mp1 - 1;
        let mut ns = ((m - 2) * (nlat + nlat - m - 1)) / 2 + 1;
        let fm = m as f64;
        let tm = fm + fm;
        let mut temp = tm * (tm - 1.0_f64);
        let mut tpn = (fm - 2.0_f64) * (fm - 1.0_f64) / (fm * (fm + 1.0_f64));
        let mut tph = fm / (fm - 2.0_f64);
        a[ns] = tph * (tpn * (tm + 1.0_f64) * (tm - 2.0_f64) / temp).sqrt();
        c[ns] = tph * (2.0_f64 / temp).sqrt();
        if m == nlat - 1 {
            continue;
        }
        ns += 1;
        temp = tm * (tm + 1.0_f64);
        tpn = (fm - 1.0_f64) * fm / ((fm + 1.0_f64) * (fm + 2.0_f64));
        tph = fm / (fm - 2.0_f64);
        a[ns] = tph * (tpn * (tm + 3.0_f64) * (tm - 2.0_f64) / temp).sqrt();
        c[ns] = tph * (6.0_f64 / temp).sqrt();
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
            tpn = (fnn - 2.0_f64) * (fnn - 1.0_f64) / (fnn * (fnn + 1.0_f64));
            tph = fm / (fm - 2.0_f64);
            a[ns] = tph * (tpn * cn * (fnpm - 3.0_f64) * (fnpm - 2.0_f64) / temp).sqrt();
            b[ns] = (tpn * cn * fnmm * (fnmm - 1.0_f64) / temp).sqrt();
            c[ns] = tph * (((fnmm + 1.0_f64) * (fnmm + 2.0_f64)) / temp).sqrt();
        }
    }

    (a, b, c)
}

fn pack_init_tables(
    base0: &[f64],
    base1: &[f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    lim: usize,
) -> Vec<f64> {
    let labc = a.len().saturating_sub(1);
    let mut out = vec![0.0_f64; 2 * lim + 3 * labc];
    out[..lim].copy_from_slice(base0);
    out[lim..2 * lim].copy_from_slice(base1);
    for i in 0..labc {
        out[2 * lim + i] = a[i + 1];
        out[2 * lim + labc + i] = b[i + 1];
        out[2 * lim + 2 * labc + i] = c[i + 1];
    }
    out
}

pub fn zvinit_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);
    let mdo = 2.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f64; lim];
    let mut base1 = vec![0.0_f64; lim];

    for mp1 in 1..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let czv = dzvk(nlat as i32, m, n);
            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let zvh = dzvt(nlat as i32, m, n, th, &czv);
                let idx = (np1 - 1) * imid + (i - 1);
                if mp1 == 1 {
                    base0[idx] = zvh;
                } else {
                    base1[idx] = zvh;
                }
            }
            let idx = (np1 - 1) * imid;
            if mp1 == 1 {
                base0[idx] *= 0.5_f64;
            } else {
                base1[idx] *= 0.5_f64;
            }
        }
    }

    let (a, b, c) = rabcv1(nlat, nlon);
    pack_init_tables(&base0, &base1, &a, &b, &c, lim)
}

pub fn zwinit_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);
    let mdo = 3.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f64; lim];
    let mut base1 = vec![0.0_f64; lim];
    if mdo < 2 {
        let (a, b, c) = rabcw1(nlat, nlon);
        return pack_init_tables(&base0, &base1, &a, &b, &c, lim);
    }

    for mp1 in 2..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let czw = dzwk(nlat as i32, m, n);

            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let zwh = dzwt(nlat as i32, m, n, th, &czw);
                let idx = (np1 - 1) * imid + (i - 1);
                if m == 1 {
                    base0[idx] = zwh;
                } else {
                    base1[idx] = zwh;
                }
            }
            let idx = (np1 - 1) * imid;
            if m == 1 {
                base0[idx] *= 0.5_f64;
            } else {
                base1[idx] *= 0.5_f64;
            }
        }
    }

    let (a, b, c) = rabcw1(nlat, nlon);
    pack_init_tables(&base0, &base1, &a, &b, &c, lim)
}

pub fn vbinit_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);
    let mdo = 2.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f64; lim];
    let mut base1 = vec![0.0_f64; lim];

    for mp1 in 1..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cv = dvbk(m, n);
            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let vbh = dvbt(m, n, th, &cv);
                let idx = (np1 - 1) * imid + (i - 1);
                if mp1 == 1 {
                    base0[idx] = vbh;
                } else {
                    base1[idx] = vbh;
                }
            }
        }
    }

    let (a, b, c) = rabcv1(nlat, nlon);
    pack_init_tables(&base0, &base1, &a, &b, &c, lim)
}

pub fn wbinit_impl(nlat: usize, nlon: usize) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = imid * nlat;
    let pi = 4.0_f64 * (1.0_f64).atan();
    let dt = pi / ((nlat - 1) as f64);
    let mdo = 3.min(nlat).min((nlon + 1) / 2);
    let mut base0 = vec![0.0_f64; lim];
    let mut base1 = vec![0.0_f64; lim];
    if mdo < 2 {
        let (a, b, c) = rabcw1(nlat, nlon);
        return pack_init_tables(&base0, &base1, &a, &b, &c, lim);
    }

    for mp1 in 2..=mdo {
        let m = (mp1 - 1) as i32;
        for np1 in mp1..=nlat {
            let n = (np1 - 1) as i32;
            let cw = dwbk(m, n);
            for i in 1..=imid {
                let th = ((i - 1) as f64) * dt;
                let wbh = dwbt(m, n, th, &cw);
                let idx = (np1 - 1) * imid + (i - 1);
                if m == 1 {
                    base0[idx] = wbh;
                } else {
                    base1[idx] = wbh;
                }
            }
        }
    }

    let (a, b, c) = rabcw1(nlat, nlon);
    pack_init_tables(&base0, &base1, &a, &b, &c, lim)
}

pub fn zvin_column(nlat: usize, nlon: usize, ityp: i32, m: usize, wzvin: &[f64]) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = nlat * imid;
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let zvz = &wzvin[0..lim];
    let zv1 = &wzvin[lim..2 * lim];
    let a = &wzvin[2 * lim..2 * lim + labc];
    let b = &wzvin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &wzvin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    let mut zv = vec![0.0_f64; imid * nlat * 3];
    let (mut i1, mut i2, mut i3) = (0usize, 1usize, 2usize);

    for mm in 0..=m {
        if mm == 0 {
            i1 = 0;
            i2 = 1;
            i3 = 2;
            for np1 in 1..=nlat {
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                    let src = (np1 - 1) * imid + (i - 1);
                    zv[dst] = zvz[src];
                }
            }
        } else {
            let ihold = i1;
            i1 = i2;
            i2 = i3;
            i3 = ihold;
            if mm == 1 {
                for np1 in 2..=nlat {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                        let src = (np1 - 1) * imid + (i - 1);
                        zv[dst] = zv1[src];
                    }
                }
            } else {
                let mut ns = ((mm - 2) * (nlat + nlat - mm - 1)) / 2;
                if ityp != 1 {
                    for i in 1..=imid {
                        let dst = (((i - 1) * nlat) + mm) * 3 + i3;
                        let src_mm2 = (((i - 1) * nlat) + (mm - 2)) * 3 + i1;
                        let src_mm = (((i - 1) * nlat) + mm) * 3 + i1;
                        zv[dst] = a[ns] * zv[src_mm2] - c[ns] * zv[src_mm];
                    }
                }
                if mm != nlat - 1 {
                    if ityp != 2 {
                        ns += 1;
                        for i in 1..=imid {
                            let dst = (((i - 1) * nlat) + (mm + 1)) * 3 + i3;
                            let src_mm1 = (((i - 1) * nlat) + (mm - 1)) * 3 + i1;
                            let src_mp1 = (((i - 1) * nlat) + (mm + 1)) * 3 + i1;
                            zv[dst] = a[ns] * zv[src_mm1] - c[ns] * zv[src_mp1];
                        }
                    }
                    let mut nstrt = mm + 3;
                    if ityp == 1 {
                        nstrt = mm + 4;
                    }
                    if nstrt <= nlat {
                        let nstp = if ityp == 0 { 1 } else { 2 };
                        let mut np1 = nstrt;
                        while np1 <= nlat {
                            ns += nstp;
                            for i in 1..=imid {
                                let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                                let src_nm2_i1 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i1;
                                let src_nm2_i3 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i3;
                                let src_n_i1 = (((i - 1) * nlat) + (np1 - 1)) * 3 + i1;
                                zv[dst] = a[ns] * zv[src_nm2_i1] + b[ns] * zv[src_nm2_i3]
                                    - c[ns] * zv[src_n_i1];
                            }
                            np1 += nstp;
                        }
                    }
                }
            }
        }
    }

    let mut out = vec![0.0_f64; imid * nlat];
    for np1 in 1..=nlat {
        for i in 1..=imid {
            let src = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
            let dst = (np1 - 1) * imid + (i - 1);
            out[dst] = zv[src];
        }
    }
    out
}

pub fn zwin_column(nlat: usize, nlon: usize, ityp: i32, m: usize, wzwin: &[f64]) -> Vec<f64> {
    let imid = (nlat + 1) / 2;
    let lim = nlat * imid;
    let mmax = nlat.min((nlon + 1) / 2);
    let labc = (mmax.saturating_sub(2) * (nlat + nlat - mmax - 1)) / 2;
    let zw1 = &wzwin[0..lim];
    let zw2 = &wzwin[lim..2 * lim];
    let a = &wzwin[2 * lim..2 * lim + labc];
    let b = &wzwin[2 * lim + labc..2 * lim + 2 * labc];
    let c = &wzwin[2 * lim + 2 * labc..2 * lim + 3 * labc];

    if m < 2 {
        let mut out = vec![0.0_f64; imid * nlat];
        for np1 in 2..=nlat {
            for i in 1..=imid {
                out[(np1 - 1) * imid + (i - 1)] = zw1[(np1 - 1) * imid + (i - 1)];
            }
        }
        return out;
    }

    if m == 2 {
        let mut out = vec![0.0_f64; imid * nlat];
        for np1 in 3..=nlat {
            for i in 1..=imid {
                out[(np1 - 1) * imid + (i - 1)] = zw2[(np1 - 1) * imid + (i - 1)];
            }
        }
        return out;
    }

    let mut zw = vec![0.0_f64; imid * nlat * 3];
    let (mut i1, mut i2, mut i3);
    i1 = 0;
    i2 = 1;
    i3 = 2;
    for np1 in 2..=nlat {
        for i in 1..=imid {
            let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i2;
            zw[dst] = zw1[(np1 - 1) * imid + (i - 1)];
        }
    }
    for np1 in 3..=nlat {
        for i in 1..=imid {
            let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
            zw[dst] = zw2[(np1 - 1) * imid + (i - 1)];
        }
    }

    for mm in 3..=m {
        let ihold = i1;
        i1 = i2;
        i2 = i3;
        i3 = ihold;
        let mut ns = ((mm - 2) * (nlat + nlat - mm - 1)) / 2;
        if ityp != 1 {
            for i in 1..=imid {
                let dst = (((i - 1) * nlat) + mm) * 3 + i3;
                let src_mm2 = (((i - 1) * nlat) + (mm - 2)) * 3 + i1;
                let src_mm = (((i - 1) * nlat) + mm) * 3 + i1;
                zw[dst] = a[ns] * zw[src_mm2] - c[ns] * zw[src_mm];
            }
        }
        if mm == nlat - 1 {
            continue;
        }
        if ityp != 2 {
            ns += 1;
            for i in 1..=imid {
                let dst = (((i - 1) * nlat) + (mm + 1)) * 3 + i3;
                let src_mm1 = (((i - 1) * nlat) + (mm - 1)) * 3 + i1;
                let src_mp1 = (((i - 1) * nlat) + (mm + 1)) * 3 + i1;
                zw[dst] = a[ns] * zw[src_mm1] - c[ns] * zw[src_mp1];
            }
        }
        let mut nstrt = mm + 3;
        if ityp == 1 {
            nstrt = mm + 4;
        }
        if nstrt <= nlat {
            let nstp = if ityp == 0 { 1 } else { 2 };
            let mut np1 = nstrt;
            while np1 <= nlat {
                ns += nstp;
                for i in 1..=imid {
                    let dst = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
                    let src_nm2_i1 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i1;
                    let src_nm2_i3 = (((i - 1) * nlat) + (np1 - 3)) * 3 + i3;
                    let src_n_i1 = (((i - 1) * nlat) + (np1 - 1)) * 3 + i1;
                    zw[dst] =
                        a[ns] * zw[src_nm2_i1] + b[ns] * zw[src_nm2_i3] - c[ns] * zw[src_n_i1];
                }
                np1 += nstp;
            }
        }
    }

    let mut out = vec![0.0_f64; imid * nlat];
    for np1 in 1..=nlat {
        for i in 1..=imid {
            let src = (((i - 1) * nlat) + (np1 - 1)) * 3 + i3;
            let dst = (np1 - 1) * imid + (i - 1);
            out[dst] = zw[src];
            if m <= 3 && np1 <= 5 {}
        }
    }

    out
}
