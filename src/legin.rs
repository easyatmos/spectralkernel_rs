/// Compute one of the stored Legendre initialization tables selected by the requested mode.
///
/// # Parameters
/// - `mode`: Selector controlling which stored Legendre recurrence table is generated.
/// - `l`: Leading degree or table width used by the selected recurrence branch.
/// - `nlat`: Number of latitudes in the grid.
/// - `m`: Zonal wavenumber or refinement level, depending on the routine.
/// - `w`: Input workspace or secondary component, depending on the routine.
/// - `pmn`: Output buffer that receives the generated Legendre values.
/// - `km_state`: Mutable state tuple updated with the recurrence bookkeeping indices.
///
/// # Returns
/// The computed index, table length, or updated state position returned by the routine.
pub fn legin_compute(
    mode: usize,
    l: usize,
    nlat: usize,
    m: usize,
    w: &[f32],
    pmn: &mut [f32],
    km_state: &mut (usize, usize, usize),
) -> usize {
    let late = (nlat + (nlat % 2)) / 2;
    let i1 = nlat;
    let i2 = i1 + nlat * late;
    let i3 = i2 + nlat * late;
    let i4 = i3 + (2 * nlat - l) * (l - 1) / 2;
    let i5 = i4 + (2 * nlat - l) * (l - 1) / 2;
    let p0n = &w[i1..i2];
    let p1n = &w[i2..i3];
    let abel = &w[i3..i4];
    let bbel = &w[i4..i5];
    let cbel = &w[i5..i5 + (2 * nlat - l) * (l - 1) / 2];

    let indx = |m: usize, n: usize| (n - 1) * (n - 2) / 2 + (m - 2);
    let imndx = |m: usize, n: usize| (l - 1) * (l - 2) / 2 + (n - l) * (l - 1) + (m - 2);

    let (mut km0, mut km1, mut km2) = *km_state;
    let mut ms = m + 1;
    let mut ninc = 1;
    if mode == 1 {
        ms = m + 2;
        ninc = 2;
    } else if mode == 2 {
        ms = m + 1;
        ninc = 2;
    }

    if m > 1 {
        let mut np1 = ms;
        while np1 <= nlat {
            let n = np1 - 1;
            let imn = if n >= l { imndx(m, n) } else { indx(m, n) };
            for i in 1..=late {
                let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                let src1 = ((km2 * late + (i - 1)) * nlat) + (n - 2);
                let src2 = ((km0 * late + (i - 1)) * nlat) + (n - 2);
                let src3 = ((km2 * late + (i - 1)) * nlat) + (np1 - 1);
                pmn[dst] = abel[imn] * pmn[src1] + bbel[imn] * pmn[src2] - cbel[imn] * pmn[src3];
            }
            np1 += ninc;
        }
    } else if m == 0 {
        let mut np1 = ms;
        while np1 <= nlat {
            for i in 1..=late {
                let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                let src = (i - 1) * nlat + (np1 - 1);
                pmn[dst] = p0n[src];
            }
            np1 += ninc;
        }
    } else {
        let mut np1 = ms;
        while np1 <= nlat {
            for i in 1..=late {
                let dst = ((km0 * late + (i - 1)) * nlat) + (np1 - 1);
                let src = (i - 1) * nlat + (np1 - 1);
                pmn[dst] = p1n[src];
            }
            np1 += ninc;
        }
    }

    let kmt = km0;
    km0 = km2;
    km2 = km1;
    km1 = kmt;
    *km_state = (km0, km1, km2);
    kmt
}
