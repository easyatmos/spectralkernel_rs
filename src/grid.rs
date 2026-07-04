use crate::gaqd::gaqd_impl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridType {
    Regular,
    Gaussian,
}

impl GridType {
    /// Rust entry point for `as_str`.
    ///
    /// # Returns
    /// The value produced by this routine.
    pub fn as_str(self) -> &'static str {
        match self {
            GridType::Regular => "regular",
            GridType::Gaussian => "gaussian",
        }
    }
}

/// Rust entry point for `regular_latitudes`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
pub fn regular_latitudes(nlat: usize) -> Vec<f32> {
    if nlat % 2 == 1 {
        (0..nlat)
            .map(|i| 90.0_f32 - 180.0_f32 * i as f32 / (nlat as f32 - 1.0_f32))
            .collect()
    } else {
        let dlat = 180.0_f32 / nlat as f32;
        (0..nlat)
            .map(|i| 90.0_f32 - 0.5_f32 * dlat - dlat * i as f32)
            .collect()
    }
}

/// Rust entry point for `gaussian_latitudes_weights`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
///
/// # Returns
/// The value produced by this routine.
pub fn gaussian_latitudes_weights(nlat: usize) -> (Vec<f32>, Vec<f32>, i32) {
    let (theta, weights, ierr) = gaqd_impl(nlat as i32);
    let lat = theta
        .into_iter()
        .map(|theta| 90.0_f32 - (theta as f32).to_degrees())
        .collect();
    let weights = weights.into_iter().map(|w| w as f32).collect();
    (lat, weights, ierr)
}

/// Rust entry point for `longitudes`.
///
/// # Parameters
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// A contiguous workspace or coefficient vector in storage.
pub fn longitudes(nlon: usize) -> Vec<f32> {
    (0..nlon)
        .map(|i| 360.0_f32 * i as f32 / nlon as f32)
        .collect()
}
