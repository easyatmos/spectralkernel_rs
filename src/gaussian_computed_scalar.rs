use crate::gaussian_stored_scalar::shagsp_impl;

/// Build the core workspace used by `shagci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shagci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    shagsp_impl(nlat, nlon)
}

/// Build the core workspace used by `shsgci`.
///
/// # Parameters
/// - `nlat`: Number of latitudes in the grid.
/// - `nlon`: Number of longitudes in the grid.
///
/// # Returns
/// `Ok` with the workspace vector, or `Err` with a error code.
///
/// This routine follows this crate's spectral workspace and coefficient conventions.
pub fn shsgci_core(nlat: usize, nlon: usize) -> Result<Vec<f32>, i32> {
    shagsp_impl(nlat, nlon)
}
