from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


SUMMARY_MAP = {
    "gaqd_impl": "Compute Gaussian colatitudes and quadrature weights using a Fourier-Newton quadrature method.",
    "gaqd": "Python wrapper for `gaqd_impl` that returns NumPy arrays for Gaussian colatitudes and weights.",
    "getlegfunc_impl": "Evaluate normalized associated Legendre basis values for a latitude and triangular truncation.",
    "getlegfunc": "Python wrapper for `getlegfunc_impl` that returns the Legendre table as a NumPy array.",
    "hrffti_impl": "Build the trigonometric workspace used by the real Fourier transform routines.",
    "hrffti": "Python wrapper for `hrffti_impl` that exposes the FFT workspace as a NumPy array.",
    "ihgeod_impl": "Generate the geodesic grid coordinates on the sphere.",
    "ihgeod": "Python wrapper for `ihgeod_impl` that returns geodesic coordinates as NumPy arrays.",
    "lap_impl": "Apply the spherical Laplacian in packed spectral space.",
    "lap": "Python wrapper for `lap_impl` that accepts and returns NumPy complex arrays.",
    "invlap_impl": "Apply the inverse spherical Laplacian in packed spectral space.",
    "invlap": "Python wrapper for `invlap_impl` that accepts and returns NumPy complex arrays.",
    "invlap_nogil": "Python wrapper for `invlap_impl` that releases the GIL during computation.",
    "multsmoothfact_impl": "Multiply spectral coefficients by the supplied smoothing factors.",
    "multsmoothfact": "Python wrapper for `multsmoothfact_impl` that accepts NumPy complex arrays.",
    "onedtotwod_impl": "Expand packed triangular spectral coefficients into separate cosine and sine coefficient planes.",
    "onedtotwod": "Python wrapper for `onedtotwod_impl` that reshapes packed coefficients into NumPy arrays.",
    "onedtotwod_vrtdiv_impl": "Expand packed vorticity and divergence spectra into separate coefficient planes.",
    "onedtotwod_vrtdiv": "Python wrapper for `onedtotwod_vrtdiv_impl` that returns NumPy arrays.",
    "twodtooned_impl": "Pack a two-dimensional spectral coefficient layout into the triangular one-dimensional representation.",
    "twodtooned": "Python wrapper for `twodtooned_impl` that returns packed NumPy arrays.",
    "twodtooned_vrtdiv_impl": "Pack two-dimensional vorticity and divergence coefficient arrays into one-dimensional spectra.",
    "twodtooned_vrtdiv": "Python wrapper for `twodtooned_vrtdiv_impl` that returns packed NumPy arrays.",
    "specintrp_impl": "Interpolate a scalar spectral field at a target longitude using precomputed Legendre values.",
    "specintrp": "Python wrapper for `specintrp_impl` that validates NumPy inputs.",
    "legin_compute": "Compute one of the stored Legendre initialization tables selected by the requested mode.",
    "shaec_impl": "Analyze scalar fields on a regular grid using computed Legendre tables.",
    "shaec": "Python wrapper for `shaec_impl` that accepts rank-2 or rank-3 NumPy arrays.",
    "shaeci_impl": "Initialize the workspace required by `shaec_impl`.",
    "shaeci": "Python wrapper for `shaeci_impl` that returns the initialized workspace.",
    "shaes_impl": "Analyze scalar fields on a regular grid using stored Legendre tables.",
    "shaes": "Python wrapper for `shaes_impl` that accepts rank-2 or rank-3 NumPy arrays.",
    "shaesi_impl": "Initialize the workspace required by `shaes_impl`.",
    "shaesi": "Python wrapper for `shaesi_impl` that returns the initialized workspace.",
    "shagc_impl": "Analyze scalar fields on a Gaussian grid using computed Legendre tables.",
    "shagc": "Python wrapper for `shagc_impl` that accepts rank-2 or rank-3 NumPy arrays.",
    "shagci_impl": "Initialize the workspace required by `shagc_impl`.",
    "shagci": "Python wrapper for `shagci_impl` that returns the initialized workspace.",
    "shags_impl": "Analyze scalar fields on a Gaussian grid using stored Legendre tables.",
    "shags": "Python wrapper for `shags_impl` that accepts rank-2 or rank-3 NumPy arrays.",
    "shagsi_impl": "Initialize the workspace required by `shags_impl`.",
    "shagsi": "Python wrapper for `shagsi_impl` that returns the initialized workspace.",
    "shsec_impl": "Synthesize scalar fields on a regular grid using computed Legendre tables.",
    "shsec_impl_parallel": "Parallel synthesis of scalar fields on a regular grid using computed Legendre tables.",
    "shsec": "Python wrapper for `shsec_impl` that returns NumPy arrays.",
    "shsec_nogil": "Python wrapper for `shsec_impl_parallel` that releases the GIL during synthesis.",
    "shseci_impl": "Initialize the workspace required by `shsec_impl`.",
    "shseci": "Python wrapper for `shseci_impl` that returns the initialized workspace.",
    "shses_impl": "Synthesize scalar fields on a regular grid using stored Legendre tables.",
    "shses": "Python wrapper for `shses_impl` that returns NumPy arrays.",
    "shsesi_impl": "Initialize the workspace required by `shses_impl`.",
    "shsesi": "Python wrapper for `shsesi_impl` that returns the initialized workspace.",
    "shsgc_impl": "Synthesize scalar fields on a Gaussian grid using computed Legendre tables.",
    "shsgc": "Python wrapper for `shsgc_impl` that returns NumPy arrays.",
    "shsgci_impl": "Initialize the workspace required by `shsgc_impl`.",
    "shsgci": "Python wrapper for `shsgci_impl` that returns the initialized workspace.",
    "shsgs_impl": "Synthesize scalar fields on a Gaussian grid using stored Legendre tables.",
    "shsgs": "Python wrapper for `shsgs_impl` that returns NumPy arrays.",
    "shsgsi_impl": "Initialize the workspace required by `shsgs_impl`.",
    "shsgsi": "Python wrapper for `shsgsi_impl` that returns the initialized workspace.",
    "vhaec_impl": "Analyze vector fields on a regular grid using computed Legendre tables.",
    "vhaec_impl_parallel": "Parallel analysis of vector fields on a regular grid using computed Legendre tables.",
    "vhaec": "Python wrapper for `vhaec_impl` using the default vector layout.",
    "vhaec_nogil": "Python wrapper for `vhaec_impl_parallel` that releases the GIL during analysis.",
    "vhaec_ityp": "Python wrapper for `vhaec_impl` with an explicit `ityp` selector.",
    "vhaeci_impl": "Initialize the workspace required by `vhaec_impl`.",
    "vhaeci": "Python wrapper for `vhaeci_impl` that returns the initialized workspace.",
    "vhaes_impl": "Analyze vector fields on a regular grid using stored Legendre tables.",
    "vhaes": "Python wrapper for `vhaes_impl` using the default vector layout.",
    "vhaes_nogil": "Python wrapper for `vhaes_impl` that releases the GIL during analysis.",
    "vhaes_latpar_nogil": "Python wrapper for the latitude-parallel `vhaes_impl` path that releases the GIL.",
    "vhaes_ityp": "Python wrapper for `vhaes_impl` with an explicit `ityp` selector.",
    "vhaesi_impl": "Initialize the workspace required by `vhaes_impl`.",
    "vhaesi_impl_parallel": "Parallel initializer for the workspace required by `vhaes_impl`.",
    "vhaesi": "Python wrapper for `vhaesi_impl` that returns the initialized workspace.",
    "vhagc_impl": "Analyze vector fields on a Gaussian grid using computed Legendre tables.",
    "vhagc": "Python wrapper for `vhagc_impl` using the default vector layout.",
    "vhagc_ityp": "Python wrapper for `vhagc_impl` with an explicit `ityp` selector.",
    "vhagci_impl": "Initialize the workspace required by `vhagc_impl`.",
    "vhagci": "Python wrapper for `vhagci_impl` that returns the initialized workspace.",
    "vhags_impl": "Analyze vector fields on a Gaussian grid using stored Legendre tables.",
    "vhags": "Python wrapper for `vhags_impl` using the default vector layout.",
    "vhags_nogil": "Python wrapper for `vhags_impl` that releases the GIL during analysis.",
    "vhags_latpar_nogil": "Python wrapper for the latitude-parallel `vhags_impl` path that releases the GIL.",
    "vhags_ityp": "Python wrapper for `vhags_impl` with an explicit `ityp` selector.",
    "vhagsi_impl": "Initialize the workspace required by `vhags_impl`.",
    "vhagsi": "Python wrapper for `vhagsi_impl` that returns the initialized workspace.",
    "vhsec_impl": "Synthesize vector fields on a regular grid using computed Legendre tables.",
    "vhsec": "Python wrapper for `vhsec_impl` using the default vector layout.",
    "vhsec_ityp": "Python wrapper for `vhsec_impl` with an explicit `ityp` selector.",
    "vhseci_impl": "Initialize the workspace required by `vhsec_impl`.",
    "vhseci": "Python wrapper for `vhseci_impl` that returns the initialized workspace.",
    "vhses_impl": "Synthesize vector fields on a regular grid using stored Legendre tables.",
    "vhses": "Python wrapper for `vhses_impl` using the default vector layout.",
    "vhses_ityp": "Python wrapper for `vhses_impl` with an explicit `ityp` selector.",
    "vhsesi_impl": "Initialize the workspace required by `vhses_impl`.",
    "vhsesi_impl_parallel": "Parallel initializer for the workspace required by `vhses_impl`.",
    "vhsesi": "Python wrapper for `vhsesi_impl` that returns the initialized workspace.",
    "vhsgc_impl": "Synthesize vector fields on a Gaussian grid using computed Legendre tables.",
    "vhsgc": "Python wrapper for `vhsgc_impl` using the default vector layout.",
    "vhsgc_ityp": "Python wrapper for `vhsgc_impl` with an explicit `ityp` selector.",
    "vhsgci_impl": "Initialize the workspace required by `vhsgc_impl`.",
    "vhsgci": "Python wrapper for `vhsgci_impl` that returns the initialized workspace.",
    "vhsgs_impl": "Synthesize vector fields on a Gaussian grid using stored Legendre tables.",
    "vhsgs": "Python wrapper for `vhsgs_impl` using the default vector layout.",
    "vhsgs_ityp": "Python wrapper for `vhsgs_impl` with an explicit `ityp` selector.",
    "vhsgsi_impl": "Initialize the workspace required by `vhsgs_impl`.",
    "vhsgsi": "Python wrapper for `vhsgsi_impl` that returns the initialized workspace.",
    "sea1_debug": "Python helper that exposes the scalar synthesis workspace built by `sea1_impl` for inspection.",
    "regular_stored_init_nogil": "Build regular-grid stored workspaces for scalar and vector transforms while releasing the GIL.",
    "gaussian_stored_init_nogil": "Build Gaussian-grid stored workspaces for scalar and vector transforms while releasing the GIL.",
    "zfinit_debug": "Python helper that exposes the scalar analysis workspace built by `zfinit_impl` for inspection.",
}


PARAM_MAP = {
    "nlat": "Number of latitudes in the grid.",
    "nlon": "Number of longitudes in the grid.",
    "nt": "Number of stacked fields processed together.",
    "lwork": "Length of the caller-provided work array.",
    "ldwork": "Length of the auxiliary workspace expected by the low-level interface.",
    "nmdim": "Number of packed spectral coefficients per field.",
    "rsphere": "Sphere radius used to scale Laplacian operators.",
    "rlon": "Longitude in radians at which interpolation is evaluated.",
    "ntrunc": "Triangular spectral truncation.",
    "lat": "Latitude in radians.",
    "m": "Zonal wavenumber or refinement level, depending on the routine.",
    "n": "Total spherical harmonic degree.",
    "theta": "Colatitude in radians.",
    "th": "Colatitude in radians.",
    "isym": "Symmetry selector used by Legendre tables.",
    "ityp": "Vector storage selector controlling the coefficient families in use.",
    "g": "Input scalar grid values stored in `(nlat, nlon[, nt])` order.",
    "v": "Input vector component stored in `(nlat, nlon[, nt])` order.",
    "w": "Input workspace or secondary component, depending on the routine.",
    "a": "Cosine spectral coefficients in scalar layout.",
    "b": "Sine spectral coefficients in scalar layout.",
    "br": "First vector coefficient family in vector layout.",
    "bi": "Second vector coefficient family in vector layout.",
    "cr": "Third vector coefficient family in vector layout.",
    "ci": "Fourth vector coefficient family in vector layout.",
    "dataspec": "Packed complex spectral coefficients.",
    "vrtspec": "Packed vorticity spectral coefficients.",
    "divspec": "Packed divergence spectral coefficients.",
    "datnm": "Packed spectral coefficients ordered by `(m, n)`.",
    "pnm": "Precomputed Legendre values compatible with the chosen truncation.",
    "py": "Python interpreter token supplied by PyO3.",
    "data": "Rank-2 real array analyzed by the internal Fourier kernel.",
    "mode": "Selector controlling which stored Legendre recurrence table is generated.",
    "l": "Leading degree or table width used by the selected recurrence branch.",
    "pmn": "Output buffer that receives the generated Legendre values.",
    "km_state": "Mutable state tuple updated with the recurrence bookkeeping indices.",
    "walin": "Stored scalar analysis workspace produced by `alinit_impl`.",
    "smooth": "Per-coefficient smoothing factors applied in spectral space.",
    "wshaec": "Workspace initialized by `shaeci_impl` for regular-grid scalar analysis with computed tables.",
    "lshaec": "Declared length of the `wshaec` workspace.",
    "wshaes": "Workspace initialized by `shaesi_impl` for regular-grid scalar analysis with stored tables.",
    "lshaes": "Declared length of the `wshaes` workspace.",
    "wshagc": "Workspace initialized by `shagci_impl` for Gaussian-grid scalar analysis with computed tables.",
    "lshagc": "Declared length of the `wshagc` workspace.",
    "wshags": "Workspace initialized by `shagsi_impl` for Gaussian-grid scalar analysis with stored tables.",
    "lshags": "Declared length of the `wshags` workspace.",
    "wshsec": "Workspace initialized by `shseci_impl` for regular-grid scalar synthesis with computed tables.",
    "lshsec": "Declared length of the `wshsec` workspace.",
    "wshses": "Workspace initialized by `shsesi_impl` for regular-grid scalar synthesis with stored tables.",
    "lshses": "Declared length of the `wshses` workspace.",
    "wshsgc": "Workspace initialized by `shsgci_impl` for Gaussian-grid scalar synthesis with computed tables.",
    "lshsgc": "Declared length of the `wshsgc` workspace.",
    "wshsgs": "Workspace initialized by `shsgsi_impl` for Gaussian-grid scalar synthesis with stored tables.",
    "lshsgs": "Declared length of the `wshsgs` workspace.",
    "cp": "Fourier coefficients of the scalar Legendre basis.",
    "cv": "Fourier coefficients of the vector v basis.",
    "cw": "Fourier coefficients of the vector w basis.",
    "czv": "Precomputed Gaussian v-basis coefficients.",
    "czw": "Precomputed Gaussian w-basis coefficients.",
    "wzfin": "Computed scalar analysis workspace produced by `zfinit_impl`.",
    "wzvin": "Computed vector analysis workspace for the v basis.",
    "wzwin": "Computed vector analysis workspace for the w basis.",
    "wvhaec": "Workspace initialized by `vhaeci_impl` for regular-grid vector analysis with computed tables.",
    "lvhaec": "Declared length of the `wvhaec` workspace.",
    "wvhaes": "Workspace initialized by `vhaesi_impl` for regular-grid vector analysis with stored tables.",
    "lvhaes": "Declared length of the `wvhaes` workspace.",
    "wvhagc": "Workspace initialized by `vhagci_impl` for Gaussian-grid vector analysis with computed tables.",
    "lvhagc": "Declared length of the `wvhagc` workspace.",
    "wvhags": "Workspace initialized by `vhagsi_impl` for Gaussian-grid vector analysis with stored tables.",
    "lvhags": "Declared length of the `wvhags` workspace.",
    "wvhsec": "Workspace initialized by `vhseci_impl` for regular-grid vector synthesis with computed tables.",
    "lvhsec": "Declared length of the `wvhsec` workspace.",
    "wvhses": "Workspace initialized by `vhsesi_impl` for regular-grid vector synthesis with stored tables.",
    "lvhses": "Declared length of the `wvhses` workspace.",
    "wvhsgc": "Workspace initialized by `vhsgci_impl` for Gaussian-grid vector synthesis with computed tables.",
    "lvhsgc": "Declared length of the `wvhsgc` workspace.",
    "wvhsgs": "Workspace initialized by `vhsgsi_impl` for Gaussian-grid vector synthesis with stored tables.",
    "lvhsgs": "Declared length of the `wvhsgs` workspace.",
    "lshaes_work": "Requested storage for the scalar regular-grid stored-analysis workspace.",
    "lvhaes_work": "Requested storage for the vector regular-grid stored-analysis workspace.",
    "ldwork_scalar": "Auxiliary workspace length used by the scalar stored initializer.",
    "ldwork_vector": "Auxiliary workspace length used by the vector stored initializer.",
    "lshags_work": "Requested storage for the scalar Gaussian stored-analysis workspace.",
    "lshags_dwork": "Auxiliary workspace length for the scalar Gaussian stored initializer.",
    "lvhags_dwork": "Auxiliary workspace length for the Gaussian vector analysis initializer.",
    "lvhsgs_dwork": "Auxiliary workspace length for the Gaussian vector synthesis initializer.",
}


RETURN_MAP = {
    "Vec<f32>": "A contiguous workspace or coefficient vector in storage.",
    "Vec<f64>": "A contiguous workspace or coefficient vector stored in double precision.",
    "f32": "The interpolated or evaluated scalar value.",
    "f64": "The evaluated basis value.",
    "usize": "The computed index, table length, or updated state position returned by the routine.",
    "(Vec<f32>, i32)": "A tuple containing the workspace/result vector and the error code.",
    "(Vec<f64>, Vec<f64>)": "A pair of double-precision work tables.",
    "(Vec<f64>, Vec<f64>, Vec<f64>)": "Three double-precision recurrence or workspace tables.",
    "(Vec<f64>, Vec<f64>, i32)": "Gaussian colatitudes, quadrature weights, and an error code.",
    "(Vec<f32>, Vec<f32>, Vec<f32>)": "Three coordinate arrays describing the generated grid.",
    "Result<Vec<f32>, i32>": "`Ok` with the workspace vector, or `Err` with a error code.",
    "PyResult<f32>": "A Python result containing the interpolated scalar value.",
    "PyResult<Vec<Complex32>>": "A Python result containing the transformed packed spectral coefficients.",
    "PyResult<Vec<f32>>": "A Python result containing the computed values as a contiguous vector.",
    "PyResult<(Vec<f32>, Vec<f32>, i32)>": "A Python result containing the cosine coefficients, sine coefficients, and an error code.",
    "PyResult<(Vec<Complex32>, Vec<Complex32>)>": "A Python result containing the transformed complex coefficient arrays.",
    "PyResult<(Vec<f32>, Vec<f32>)>": "A Python result containing the cosine and sine coefficient arrays.",
    "PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>": "A Python result containing four coefficient arrays in vector layout.",
    "PyResult<Py<PyAny>>": "A Python object containing the returned NumPy array.",
    "PyResult<(Py<PyAny>, Py<PyAny>, i32)>": "Two NumPy arrays together with a error code.",
    "PyResult<(Py<PyAny>, Py<PyAny>)>": "Two NumPy arrays containing the returned coefficient fields.",
    "PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)>": "Four NumPy arrays containing the returned coefficient families.",
    "PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)>": "Four NumPy arrays together with a error code.",
    "PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, i32)>": "Two one-dimensional NumPy arrays together with a error code.",
    "PyResult<(Bound<'py, PyArray1<f32>>, i32)>": "A one-dimensional NumPy workspace array together with a error code.",
    "PyResult<Bound<'py, PyArray1<f32>>>": "A one-dimensional NumPy array containing the computed workspace.",
    "PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>": "Three one-dimensional NumPy arrays containing the generated coordinate data.",
}


SUMMARY_MAP.update(
    {
        "SphereOps": "High-level spectral-operator handle that caches workspaces for one spherical grid.",
        "regular": "Create a high-level operator set for an equally spaced global latitude-longitude grid.",
        "gaussian": "Create a high-level operator set for a Gaussian latitude-longitude grid.",
        "nlat": "Return the number of latitude points owned by this operator set.",
        "nlon": "Return the number of longitude points owned by this operator set.",
        "radius": "Return the sphere radius used when scaling differential operators.",
        "grid_type": "Return whether this operator uses a regular or Gaussian grid.",
        "legfunc": "Return whether Legendre functions are stored or computed for this operator.",
        "lat": "Return latitude coordinates in degrees for the operator grid.",
        "lon": "Return longitude coordinates in degrees for the operator grid.",
        "weights": "Return Gaussian quadrature weights, or unit weights for regular grids.",
        "geo_to_math_scalar": "Shift a scalar field from geographical ordering into mathematical latitude-longitude ordering.",
        "math_to_geo_scalar": "Shift a scalar field from mathematical ordering back to geographical ordering.",
        "geo_to_math_vector": "Shift vector components from geographical ordering into mathematical ordering.",
        "math_to_geo_vector": "Shift vector components from mathematical ordering back to geographical ordering.",
        "scalar_shift": "Apply the half-grid scalar shift between offset and regular grids.",
        "vector_shift": "Apply the half-grid vector shift to both vector components.",
        "scalar_to_spec": "Analyze grid-space scalar fields into packed complex spherical-harmonic coefficients.",
        "spec_to_scalar": "Synthesize grid-space scalar fields from packed complex spherical-harmonic coefficients.",
        "vector_to_spec": "Analyze grid-space vector components into packed vorticity and divergence spectra.",
        "spec_to_vector": "Synthesize grid-space vector components from packed vorticity and divergence spectra.",
        "truncate_scalar": "Apply triangular spectral truncation to a scalar grid field and synthesize it back to grid space.",
        "grad": "Compute the horizontal gradient of a scalar spectral field.",
        "gradient": "Alias for computing the horizontal gradient of a scalar spectral field.",
        "gradient_grid": "Analyze a scalar grid field and return its horizontal gradient on the grid.",
        "gradient_from_spec": "Synthesize the horizontal gradient from packed scalar spectral coefficients.",
        "inverse_grad": "Recover the scalar potential whose gradient best matches the supplied vector field.",
        "inverse_gradient": "Alias for recovering a scalar potential from vector-gradient components.",
        "div": "Compute horizontal divergence from vector components on the grid.",
        "divergence": "Alias for computing horizontal divergence from vector components on the grid.",
        "inverse_div": "Recover an irrotational vector field from a divergence field.",
        "inverse_divergence": "Alias for recovering an irrotational vector field from divergence.",
        "vort": "Compute vertical vorticity from vector components on the grid.",
        "vorticity": "Alias for computing vertical vorticity from vector components on the grid.",
        "inverse_vort": "Recover a non-divergent vector field from a vorticity field.",
        "inverse_vorticity": "Alias for recovering a non-divergent vector field from vorticity.",
        "laplacian": "Apply the scalar spherical Laplacian to a grid-space field.",
        "inverse_laplacian": "Apply the inverse scalar spherical Laplacian to a grid-space field.",
        "wind_to_vrtdiv": "Analyze vector wind components into vorticity and divergence grid fields.",
        "vrtdiv_to_wind": "Synthesize vector wind components from vorticity and divergence grid fields.",
        "reconstruct_vector_vts": "Reconstruct vector components using the vector triangular synthesis path.",
        "streamfunction_to_wind": "Compute a non-divergent wind field from a streamfunction.",
        "velocity_potential_to_wind": "Compute an irrotational wind field from a velocity potential.",
        "streamfunction_velocity_potential": "Recover streamfunction and velocity potential from vector wind components.",
        "helmholtz_decompose": "Decompose vector wind components into non-divergent and irrotational parts.",
        "helmsph": "Solve the scalar Helmholtz equation on the sphere using the spectral plan.",
        "coriolis": "Compute the Coriolis parameter on the operator grid.",
        "absolute_vorticity": "Add planetary vorticity to relative vorticity computed from vector components.",
        "kinetic_energy": "Compute pointwise kinetic energy from vector components.",
        "k_cross": "Rotate vector components by the vertical unit vector.",
        "advect_scalar": "Compute horizontal scalar advection by a vector field.",
        "advect_scalar_x": "Compute the longitudinal contribution to scalar advection.",
        "advect_scalar_y": "Compute the latitudinal contribution to scalar advection.",
        "advect_scalar_components": "Return both component contributions to scalar advection.",
        "flux_div_scalar": "Compute conservative scalar flux divergence.",
        "split_advect_scalar": "Compute scalar advection with the split advective/conservative form.",
        "spectral_filter": "Apply a triangular spectral filter to a scalar grid field.",
        "hyperdiffusion": "Apply scalar hyperdiffusion using powers of the spherical Laplacian.",
        "vector_hyperdiffusion": "Apply vector hyperdiffusion to both vector components.",
        "vector_laplacian": "Apply the vector spherical Laplacian to vector components.",
        "inverse_vector_laplacian": "Apply the inverse vector spherical Laplacian to vector components.",
        "project_nondivergent": "Project vector components onto the non-divergent Helmholtz component.",
        "project_irrotational": "Project vector components onto the irrotational Helmholtz component.",
        "rotated_grad": "Compute the gradient rotated by the vertical unit vector.",
        "vorticity_flux": "Compute the vector-invariant vorticity flux term.",
        "kinetic_energy_grad": "Compute the gradient of kinetic energy.",
        "momentum_vector_invariant": "Compute the vector-invariant momentum tendency terms.",
        "advect_vector": "Compute vector advection tendencies using the high-level operator API.",
        "scalar_advection_rhs": "Compute scalar-advection tendency for transport equations on the sphere.",
        "barotropic_vorticity_rhs": "Compute the barotropic-vorticity tendency from relative vorticity.",
        "shallow_water_rhs": "Compute shallow-water tendencies using the vector-invariant form.",
    }
)

PARAM_MAP.update(
    {
        "ops": "High-level SphereOps spectral operator set used for transforms and differential operators.",
        "q": "Scalar tracer field on the operator grid.",
        "h": "Fluid layer depth or height field on the operator grid.",
        "u": "Zonal or first vector component on the operator grid.",
        "f": "Scalar grid field on the operator grid.",
        "spec": "Packed complex spherical-harmonic coefficient array.",
        "chispec": "Packed scalar spectral coefficients representing a potential field.",
        "vort_spec": "Packed vorticity spectral coefficients.",
        "div_spec": "Packed divergence spectral coefficients.",
        "vrtspec": "Packed vorticity spectral coefficients.",
        "truncation": "Optional triangular spectral truncation to apply.",
        "backend": "Execution backend selector for supported transform paths.",
        "legfunc": "Legendre-function strategy: stored tables or computed recurrences.",
        "radius": "Sphere radius used to scale gradients, Laplacians, and dynamics.",
        "omega": "Planetary rotation rate.",
        "form": "Equation form selector for the high-level tendency routine.",
        "kind": "Spectral filter family.",
        "strength": "Filter damping strength.",
        "order": "Even differential order used by filters or hyperdiffusion.",
        "tau": "Optional damping time scale for hyperdiffusion.",
        "nu": "Optional explicit diffusion coefficient.",
        "absolute": "Whether to include planetary vorticity.",
        "zero_mean": "Whether to remove the unresolved mean before inverse-Laplacian reconstruction.",
    }
)

ITEM_RE = re.compile(r"^(?P<indent>\s*)pub\s+(?P<kind>fn|struct)\s+(?P<name>[A-Za-z0-9_]+)")


def fallback_summary(name: str) -> str:
    if name.endswith("_impl_parallel"):
        return f"Parallel implementation of `{name.removesuffix('_impl_parallel')}_impl`."
    if name.endswith("_impl"):
        return f"Core Rust implementation of `{name.removesuffix('_impl')}`."
    if name.endswith("_core"):
        return f"Build the core workspace used by `{name.removesuffix('_core')}`."
    return f"Rust entry point for `{name}`."


def split_params(signature: str) -> list[tuple[str, str]]:
    inside = signature[signature.find("(") + 1 : signature.rfind(")")]
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in inside:
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            text = "".join(current).strip()
            if text:
                parts.append(text)
            current = []
        else:
            current.append(char)
    text = "".join(current).strip()
    if text:
        parts.append(text)

    params: list[tuple[str, str]] = []
    for part in parts:
        if ":" not in part:
            continue
        name, typ = [item.strip() for item in part.split(":", 1)]
        if name == "py":
            continue
        params.append((name, typ))
    return params


def parse_signature(lines: list[str], start: int) -> tuple[str, int]:
    collected = [lines[start]]
    end = start
    while end + 1 < len(lines) and "{" not in lines[end]:
        end += 1
        collected.append(lines[end])
    signature = " ".join(line.strip() for line in collected)
    return signature, end


def build_doc(indent: str, name: str, signature: str, kind: str = "fn") -> list[str]:
    doc = [f"{indent}/// {SUMMARY_MAP.get(name, fallback_summary(name))}"]
    if kind == "struct":
        doc.extend(
            [
                f"{indent}///",
                f"{indent}/// The object owns a SpectralPlan so repeated operations reuse cached",
                f"{indent}/// initialization data instead of rebuilding work arrays for every transform.",
            ]
        )
        return doc
    params = split_params(signature)
    if params:
        doc.extend([f"{indent}///", f"{indent}/// # Parameters"])
        for param_name, _ in params:
            desc = PARAM_MAP.get(param_name, f"Parameter `{param_name}` passed through to the routine.")
            doc.append(f"{indent}/// - `{param_name}`: {desc}")

    if "->" in signature:
        return_type = signature.split("->", 1)[1].split("{", 1)[0].strip()
        desc = RETURN_MAP.get(return_type)
        if desc is None:
            if return_type.startswith("PyResult<(") and "PyArray3" in return_type:
                desc = "A Python result containing three three-dimensional NumPy arrays."
            elif return_type.startswith("PyResult<(") and return_type.count("Py<PyAny>") == 2:
                desc = "A Python result containing two NumPy arrays."
            elif return_type.startswith("PyResult<(") and return_type.count("Py<PyAny>") == 4:
                desc = "A Python result containing four NumPy arrays."
            elif return_type.startswith("PyResult<Bound<'py, PyArray1"):
                desc = "A Python result containing a one-dimensional NumPy array."
            elif return_type.startswith("PyResult<"):
                desc = "A Python result containing the values produced by this routine."
            else:
                desc = "The value produced by this routine."
        doc.extend([f"{indent}///", f"{indent}/// # Returns", f"{indent}/// {desc}"])

    if name.endswith(("_impl", "_core", "_impl_parallel")):
        doc.extend(
            [
                f"{indent}///",
                f"{indent}/// This routine follows this crate's spectral workspace and coefficient conventions.",
            ]
        )
    elif name in {
        "scalar_advection_rhs",
        "barotropic_vorticity_rhs",
        "shallow_water_rhs",
        "regular",
        "gaussian",
        "gradient_grid",
        "wind_to_vrtdiv",
        "vrtdiv_to_wind",
        "helmholtz_decompose",
        "spectral_filter",
        "hyperdiffusion",
        "vector_hyperdiffusion",
        "momentum_vector_invariant",
        "advect_vector",
    }:
        doc.extend(
            [
                f"{indent}///",
                f"{indent}/// This is a high-level Rust/Python-facing convenience API built on top of this crate's lower-level spectral kernels.",
            ]
        )
    return doc


def process_file(path: pathlib.Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        match = ITEM_RE.match(line)
        if not match:
            out.append(line)
            i += 1
            continue

        prev = ""
        j = len(out) - 1
        while j >= 0 and out[j].strip() == "":
            j -= 1
        if j >= 0:
            prev = out[j].lstrip()

        doc_start = None
        doc_end = None
        k = len(out) - 1
        while k >= 0 and out[k].strip() == "":
            k -= 1
        while k >= 0 and out[k].lstrip().startswith("///"):
            doc_start = k
            k -= 1
        if doc_start is not None:
            doc_end = len(out)
        elif k >= 0 and out[k].lstrip().startswith("#[pyfunction]"):
            marker_index = k
            k -= 1
            while k >= 0 and out[k].strip() == "":
                k -= 1
            while k >= 0 and out[k].lstrip().startswith("///"):
                doc_start = k
                k -= 1
            if doc_start is not None:
                doc_end = marker_index

        if doc_start is not None and doc_end is not None:
            existing_doc = "\n".join(out[doc_start:doc_end])
            autogenerated_markers = (
                "# Parameters",
                "# Returns",
                "naming and workspace conventions",
                "compatible workspaces",
                "repeated operations reuse",
                "The value produced by this routine.",
                "Python wrapper for `",
                "Core Rust implementation of `",
                "Rust entry point for `",
            )
            if any(marker in existing_doc for marker in autogenerated_markers):
                del out[doc_start:doc_end]
                changed = True
            else:
                out.append(line)
                i += 1
                continue
        elif prev.startswith("///"):
            out.append(line)
            i += 1
            continue

        signature, _ = parse_signature(lines, i)
        out.extend(build_doc(match.group("indent"), match.group("name"), signature, match.group("kind")))
        out.append(line)
        changed = True
        i += 1

    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    modified = []
    for path in sorted(SRC.glob("*.rs")):
        if process_file(path):
            modified.append(path.relative_to(ROOT).as_posix())
    for item in modified:
        print(item)


if __name__ == "__main__":
    main()
