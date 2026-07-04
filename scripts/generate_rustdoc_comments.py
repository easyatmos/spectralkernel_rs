from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


SUMMARY_MAP = {
    "gaqd_impl": "Compute Gaussian colatitudes and quadrature weights using the Fourier-Newton method adopted by SPHEREPACK.",
    "gaqd": "Python wrapper for `gaqd_impl` that returns NumPy arrays for Gaussian colatitudes and weights.",
    "getlegfunc_impl": "Evaluate normalized associated Legendre basis values for a latitude and triangular truncation.",
    "getlegfunc": "Python wrapper for `getlegfunc_impl` that returns the Legendre table as a NumPy array.",
    "hrffti_impl": "Build the trigonometric workspace used by the real Fourier transform routines.",
    "hrffti": "Python wrapper for `hrffti_impl` that exposes the FFT workspace as a NumPy array.",
    "ihgeod_impl": "Generate the geodesic grid coordinates produced by the original SPHEREPACK routine.",
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
    "onedtotwod_vrtdiv_impl": "Expand packed vorticity and divergence spectra into separate SPHEREPACK coefficient planes.",
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
    "vhaec": "Python wrapper for `vhaec_impl` using the default SPHEREPACK vector layout.",
    "vhaec_nogil": "Python wrapper for `vhaec_impl_parallel` that releases the GIL during analysis.",
    "vhaec_ityp": "Python wrapper for `vhaec_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhaeci_impl": "Initialize the workspace required by `vhaec_impl`.",
    "vhaeci": "Python wrapper for `vhaeci_impl` that returns the initialized workspace.",
    "vhaes_impl": "Analyze vector fields on a regular grid using stored Legendre tables.",
    "vhaes": "Python wrapper for `vhaes_impl` using the default SPHEREPACK vector layout.",
    "vhaes_nogil": "Python wrapper for `vhaes_impl` that releases the GIL during analysis.",
    "vhaes_latpar_nogil": "Python wrapper for the latitude-parallel `vhaes_impl` path that releases the GIL.",
    "vhaes_ityp": "Python wrapper for `vhaes_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhaesi_impl": "Initialize the workspace required by `vhaes_impl`.",
    "vhaesi_impl_parallel": "Parallel initializer for the workspace required by `vhaes_impl`.",
    "vhaesi": "Python wrapper for `vhaesi_impl` that returns the initialized workspace.",
    "vhagc_impl": "Analyze vector fields on a Gaussian grid using computed Legendre tables.",
    "vhagc": "Python wrapper for `vhagc_impl` using the default SPHEREPACK vector layout.",
    "vhagc_ityp": "Python wrapper for `vhagc_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhagci_impl": "Initialize the workspace required by `vhagc_impl`.",
    "vhagci": "Python wrapper for `vhagci_impl` that returns the initialized workspace.",
    "vhags_impl": "Analyze vector fields on a Gaussian grid using stored Legendre tables.",
    "vhags": "Python wrapper for `vhags_impl` using the default SPHEREPACK vector layout.",
    "vhags_nogil": "Python wrapper for `vhags_impl` that releases the GIL during analysis.",
    "vhags_latpar_nogil": "Python wrapper for the latitude-parallel `vhags_impl` path that releases the GIL.",
    "vhags_ityp": "Python wrapper for `vhags_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhagsi_impl": "Initialize the workspace required by `vhags_impl`.",
    "vhagsi": "Python wrapper for `vhagsi_impl` that returns the initialized workspace.",
    "vhsec_impl": "Synthesize vector fields on a regular grid using computed Legendre tables.",
    "vhsec": "Python wrapper for `vhsec_impl` using the default SPHEREPACK vector layout.",
    "vhsec_ityp": "Python wrapper for `vhsec_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhseci_impl": "Initialize the workspace required by `vhsec_impl`.",
    "vhseci": "Python wrapper for `vhseci_impl` that returns the initialized workspace.",
    "vhses_impl": "Synthesize vector fields on a regular grid using stored Legendre tables.",
    "vhses": "Python wrapper for `vhses_impl` using the default SPHEREPACK vector layout.",
    "vhses_ityp": "Python wrapper for `vhses_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhsesi_impl": "Initialize the workspace required by `vhses_impl`.",
    "vhsesi_impl_parallel": "Parallel initializer for the workspace required by `vhses_impl`.",
    "vhsesi": "Python wrapper for `vhsesi_impl` that returns the initialized workspace.",
    "vhsgc_impl": "Synthesize vector fields on a Gaussian grid using computed Legendre tables.",
    "vhsgc": "Python wrapper for `vhsgc_impl` using the default SPHEREPACK vector layout.",
    "vhsgc_ityp": "Python wrapper for `vhsgc_impl` with an explicit SPHEREPACK `ityp` selector.",
    "vhsgci_impl": "Initialize the workspace required by `vhsgc_impl`.",
    "vhsgci": "Python wrapper for `vhsgci_impl` that returns the initialized workspace.",
    "vhsgs_impl": "Synthesize vector fields on a Gaussian grid using stored Legendre tables.",
    "vhsgs": "Python wrapper for `vhsgs_impl` using the default SPHEREPACK vector layout.",
    "vhsgs_ityp": "Python wrapper for `vhsgs_impl` with an explicit SPHEREPACK `ityp` selector.",
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
    "ldwork": "Length of the auxiliary workspace expected by the SPHEREPACK interface.",
    "nmdim": "Number of packed spectral coefficients per field.",
    "rsphere": "Sphere radius used to scale Laplacian operators.",
    "rlon": "Longitude in radians at which interpolation is evaluated.",
    "ntrunc": "Triangular spectral truncation.",
    "lat": "Latitude in radians.",
    "m": "Zonal wavenumber or refinement level, depending on the routine.",
    "n": "Total spherical harmonic degree.",
    "theta": "Colatitude in radians.",
    "th": "Colatitude in radians.",
    "isym": "Symmetry selector used by SPHEREPACK Legendre tables.",
    "ityp": "SPHEREPACK vector storage selector controlling the coefficient families in use.",
    "g": "Input scalar grid values stored in `(nlat, nlon[, nt])` order.",
    "v": "Input vector component stored in `(nlat, nlon[, nt])` order.",
    "w": "Input workspace or secondary component, depending on the routine.",
    "a": "Cosine spectral coefficients in SPHEREPACK scalar layout.",
    "b": "Sine spectral coefficients in SPHEREPACK scalar layout.",
    "br": "First vector coefficient family in SPHEREPACK layout.",
    "bi": "Second vector coefficient family in SPHEREPACK layout.",
    "cr": "Third vector coefficient family in SPHEREPACK layout.",
    "ci": "Fourth vector coefficient family in SPHEREPACK layout.",
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
    "Vec<f32>": "A contiguous workspace or coefficient vector in SPHEREPACK-compatible storage.",
    "Vec<f64>": "A contiguous workspace or coefficient vector stored in double precision.",
    "f32": "The interpolated or evaluated scalar value.",
    "f64": "The evaluated basis value.",
    "usize": "The computed index, table length, or updated state position returned by the routine.",
    "(Vec<f32>, i32)": "A tuple containing the workspace/result vector and the SPHEREPACK-style error code.",
    "(Vec<f64>, Vec<f64>)": "A pair of double-precision work tables.",
    "(Vec<f64>, Vec<f64>, Vec<f64>)": "Three double-precision recurrence or workspace tables.",
    "(Vec<f64>, Vec<f64>, i32)": "Gaussian colatitudes, quadrature weights, and an error code.",
    "(Vec<f32>, Vec<f32>, Vec<f32>)": "Three coordinate arrays describing the generated grid.",
    "Result<Vec<f32>, i32>": "`Ok` with the workspace vector, or `Err` with a SPHEREPACK-style error code.",
    "PyResult<f32>": "A Python result containing the interpolated scalar value.",
    "PyResult<Vec<Complex32>>": "A Python result containing the transformed packed spectral coefficients.",
    "PyResult<Vec<f32>>": "A Python result containing the computed values as a contiguous vector.",
    "PyResult<(Vec<f32>, Vec<f32>, i32)>": "A Python result containing the cosine coefficients, sine coefficients, and an error code.",
    "PyResult<(Vec<Complex32>, Vec<Complex32>)>": "A Python result containing the transformed complex coefficient arrays.",
    "PyResult<(Vec<f32>, Vec<f32>)>": "A Python result containing the cosine and sine coefficient arrays.",
    "PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>": "A Python result containing four coefficient arrays in SPHEREPACK vector layout.",
    "PyResult<Py<PyAny>>": "A Python object containing the returned NumPy array.",
    "PyResult<(Py<PyAny>, Py<PyAny>, i32)>": "Two NumPy arrays together with a SPHEREPACK-style error code.",
    "PyResult<(Py<PyAny>, Py<PyAny>)>": "Two NumPy arrays containing the returned coefficient fields.",
    "PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)>": "Four NumPy arrays containing the returned SPHEREPACK coefficient families.",
    "PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, i32)>": "Four NumPy arrays together with a SPHEREPACK-style error code.",
    "PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, i32)>": "Two one-dimensional NumPy arrays together with a SPHEREPACK-style error code.",
    "PyResult<(Bound<'py, PyArray1<f32>>, i32)>": "A one-dimensional NumPy workspace array together with a SPHEREPACK-style error code.",
    "PyResult<Bound<'py, PyArray1<f32>>>": "A one-dimensional NumPy array containing the computed workspace.",
    "PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>": "Three one-dimensional NumPy arrays containing the generated coordinate data.",
}


FUNCTION_RE = re.compile(r"^(?P<indent>\s*)pub\s+fn\s+(?P<name>[A-Za-z0-9_]+)")


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


def build_doc(indent: str, name: str, signature: str) -> list[str]:
    doc = [f"{indent}/// {SUMMARY_MAP.get(name, fallback_summary(name))}"]
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
                f"{indent}/// This routine follows the SPHEREPACK naming and workspace conventions so it remains easy to compare with the original Fortran sources under `fortran/`.",
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
        match = FUNCTION_RE.match(line)
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
                "SPHEREPACK naming and workspace conventions",
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
        out.extend(build_doc(match.group("indent"), match.group("name"), signature))
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
