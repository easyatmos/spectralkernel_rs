<p align="center">
  <img src="https://github.com/easyatmos/spectralkernel_rs/blob/main/readme/spherepack_docs_style_cropped.png?raw=true" alt="spectralkernel_rs" width="200">
</p>

<h2 align="center">Rust atmosphere spectral transform kernels with Python bindings</h2>

<p align="center">
<a href="https://easyatmos.github.io/spectralkernel_rs/"><strong>Documentation</strong> (latest)</a>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/spectralkernel-rs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spectralkernel-rs)
![Crates.io Version](https://img.shields.io/crates/v/spectralkernel_rs)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

<div align="center">
<center>English / <a href = "readme/README.zh_CN.md">简体中文</a> / <a href = "readme/README.ja_JP.md">日本語</a></center>
</div>

## What is spectralkernel_rs?

**spectralkernel_rs** provides Rust implementations of atmosphere-oriented
spherical spectral transform kernels, with Python bindings built through
[PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/).

The project focuses on reusable low-level kernels and high-level spherical
operators for workflows that need:

- scalar and vector spherical harmonic analysis/synthesis,
- gradients, divergence, vorticity, Laplacians, and inverse operators,
- streamfunction / velocity-potential transforms,
- Helmholtz-style vector decomposition,
- shallow-water and advection helper tendencies,
- Python-accessible NumPy array interfaces.

The Rust core is designed for performance-sensitive spectral computations while
the Python package exposes a compact API for scientific workflows.

## Installation

Install the Python package from PyPI:

```bash
pip install spectralkernel-rs
```

The import name is:

```python
import spectralkernel_rs as sk
```

## Quick start

```python
import numpy as np
import spectralkernel_rs as sk

ops = sk.SphereOps.gaussian(9, 16, radius=1.0)

lat = np.deg2rad(ops.lat())
lon = np.deg2rad(ops.lon())
field = np.cos(lat)[:, None] * np.cos(lon)[None, :]

grad_u, grad_v = ops.grad(field)
lap = ops.laplacian(field)
recovered = ops.inverse_laplacian(lap)

print(grad_u.shape, grad_v.shape, recovered.shape)
```

## Build instructions

### Requirements

- Python >= 3.10
- NumPy >= 1.24
- xarray >= 2023.1.0
- Rust stable toolchain for local builds
- maturin for Python extension builds
- uv for the provided multi-version wheel build scripts
- Docker for the manylinux wheel build script

### Develop locally

From the project root:

```bash
python -m pip install --upgrade pip maturin pytest
python -m maturin develop --release
python -m pytest -q tests/test_sphere_ops.py
```

### Windows wheels

Install Rust and uv, then run:

```powershell
.\scripts\build_manywindows_wheel.ps1
```

Generated wheels are written to `dist/`.

### Linux wheels

Install Docker, then run on a Linux host:

```bash
bash ./scripts/build_manylinux_wheel.sh
```

Generated wheels are written to `dist/`.

### macOS wheels

Install Rust and uv, then run on macOS:

```bash
bash ./scripts/build_macos_wheel.sh
```

Generated wheels are written to `dist/`.

## Tests

Run the focused Python test used by CI:

```bash
python -m pytest -q tests/test_sphere_ops.py
```

The helper scripts use the active `python` from your shell:

```powershell
.\scripts\run_test.ps1
```

```bash
./scripts/run_test.sh
```

Rust formatting and checks:

```bash
cargo fmt --all -- --check
cargo check --all-targets
```

## Python API overview

The high-level Python API starts from `SphereOps`:

```python
ops = sk.SphereOps.regular(nlat, nlon, radius=6.3712e6)
ops = sk.SphereOps.gaussian(nlat, nlon, radius=6.3712e6)
```

Common operations include:

- `scalar_to_spec`, `spec_to_scalar`
- `vector_to_spec`, `spec_to_vector`
- `grad`, `div`, `vort`
- `laplacian`, `inverse_laplacian`
- `streamfunction_to_wind`
- `velocity_potential_to_wind`
- `streamfunction_velocity_potential`
- `helmholtz_decompose`
- `spectral_filter`
- `hyperdiffusion`, `vector_hyperdiffusion`

Module-level equation helpers include:

- `scalar_advection_rhs`
- `barotropic_vorticity_rhs`
- `shallow_water_rhs`

## Rust crate

The crate is published as:

```toml
[dependencies]
spectralkernel_rs = "1"
```

The current Rust library primarily backs the Python extension and exposes the
same spectral kernel implementation used by the Python package.

## Reference

- Adams, J. C., & Swarztrauber, P. N. (1999). SPHEREPACK 3.0: A Model Development Facility. *Monthly Weather Review*, 127(8), 1872-1878. https://doi.org/10.1175/1520-0493(1999)127[1872:SAMDF](1872:SAMDF)2.0.CO;2
- https://github.com/NCAR/NCAR-Classic-Libraries-for-Geophysics
- https://github.com/jlokimlin/spherepack

## License

This project is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for details.
