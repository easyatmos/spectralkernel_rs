<img src="https://github.com/easyatmos/spectralkernel_rs/blob/main/readme/spherepack_docs_style_cropped.png?raw=true" alt="spectralkernel_rs">

<h2 align="center">Rust 编写附带 Python 接口的大气谱变换内核</h2>

<p align="center">
<a href="https://easyatmos.github.io/spectralkernel_rs/"><strong>文档</strong>（最新版）</a>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/spectralkernel-rs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spectralkernel-rs)
![Crates.io Version](https://img.shields.io/crates/v/spectralkernel_rs)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

<div align="center">
<center><a href="../README.md">English</a> / 简体中文 / <a href="README.ja_JP.md">日本語</a></center>
</div>

## spectralkernel_rs 是什么？

**spectralkernel_rs** 提供面向大气领域的球面谱变换核的 Rust 实现，并通过 [PyO3](https://pyo3.rs/) 和 [maturin](https://www.maturin.rs/) 构建了 Python 绑定。

该项目专注于可复用的底层计算核心和高层球面算子，适用于需要以下功能的工作流：

* 标量和矢量球谐分析与合成
* 梯度、散度、涡度、拉普拉斯算子及其逆算子
* 流函数/速度势变换
* 亥姆霍兹式矢量分解
* 浅水方程与平流辅助倾向项
* Python可访问的NumPy数组接口

Rust核心专为性能敏感的谱计算而设计，Python包则为科学工作流提供了简洁的API。

## 安装

从 PyPI 安装 Python 包：

```bash
pip install spectralkernel-rs
```

Python 可通过如下方式导入：

```python
import spectralkernel_rs as sk
```

## 快速开始

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

## 构建说明

## 依赖要求

- Python >= 3.10
- NumPy >= 1.24
- xarray >= 2023.1.0
- 用于本地构建的 Rust stable 工具链
- 用于构建 Python 扩展的 maturin
- 用于项目提供的多版本 wheel 构建脚本的 uv
- 用于 manylinux wheel 构建脚本的 Docker

### 本地开发

在项目根目录运行：

```bash
python -m pip install --upgrade pip maturin pytest
python -m maturin develop --release
python -m pytest -q tests/test_sphere_ops.py
```

### Windows wheels

安装 Rust 和 uv，然后运行：

```powershell
.\scripts\build_manywindows_wheel.ps1
```

生成的 wheel 文件会写入 `dist/`。

### Linux wheels

安装 Docker，然后在 Linux 主机上运行：

```bash
bash ./scripts/build_manylinux_wheel.sh
```

生成的 wheel 文件会写入 `dist/`。

### macOS wheels

安装 Rust 和 uv，然后在 macOS 上运行：

```bash
bash ./scripts/build_macos_wheel.sh
```

生成的 wheel 文件会写入 `dist/`。

## 测试

运行 CI 使用的重点 Python 测试：

```bash
python -m pytest -q tests/test_sphere_ops.py
```

辅助脚本会使用当前 shell 中激活的 `python`：

```powershell
.\scripts\run_test.ps1
```

```bash
./scripts/run_test.sh
```

Rust 格式检查和编译检查：

```bash
cargo fmt --all -- --check
cargo check --all-targets
```

## Python API 概览

高层 Python API 从 `SphereOps` 开始：

```python
ops = sk.SphereOps.regular(nlat, nlon, radius=6.3712e6)
ops = sk.SphereOps.gaussian(nlat, nlon, radius=6.3712e6)
```

常用操作包括：

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

模块级方程辅助函数包括：

- `scalar_advection_rhs`
- `barotropic_vorticity_rhs`
- `shallow_water_rhs`

## Rust crate

crate 发布名为：

```toml
[dependencies]
spectralkernel_rs = "1"
```

当前 Rust 库主要支撑 Python 扩展，并暴露与 Python 包相同的谱核心实现。

## 参考资料

- Adams, J. C., & Swarztrauber, P. N. (1999). SPHEREPACK 3.0: A Model Development Facility. *Monthly Weather Review*, 127(8), 1872-1878. https://doi.org/10.1175/1520-0493(1999)127[1872:SAMDF](1872:SAMDF)2.0.CO;2
- https://github.com/NCAR/NCAR-Classic-Libraries-for-Geophysics
- https://github.com/jlokimlin/spherepack

## 许可证

本项目基于 BSD 3-Clause License 发布。详情请参见
[LICENSE](../LICENSE)。
