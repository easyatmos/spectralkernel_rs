<img src="https://github.com/easyatmos/spectralkernel_rs/blob/main/readme/spherepack_docs_style.png?raw=true" alt="spectralkernel_rs">

<h2 align="center">spectralkernel_rs: Python バインディング付き Rust 製大気スペクトル変換カーネル</h2>

<p align="center">
<a href="https://easyatmos.github.io/spectralkernel_rs/"><strong>ドキュメント</strong>（最新版）</a>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/spectralkernel-rs)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spectralkernel-rs)
![Crates.io Version](https://img.shields.io/crates/v/spectralkernel_rs)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

<div align="center">
<center><a href="../README.md">English</a> / <a href="README.zh_CN.md">简体中文</a> / 日本語</center>
</div>

## spectralkernel_rs とは？

**spectralkernel_rs** は、大気科学向けの球面スペクトル変換カーネルをRust で実装し、[PyO3](https://pyo3.rs/) と [maturin](https://www.maturin.rs/) によって Python バインディングを提供します。

このプロジェクトは、再利用可能な低レベルカーネルと高レベルの球面演算子に焦点を当てており、次のような機能を必要とするワークフローに適しています。

- スカラーおよびベクトル球面調和解析/合成、
- 勾配、発散、渦度、ラプラシアン、および逆演算子、
- 流線関数/速度ポテンシャル変換、
- Helmholtz 型のベクトル分解、
- 浅水方程式および移流項の補助的な tendency、
- Python から利用できる NumPy 配列インターフェース。

Rust コアは性能が重要なスペクトル計算向けに設計されており、Python パッケージは科学計算ワークフロー向けにコンパクトな API を公開します。

## インストール

PyPI から Python パッケージをインストールします。

```bash
pip install spectralkernel-rs
```

インポート名は次のとおりです。

```python
import spectralkernel_rs as sk
```

## クイックスタート

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

## ビルド手順

## 要件

- Python >= 3.10
- NumPy >= 1.24
- xarray >= 2023.1.0
- ローカルビルド用の Rust stable ツールチェーン
- Python 拡張ビルド用の maturin
- 付属の複数バージョン wheel ビルドスクリプト用の uv
- manylinux wheel ビルドスクリプト用の Docker

### ローカル開発

プロジェクトルートで実行します。

```bash
python -m pip install --upgrade pip maturin pytest
python -m maturin develop --release
python -m pytest -q tests/test_sphere_ops.py
```

### Windows wheels

Rust と uv をインストールしてから、次を実行します。

```powershell
.\scripts\build_manywindows_wheel.ps1
```

生成された wheel は `dist/` に書き込まれます。

### Linux wheels

Docker をインストールし、Linux ホスト上で次を実行します。

```bash
bash ./scripts/build_manylinux_wheel.sh
```

生成された wheel は `dist/` に書き込まれます。

### macOS wheels

Rust と uv をインストールしてから、macOS 上で次を実行します。

```bash
bash ./scripts/build_macos_wheel.sh
```

生成された wheel は `dist/` に書き込まれます。

## テスト

CI で使われる重点的な Python テストを実行します。

```bash
python -m pytest -q tests/test_sphere_ops.py
```

補助スクリプトは、現在の shell で有効な `python` を使用します。

```powershell
.\scripts\run_test.ps1
```

```bash
./scripts/run_test.sh
```

Rust のフォーマット確認とチェック：

```bash
cargo fmt --all -- --check
cargo check --all-targets
```

## Python API 概要

高レベル Python API は `SphereOps` から始まります。

```python
ops = sk.SphereOps.regular(nlat, nlon, radius=6.3712e6)
ops = sk.SphereOps.gaussian(nlat, nlon, radius=6.3712e6)
```

主な操作は次のとおりです。

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

モジュールレベルの方程式ヘルパーには次が含まれます。

- `scalar_advection_rhs`
- `barotropic_vorticity_rhs`
- `shallow_water_rhs`

## Rust crate

crate は次の名前で公開されています。

```toml
[dependencies]
spectralkernel_rs = "1"
```

現在の Rust ライブラリは主に Python 拡張を支えており、
Python パッケージで使われているものと同じスペクトルカーネル実装を公開します。

## 参考

- Adams, J. C., & Swarztrauber, P. N. (1999). SPHEREPACK 3.0: A Model Development Facility. *Monthly Weather Review*, 127(8), 1872-1878. https://doi.org/10.1175/1520-0493(1999)127[1872:SAMDF](1872:SAMDF)2.0.CO;2
- https://github.com/NCAR/NCAR-Classic-Libraries-for-Geophysics
- https://github.com/jlokimlin/spherepack

## ライセンス

このプロジェクトは BSD 3-Clause License の下で配布されています。
詳細は [LICENSE](../LICENSE) を参照してください。
