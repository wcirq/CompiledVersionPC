## Build Engine Binary

This project keeps `main.py` as plain Python and compiles only the `engine/` package.

### 1. Install dependencies

```bash
pip install cython setuptools
```

If your environment already has a C/C++ toolchain and Python headers, this is enough.

### 2. Run one-click build

```bash
cd "$(dirname "$0")"
python build_engine.py --clean
```

You can also specify the target architecture explicitly:

```bash
python build_engine.py --clean --target-arch x86_64
python build_engine.py --clean --target-arch arm64
python build_engine.py --clean --target-arch aarch64
```

`arm64` is normalized to `aarch64`.

Note: this script does not perform true cross-compilation by itself. If the requested target architecture does not match the current build environment, the build exits with a clear error. Use a matching native environment or run the build inside a target-architecture container first.

### 3. Output location

Compiled binaries are collected into:

`./dist_binary/<target-arch>/engine`

The temporary `engine/*.so` or `engine/*.pyd` files generated during the build are deleted automatically after collection.

Typical result:

- Linux: `.so`
- Windows: `.pyd`

### 4. Runtime usage

Keep these files together:

- `main.py`
- `dist_binary/<target-arch>/engine/__init__.py`
- compiled files under `dist_binary/<target-arch>/engine/`

If you want to run from the binary output directory, copy `main.py` beside `dist_binary` and adjust import path as needed, or package them together in your deployment layout.
