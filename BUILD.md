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

### 3. Output location

Compiled binaries are collected into:

`./dist_binary/engine`

The temporary `engine/*.so` or `engine/*.pyd` files generated during the build are deleted automatically after collection.

Typical result:

- Linux: `.so`
- Windows: `.pyd`

### 4. Runtime usage

Keep these files together:

- `main.py`
- `dist_binary/engine/__init__.py`
- compiled files under `dist_binary/engine/`

If you want to run from the binary output directory, copy `main.py` beside `dist_binary` and adjust import path as needed, or package them together in your deployment layout.
