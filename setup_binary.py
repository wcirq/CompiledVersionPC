from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit(
        "Cython is required. Install it first, for example: pip install cython"
    ) from exc


ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "engine"


def build_extensions():
    module_files = [
        ENGINE_DIR / "augment.py",
        ENGINE_DIR / "backbone.py",
        ENGINE_DIR / "indexing.py",
        ENGINE_DIR / "indexing_bm.py",
        ENGINE_DIR / "runtime.py",
        ENGINE_DIR / "utils.py",
    ]

    extensions = []
    for file_path in module_files:
        module_name = ".".join(file_path.relative_to(ROOT).with_suffix("").parts)
        extensions.append(
            Extension(
                name=module_name,
                sources=[str(file_path)],
            )
        )
    return extensions


setup(
    name="vision-memory-binary",
    ext_modules=cythonize(
        build_extensions(),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=False,
    ),
    zip_safe=False,
)
