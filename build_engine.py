import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist_binary"
SETUP_FILE = ROOT / "setup_binary.py"


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_artifacts():
    remove_path(BUILD_DIR)
    remove_path(DIST_DIR)

    for c_file in ROOT.rglob("*.c"):
        if c_file.parent == ROOT / "engine":
            remove_path(c_file)

    for ext in ROOT.rglob("*.so"):
        remove_path(ext)
    for ext in ROOT.rglob("*.pyd"):
        remove_path(ext)


def cleanup_engine_binaries():
    for pattern in ("*.so", "*.pyd"):
        for binary in (ROOT / "engine").glob(pattern):
            remove_path(binary)


def collect_binaries():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    target_engine = DIST_DIR / "engine"
    target_engine.mkdir(parents=True, exist_ok=True)

    copied = []
    for pattern in ("*.so", "*.pyd"):
        for binary in (ROOT / "engine").glob(pattern):
            dst = target_engine / binary.name
            shutil.copy2(binary, dst)
            copied.append(dst)

    init_src = ROOT / "engine" / "__init__.py"
    init_dst = target_engine / "__init__.py"
    shutil.copy2(init_src, init_dst)
    copied.append(init_dst)

    return copied


def run_build():
    cmd = [
        sys.executable,
        str(SETUP_FILE),
        "build_ext",
        "--inplace",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build binary extensions for engine package only.")
    parser.add_argument("--clean", action="store_true", help="remove old build artifacts before building")
    parser.add_argument(
        "--keep-generated-c",
        action="store_true",
        help="keep generated C files after build",
    )
    args = parser.parse_args()

    if args.clean:
        clean_artifacts()

    run_build()
    copied = collect_binaries()
    cleanup_engine_binaries()

    if not args.keep_generated_c:
        for c_file in (ROOT / "engine").glob("*.c"):
            remove_path(c_file)

    print("Build finished.")
    print("Binary output:")
    for item in copied:
        print(item)


if __name__ == "__main__":
    main()
