import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist_binary"
SETUP_FILE = ROOT / "setup_binary.py"
ARCH_ALIASES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}


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


def normalize_arch(name: str) -> str:
    value = ARCH_ALIASES.get(str(name).strip().lower())
    if value is None:
        supported = ", ".join(sorted(ARCH_ALIASES))
        raise SystemExit(f"Unsupported target arch: {name!r}. Supported values: {supported}")
    return value


def get_host_arch() -> str:
    return normalize_arch(platform.machine())


def ensure_buildable_target(target_arch: str):
    host_arch = get_host_arch()
    if target_arch != host_arch:
        raise SystemExit(
            "Cross-architecture compilation is not handled by this script directly. "
            f"Host arch is {host_arch}, requested target arch is {target_arch}. "
            "Use a matching native environment or an arm64/x86_64 container/toolchain first, "
            "then run this script inside that target environment."
        )


def collect_binaries(target_arch: str):
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    target_engine = DIST_DIR / target_arch / "engine"
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
        "--target-arch",
        default=get_host_arch(),
        help="target CPU architecture: x86_64, arm64, or aarch64 (arm64 is normalized to aarch64)",
    )
    parser.add_argument(
        "--keep-generated-c",
        action="store_true",
        help="keep generated C files after build",
    )
    args = parser.parse_args()
    target_arch = normalize_arch(args.target_arch)
    ensure_buildable_target(target_arch)

    if args.clean:
        clean_artifacts()

    run_build()
    copied = collect_binaries(target_arch)
    cleanup_engine_binaries()

    if not args.keep_generated_c:
        for c_file in (ROOT / "engine").glob("*.c"):
            remove_path(c_file)

    print("Build finished.")
    print(f"Target arch: {target_arch}")
    print("Binary output:")
    for item in copied:
        print(item)


if __name__ == "__main__":
    main()
