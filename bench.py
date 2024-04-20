import os
import sys
import logging
import shutil
import shlex
import subprocess
from pathlib import Path


def check_output(cmd, *args, **kwargs):
    cwd = kwargs["cwd"]
    print(f"cd {cwd} && {shlex.join(cmd)}")
    return subprocess.check_output(
        cmd,
        *args,
        **kwargs,
        stderr=subprocess.PIPE,
    )


log = logging.getLogger(__name__)


def build(llamacpp_path: Path, makeflags: str, style) -> None:
    log.info("building with style %r", style)

    files = list(llamacpp_path.glob("bench-mark-*"))
    if len(files) > 0:
        log.warning("multiple mark files, ignoring them all")

    if len(files) == 1:
        existing_mark_file = files[0]
        if existing_mark_file.stem == "bench-mark-" + style:
            log.info("mark file exists, skipping build")
            return

    # tfw subprocess doesnt support Path as cwd input
    llamacpp_path = str(llamacpp_path)
    check_output(["make", "clean"], cwd=llamacpp_path)
    if style == "clean":
        check_output(["make"] + shlex.split(makeflags), cwd=llamacpp_path)
    else:
        raise AssertionError("unknown build style " + style)
    # now that its built, mark it
    mark_file = Path(llamacpp_path) / f"bench-mark-{style}.o"
    mark_file.touch(exist_ok=True)


def main():
    if not shutil.which("make"):
        raise Exception("requires make")

    if not shutil.which("gcc"):
        raise Exception("requires gcc")

    makeflags = os.environ.get("MAKEFLAGS", "")
    try:
        llamacpp_path = Path(os.environ["LLAMACPP"])
    except KeyError:
        raise Exception("LLAMACPP env var must be set. must set to the DIRECTORY")

    if not llamacpp_path.is_dir():
        raise Exception(
            "LLAMACPP is not set to a directory. must be llama.cpp source dir"
        )

    try:
        model_path = Path(os.environ["MODEL"])
    except KeyError:
        raise Exception(
            "MODEL env var must be set. must be the model file to use in the benchmark"
        )

    if not model_path.is_file():
        raise Exception("MODEL is not set to a file")

    can_vulkan = os.environ.get("VULKAN", "0") == "1"

    if can_vulkan:
        # verify header
        if not Path("/usr/include/vulkan/vulkan.h").exists():
            raise Exception("VULKAN=1 but vulkan.h not found in /usr/include/vulkan")
        if not shutil.which("vulkaninfo"):
            raise Exception("VULKAN=1 but vulkaninfo not found")

    print("LLAMACPP=", llamacpp_path)
    print("MODEL=", model_path)
    print("MAKEFLAGS", makeflags)

    # build normally
    build(llamacpp_path, makeflags, "clean")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
