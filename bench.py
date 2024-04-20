import os
import re
import sys
import logging
import shutil
import shlex
import subprocess
from decimal import Decimal
from pathlib import Path

SAMPLE_TIME_REGEX = re.compile(
    r"sample time =\s+(\d+\.\d+)\s+ms \/.*?(\d+\.\d+)\s+ms per token"
)
PROMPT_EVAL_TIME_REGEX = re.compile(
    r"prompt eval time =\s+(\d+\.\d+)\s+ms \/.*?(\d+\.\d+)\s+ms per token"
)
EVAL_TIME_REGEX = re.compile(
    r"eval time =\s+(\d+\.\d+)\s+ms \/.*?(\d+\.\d+)\s+ms per token"
)


def check_output(cmd, *args, **kwargs):
    cwd = kwargs["cwd"]
    log.info("running: %r", f"cd {cwd} && {shlex.join(cmd)}")
    return subprocess.check_output(cmd, *args, **kwargs, stderr=subprocess.STDOUT)


log = logging.getLogger(__name__)


def build(llamacpp_path: Path, makeflags: str, style) -> None:
    output_path = Path.cwd() / f"main-{style}"
    if output_path.exists():
        log.info("main output file exists, skipping build (style=%r)", style)
        return

    log.info("building with style %r", style)

    files = list(llamacpp_path.glob("bench-mark-*"))
    if len(files) > 0:
        log.warning("multiple mark files, ignoring them all")
    build = True

    if len(files) == 1:
        existing_mark_file = files[0]
        if existing_mark_file.stem == "bench-mark-" + style:
            log.info("mark file exists, skipping build")
            build = False

    # tfw subprocess doesnt support Path as cwd input
    if build:
        llamacpp_path = str(llamacpp_path)
        check_output(["make", "clean"], cwd=llamacpp_path)
        if style == "clean":
            check_output(["make"] + shlex.split(makeflags), cwd=llamacpp_path)
        else:
            raise AssertionError("unknown build style " + style)
    # now that its built, mark it
    llamacpp_path = Path(llamacpp_path)
    mark_file = llamacpp_path / f"bench-mark-{style}.o"
    mark_file.touch(exist_ok=True)

    shutil.copy(llamacpp_path / "main", output_path)


BASEPROMPT = "Write a paragraph about the hit game Among Us.\n\n"


def bench(style, model_path):
    output_path = Path.cwd() / f"main-{style}"
    if not output_path.exists():
        raise AssertionError("main file not found")

    if style == "clean":
        maxthreads = int(os.environ.get("MAXTHREADS", "16")) + 1
        log.info("running from 1 to %d threads ", maxthreads - 1)

        print(
            "threadcount,sample_ms_per_token,prompt_eval_ms_per_token,eval_ms_per_token,tokens_per_second"
        )
        for thread_count in range(1, maxthreads):
            sample_ms_per_token, prompt_eval_ms_per_token, eval_ms_per_token = (
                run_model(output_path, model_path, ["-t", str(thread_count)])
            )
            tokens_sec = round(Decimal(1000) / eval_ms_per_token, 2)
            print(
                f"{thread_count},{sample_ms_per_token},{prompt_eval_ms_per_token},{eval_ms_per_token},{tokens_sec}"
            )


def run_model(output_path, model_path, llamacpp_args):
    out = check_output(
        [
            str(output_path),
            "-m",
            str(model_path),
            "--top_k",
            "10000",
            "--temp",
            "0.4",
            "--repeat-penalty",
            "1",
            "-n",
            "12",  # use 128 lol
            "-p",
            BASEPROMPT,
        ]
        + llamacpp_args,
        cwd=Path.cwd(),
    )
    text = out.decode()
    sample_time_match = re.search(SAMPLE_TIME_REGEX, text)
    sample_ms_per_token = Decimal(sample_time_match.group(2))
    prompt_eval_time_match = re.search(PROMPT_EVAL_TIME_REGEX, text)
    prompt_eval_ms_per_token = Decimal(prompt_eval_time_match.group(2))
    eval_time_match = re.search(EVAL_TIME_REGEX, text)
    eval_ms_per_token = Decimal(eval_time_match.group(2))
    return sample_ms_per_token, prompt_eval_ms_per_token, eval_ms_per_token


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

    log.info("LLAMACPP=%s", llamacpp_path)
    log.info("MODEL=%s", model_path)
    log.info("MAKEFLAGS=%s", makeflags)

    # build normally
    build(llamacpp_path, makeflags, "clean")
    bench("clean", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
