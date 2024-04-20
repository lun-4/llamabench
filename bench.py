import os
import re
import sys
import logging
import shutil
import shlex
import subprocess
from dataclasses import dataclass
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
        elif style == "openblas":
            check_output(
                ["make", "LLAMA_OPENBLAS=1"] + shlex.split(makeflags), cwd=llamacpp_path
            )
        elif style == "vulkan":
            check_output(
                ["make", "LLAMA_VULKAN=1"] + shlex.split(makeflags), cwd=llamacpp_path
            )
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
        raise AssertionError(f"main file for style {style} not found")

    print(f" === BENCHMARKING IN STYLE {style} ===")

    system_info = None
    log.info("running probe...")
    timings, raw_info, given_system_info = run_model(
        output_path, model_path, ["-t", "1"], tokens=1, vulkan=style == "vulkan"
    )

    system_info = given_system_info
    print("# system info:", raw_info)
    n_threads = system_info["n_threads"]
    _, total_threads = n_threads.split("/")
    total_threads = int(total_threads.strip())

    blas = system_info["BLAS"]
    if style == "clean" and blas != "0":
        raise AssertionError(f"clean bench wanted BLAS=0, got {blas}")
    elif style in ("openblas", "mkl") and blas != "1":
        raise AssertionError(f"openblas/mkl bench wanted BLAS=1, got {blas}")

    print(
        "config,sample_ms_per_token,prompt_eval_ms_per_token,eval_ms_per_token,tokens_per_second"
    )

    maxthreads = int(os.environ.get("MAXTHREADS", "16")) + 1
    if maxthreads > total_threads:
        log.warning(
            "system has %d threads but MAXTHREADS is %d, likely decrease it...",
            total_threads,
            maxthreads - 1,
        )

    if style in ("clean", "openblas", "mkl"):
        log.info("running from 1 to %d threads only", maxthreads - 1)
        for thread_count in range(1, maxthreads):
            timings, raw_info, given_system_info = run_model(
                output_path, model_path, ["-t", str(thread_count)]
            )

            modifier = ""
            if style in ("openblas", "mkl"):
                modifier = f"{style}, "

            tokens_sec = round(Decimal(1000) / timings.eval_ms_per_token, 2)
            print(
                f"cpu ({modifier}-t {thread_count}),{timings.sample_ms_per_token},{timings.prompt_eval_ms_per_token},{timings.eval_ms_per_token},{tokens_sec}"
            )
    elif style == "vulkan":
        gpulayers = system_info["GPULAYERS"]
        _, max_gpu_layers = gpulayers.split("/")
        max_gpu_layers = int(max_gpu_layers.strip())

        log.info("max gpu layers: %d", max_gpu_layers)

        try:
            gpu_layer_step_count = int(os.environ["GPU_LAYER_STEP_COUNT"])
        except KeyError:
            gpu_layer_step_count = 4
            log.warning(
                "GPU_LAYER_STEP_COUNT not set, defaulting to %d", gpu_layer_step_count
            )

        log.info(
            "vulkan bench, running from -ngl 0 to -ngl %d (with step count = %d)",
            max_gpu_layers,
            gpu_layer_step_count,
        )
        for gpu_layers in range(0, max_gpu_layers, gpu_layer_step_count):
            log.info(
                "vulkan bench, running from -t 1 to -t %d for -ngl %d",
                maxthreads,
                gpu_layers,
            )

            for thread_count in range(1, maxthreads):
                timings, raw_info, given_system_info = run_model(
                    output_path,
                    model_path,
                    ["-t", str(thread_count), "-ngl", str(gpu_layers)],
                )

                tokens_sec = round(Decimal(1000) / timings.eval_ms_per_token, 2)
                print(
                    f"gpu (-t {thread_count}, -ngl {gpu_layers}),{timings.sample_ms_per_token},{timings.prompt_eval_ms_per_token},{timings.eval_ms_per_token},{tokens_sec}"
                )

    else:
        raise AssertionError(f"invalid style for bench: {style}")


@dataclass
class Timings:
    sample_ms_per_token: Decimal
    prompt_eval_ms_per_token: Decimal
    eval_ms_per_token: Decimal


def parse_system_info(raw: str) -> dict:
    info = {}
    for entry in raw.split("|"):
        if not entry.strip():
            continue
        key, value = entry.split("=")
        key = key.strip()
        value = value.strip()
        info[key] = value
    return info


def run_model(output_path, model_path, llamacpp_args, *, tokens=None, vulkan=False):
    tokens = tokens or "12"  # when ready, change to 128
    tokens = str(tokens)
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
            tokens,
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
    system_info = re.search(r"system_info: (.*)", text).group(1)
    if vulkan:
        system_info += " | VULKAN0 = " + (
            re.search(r"Vulkan0: (.*)", text).group(1).replace("|", ",")
        )
        system_info += " | GPULAYERS = " + re.search(
            r"llm_load_tensors: offloaded (.*) layers to GPU", text
        ).group(1)
    return (
        Timings(sample_ms_per_token, prompt_eval_ms_per_token, eval_ms_per_token),
        system_info,
        parse_system_info(system_info),
    )


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

    build(llamacpp_path, makeflags, "clean")
    bench("clean", model_path)
    build(llamacpp_path, makeflags, "openblas")
    bench("openblas", model_path)
    if can_vulkan:
        build(llamacpp_path, makeflags, "vulkan")
        bench("vulkan", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
