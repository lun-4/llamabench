import sys
import matplotlib.pyplot as plt
import pprint
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from bench import parse_system_info


actors_data = defaultdict(list)

BASEPATH = sys.argv[1]
MODE = sys.argv[2]
plt.rcParams["figure.figsize"] = (10, 6)


@dataclass
class MicroCSV:
    reader: any

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.reader)
        while line.startswith("#"):
            if line.startswith("# system info:"):
                sysline = line[len("# system info :") :].rstrip(" ")
                return ["system", sysline]
            line = next(self.reader)

        splitted = line.strip().split("\t")
        while splitted == [""]:
            line = next(self.reader)
            splitted = line.strip().split("\t")
        return splitted


data = defaultdict(lambda: defaultdict(list))
system_infos = {}

final_tsv = Path("./finaldata_donotcommit.tsv")
with final_tsv.open("w") as fd:
    pass  # wipe file


def print_final_tsv(line):
    print(line)
    with final_tsv.open("a") as fd:
        fd.write(line + "\n")


paths = list(Path(BASEPATH).glob("*.csv"))
pprint.pprint(paths)
for path in paths:
    with path.open("r") as fd:
        reader = MicroCSV(fd)
        sysinfo = next(reader)
        assert sysinfo[0] == "system"
        sysinfo[1] = parse_system_info(sysinfo[1])

        header = next(reader)
        if len(header) != 5:
            raise AssertionError(f"invalid header: {header}")

        for row in reader:
            if not row:
                continue
            if row[0] == "system":
                continue
            hostname, config, _, avg_tokens_per_second, stddev_tokens_per_second = row
            if hostname == "hostname":
                # this is another header lol ignore
                continue

            # ignore some hostnames
            if hostname in ("arctic-rose",):
                continue

            system_infos[hostname] = sysinfo[1]

            splitted_config = config.split(" ")

            if config.startswith("cpu (-t "):
                row_mode = "cpu"
                thread_count = int(splitted_config[2].rstrip(")"))
            elif "openblas" in config:
                row_mode = "openblas"
                thread_count = int(splitted_config[3].rstrip(")"))
            elif config.startswith("gpu (-t"):
                row_mode = "vulkan"
                thread_count = int(splitted_config[2].rstrip(","))
                ngl_count = int(splitted_config[4].rstrip(")"))
            elif "cuda" in config:
                row_mode = "cuda"
                thread_count = int(splitted_config[3].rstrip(","))
                ngl_count = int(splitted_config[5].rstrip(")"))

            avg_tokens_per_second = float(avg_tokens_per_second)
            stddev_tokens_per_second = float(stddev_tokens_per_second)
            if row_mode in ("vulkan", "cuda"):
                data[row_mode][hostname].append(
                    (
                        (thread_count, ngl_count),
                        avg_tokens_per_second,
                        stddev_tokens_per_second,
                    )
                )
                print_final_tsv(
                    f"{hostname}\t{config}\t{avg_tokens_per_second}\t{stddev_tokens_per_second}"
                )
            else:
                data[row_mode][hostname].append(
                    (thread_count, avg_tokens_per_second, stddev_tokens_per_second)
                )
                print_final_tsv(
                    f"{hostname}\t{config}\t{avg_tokens_per_second}\t{stddev_tokens_per_second}"
                )


pprint.pprint(data)
pprint.pprint(system_infos)


def subplot_with_filter(filter_fn, label_fn, title, filename, *, field: str = None):
    field = field or "cpu"
    style = "clean" if field == "cpu" else "openblas"

    fig, ax = plt.subplots()
    for actor, actor_data in data[field].items():
        sysinfo = system_infos[actor]
        plot = False
        if filter_fn(actor):
            plot = True

        if plot:
            x_values = [item[0] for item in actor_data]
            y_values = [item[1] for item in actor_data]
            ax.plot(x_values, y_values, label=label_fn(actor, sysinfo))

            y_stdev = [item[2] for item in actor_data]
            y_stdev_values_min = [y - y_stdev for y, y_stdev in zip(y_values, y_stdev)]
            y_stdev_values_max = [y + y_stdev for y, y_stdev in zip(y_values, y_stdev)]

            ax.fill_between(
                x_values,
                y_stdev_values_min,
                y_stdev_values_max,
                alpha=0.5,
                label=label_fn(actor, sysinfo),
            )
    ax.set_xlabel("thread count")
    ax.set_ylabel("tokens per second")
    ax.set_title(f"benchmark style={style} ({title})")
    fig.legend()
    fig.savefig(filename)


if MODE == "cpu":
    # plot all clean data
    for actor, actor_data in data["cpu"].items():
        x_values = [item[0] for item in actor_data]
        y_values = [item[1] for item in actor_data]
        plt.plot(x_values, y_values, label=actor)

    plt.xlabel("thread count")
    plt.ylabel("tokens per second")
    plt.title("benchmark style=clean")
    plt.legend()
    plt.savefig("llama3_clean.png")

    subplot_with_filter(
        lambda x: x in ("arctic-rose-v2", "apparition", "reticent-iris"),
        lambda actor, sysinfo: actor if sysinfo["AVX"] == "1" else f"{actor} (no avx)",
        "avx comparison",
        "llama3_avx_comparison.png",
    )
    subplot_with_filter(
        lambda x: x in ("switchblade", "chlorine", "steamdeck"),
        lambda actor, _: actor,
        "steam deck comparison",
        "llama3_deck_comparison.png",
    )
    subplot_with_filter(
        lambda x: x
        in ("ubuntu-8gb-hel1-1", "ubuntu-8gb-hel1-2", "scw-beautiful-hertz"),
        lambda actor, _: actor,
        "vps comparison",
        "llama3_vps_comparison.png",
    )


elif MODE == "openblas":
    # find the delta between a thread count made on 'cpu' mode vs 'openblas' mode
    # requires reformatting the data as dicts so its easy to find the delta
    clean_data = defaultdict(dict)
    openblas_data = defaultdict(dict)

    for actor, actor_data in data["openblas"].items():
        for x, y, stdev in actor_data:
            openblas_data[actor][x] = (y, stdev)

    for actor, actor_data in data["cpu"].items():
        for x, y, stdev in actor_data:
            clean_data[actor][x] = (y, stdev)

    for actor in openblas_data.keys():
        x_values = []
        y_values = []
        y_stdev_min = []
        y_stdev_max = []
        for threadcount in sorted(openblas_data[actor].keys()):
            delta = (
                openblas_data[actor][threadcount][0] - clean_data[actor][threadcount][0]
            )
            delta_stdev = max(
                openblas_data[actor][threadcount][1], clean_data[actor][threadcount][1]
            )

            x_values.append(threadcount)
            y_values.append(delta)
            y_stdev_min.append(max(delta - delta_stdev, 0.1))
            y_stdev_max.append(max(delta + delta_stdev, 0.1))
        plt.plot(x_values, y_values, label=actor)

    #        plt.fill_between(
    #            x_values,
    #            y_stdev_min,
    #            y_stdev_max,
    #            alpha=0.5,
    #            label=actor,
    #        )

    plt.xlabel("thread count")
    plt.ylabel("delta tokens per second")
    plt.title("speed gained or lost by going to openblas")
    plt.legend()
    plt.savefig("llama3_openblas_comparison.png")

    subplot_with_filter(
        lambda x: x in ("switchblade", "chlorine", "steamdeck"),
        lambda actor, _: actor,
        "steam deck comparison",
        "llama3_openblas_deck_comparison.png",
        field="openblas",
    )

    fig, ax = plt.subplots()
    for actor, actor_data in data["openblas"].items():
        x_values = [item[0] for item in actor_data]
        y_values = [item[1] for item in actor_data]
        ax.plot(x_values, y_values, label=actor)
    ax.set_xlabel("thread count")
    ax.set_ylabel("tokens per second")
    ax.set_title("benchmark style=openblas")
    fig.legend()
    fig.savefig("llama3_openblas.png")

elif MODE == "vulkan":
    ngl_sets = []
    for actor, actor_data in data["vulkan"].items():
        for row in actor_data:
            if row[0][1] not in ngl_sets:
                ngl_sets.append(row[0][1])
        break

    for ngl in ngl_sets:
        fig, ax = plt.subplots()

        for actor, actor_data in data["vulkan"].items():
            x_values = []
            y_values = []
            for row in actor_data:
                row_ngl = row[0][1]
                if row_ngl != ngl:
                    continue
                thread_count = row[0][0]
                tokens_sec = row[1]
                x_values.append(thread_count)
                y_values.append(tokens_sec)

            ax.plot(x_values, y_values, label=actor + " (vulkan)", linestyle="dotted")

        for actor, actor_data in data["cuda"].items():
            x_values = []
            y_values = []
            for row in actor_data:
                row_ngl = row[0][1]
                if row_ngl != ngl:
                    continue
                thread_count = row[0][0]
                tokens_sec = row[1]
                x_values.append(thread_count)
                y_values.append(tokens_sec)

            ax.plot(x_values, y_values, label=actor + " (cuda)")

        ax.set_xlabel("thread count")
        ax.set_ylabel("tokens/sec")
        ax.set_title(f"-ngl {ngl}")
        fig.legend()
        fig.savefig(f"llama3_ngl_{ngl}.png")

elif MODE == "cuda":

    ngl_sets = []
    for actor, actor_data in data["cuda"].items():
        for row in actor_data:
            if row[0][1] not in ngl_sets:
                ngl_sets.append(row[0][1])
        break

    for ngl in ngl_sets:
        fig, ax = plt.subplots()

        for actor, actor_data in data["cuda"].items():
            x_values = []
            y_values = []
            for row in actor_data:
                row_ngl = row[0][1]
                if row_ngl != ngl:
                    continue
                thread_count = row[0][0]
                tokens_sec = row[1]
                x_values.append(thread_count)
                y_values.append(tokens_sec)

            ax.plot(x_values, y_values, label=actor)
        ax.set_xlabel("thread count")
        ax.set_ylabel("tokens/sec")
        ax.set_title(f"cuda -ngl {ngl}")
        fig.legend()

else:
    raise Exception("TODO")


# Add legend
## Display the plot
plt.show()
