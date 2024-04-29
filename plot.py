import sys
import matplotlib.pyplot as plt
import pprint
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


actors_data = defaultdict(list)

BASEPATH = sys.argv[1]
MODE = sys.argv[2]


@dataclass
class CommentIgnorer:
    reader: any

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.reader)
        while line.startswith("#"):
            line = next(self.reader)
        return line


data = defaultdict(lambda: defaultdict(list))

for path in Path(BASEPATH).glob("*.csv"):
    with path.open("r") as fd:
        reader = csv.reader(CommentIgnorer(fd), delimiter="\t")
        header = next(reader)

        if len(header) != 5:
            raise AssertionError(f"invalid header: {header}")

        for row in reader:
            if not row:
                continue
            # print(row)
            hostname, config, _, avg_tokens_per_second, stddev_tokens_per_second = row
            if hostname == "hostname":
                # this is another header lol ignore
                continue

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
                ok = True
                # manual filtering lol
                # ignore elpis' data for now
                if hostname == "elpis":
                    ok = False
                    # if thread_count in (1, 6, 11, 12) and ngl_count in (0, 12, 20, 32):
                    #    ok = True

                if ok:
                    data[row_mode][hostname].append(
                        ((thread_count, ngl_count), avg_tokens_per_second)
                    )
                    print(
                        f"| {hostname} | {config} | {avg_tokens_per_second} | {stddev_tokens_per_second} |"
                    )

            else:
                data[row_mode][hostname].append((thread_count, avg_tokens_per_second))
                print(
                    f"| {hostname} | {config} | {avg_tokens_per_second} | {stddev_tokens_per_second} |"
                )


pprint.pprint(data)

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
elif MODE == "openblas":
    # find the delta between a thread count made on 'cpu' mode vs 'openblas' mode
    # requires reformatting the data as dicts so its easy to find the delta
    clean_data = defaultdict(dict)
    openblas_data = defaultdict(dict)

    for actor, actor_data in data["openblas"].items():
        for x, y in actor_data:
            openblas_data[actor][x] = y

    for actor, actor_data in data["cpu"].items():
        for x, y in actor_data:
            clean_data[actor][x] = y

    for actor in openblas_data.keys():
        x_values = []
        y_values = []
        for threadcount in sorted(openblas_data[actor].keys()):
            delta = openblas_data[actor][threadcount] - clean_data[actor][threadcount]
            x_values.append(threadcount)
            y_values.append(delta)
        plt.plot(x_values, y_values, label=actor)
    plt.xlabel("thread count")
    plt.ylabel("delta tokens per second")
    plt.title("speed gained or lost by going to openblas")
    plt.legend()

    fig, ax = plt.subplots()
    for actor, actor_data in data["openblas"].items():
        x_values = [item[0] for item in actor_data]
        y_values = [item[1] for item in actor_data]
        ax.plot(x_values, y_values, label=actor)
    ax.set_xlabel("thread count")
    ax.set_ylabel("tokens per second")
    ax.set_title("benchmark style=openblas")
    fig.legend()

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
