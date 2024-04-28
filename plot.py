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


for path in Path(BASEPATH).glob("*.csv"):
    with path.open("r") as fd:
        print(path)
        reader = csv.reader(CommentIgnorer(fd), delimiter="\t")
        header = next(reader)

        if len(header) != 5:
            raise AssertionError(f"invalid header: {header}")

        for row in reader:
            if not row:
                continue
            print(row)
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

            if row_mode != MODE:
                continue

            avg_tokens_per_second = float(avg_tokens_per_second)
            stddev_tokens_per_second = float(stddev_tokens_per_second)
            actors_data[hostname].append((thread_count, avg_tokens_per_second))


pprint.pprint(actors_data)
# Plot each actor's data
for actor, data in actors_data.items():
    x_values = [item[0] for item in data]
    y_values = [item[1] for item in data]
    plt.plot(x_values, y_values, label=actor)

# Add labels and title
plt.xlabel("thread count")
plt.ylabel("tokens per second")
if MODE == "cpu":
    plt.title("benchmark style=clean")
else:
    raise Exception("TODO")


# Add legend
plt.legend()
# Display the plot
plt.show()
