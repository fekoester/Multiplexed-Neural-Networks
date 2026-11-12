"""Plot results/curves.csv (written by mnn_bench): mean test accuracy over seeds per epoch."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/curves.csv")
    ap.add_argument("--out", default="docs/curves.png")
    args = ap.parse_args()

    acc = defaultdict(lambda: defaultdict(list))  # (task, model) -> epoch -> [acc]
    params = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            key = (row["task"], row["model"])
            acc[key][int(row["epoch"])].append(float(row["test_accuracy"]))
            params[key] = (row["main"], row["control"])

    tasks = sorted({t for t, _ in acc})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.2))
    if len(tasks) == 1:
        axes = [axes]
    titles = {"gated": "context-gated classification", "letter": "UCI letter recognition"}
    for ax, task in zip(axes, tasks, strict=True):
        for (t, model), curve in acc.items():
            if t != task:
                continue
            epochs = sorted(curve)
            mean = [100 * sum(curve[e]) / len(curve[e]) for e in epochs]
            style = "-" if model.startswith("multiplexed") else "--"
            ax.plot(epochs, mean, style, label=model)
        ax.set_title(titles.get(task, task))
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
