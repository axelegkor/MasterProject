import matplotlib.pyplot as plt
import numpy as np

# Dummy data
leaking_amounts = ["0.1", "0.2", "0.5", "1.0"]
x = np.arange(len(leaking_amounts))

accuracy_data = {
    "Liar": [0.2011, 0.1992, 0.1984, 0.2016],
    "Politifact": [0.2300, 0.2183, 0.2292, 0.2356],
    "Averitec": [0.6297, 0.5918, 0.5963, 0.6161]
}

coverage_data = {
    "Liar": [0.9968, 0.9953, 0.9969, 0.9976],
    "Politifact": [0.9975, 0.9975, 0.9978, 0.9981],
    "Averitec": [0.9325, 0.9139, 0.9668, 0.9526]
}

runtime_data = {
    "Liar": [77.0608, 83.5244, 102.1224, 122.2707],
    "Politifact": [195.4253, 216.8249, 249.0719, 300.7513],
    "Averitec": [9.3600, 9.2851, 12.8283, 13.0550]
}

distinct_colors = {
    "Liar": "#e41a1c",        # Red
    "Politifact": "#377eb8",  # Blue
    "Averitec": "#4daf4a"     # Green
}

def plot_metric(metric_data, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset, values in metric_data.items():
        ax.plot(x, values, marker='o', label=dataset, color=distinct_colors[dataset], linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(leaking_amounts)
    ax.set_xlabel("Leaking Amount")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Create updated plots with distinct colors
plot_metric(accuracy_data, "Accuracy", "Accuracy vs Leaking Amount")
plot_metric(coverage_data, "Coverage", "Coverage vs Leaking Amount")
plot_metric(runtime_data, "Runtime (s)", "Runtime vs Leaking Amount")
