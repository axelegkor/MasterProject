import matplotlib.pyplot as plt
import numpy as np

# Dummy data
population_sizes = ["0.001", "0.002", "0.005", "0.01"]
x = np.arange(len(population_sizes))

accuracy_data = {
    "Liar": [0.2027, 0.1878, 0.1954, 0.1971],
    "Politifact": [0.1997, 0.2093, 0.2202, 0.2360],
    "Averitec": [0.4572, 0.5123, 0.4777, 0.5807]
}

coverage_data = {
    "Liar": [0.7679, 0.9274, 0.9838, 0.9977],
    "Politifact": [0.9041, 0.9673, 0.9900, 0.9973],
    "Averitec": [0.4571, 0.6856, 0.7995, 0.9113]
}

runtime_data = {
    "Liar": [7.6584, 15.3116, 37.6339, 86.5321],
    "Politifact": [19.7531, 38.1507, 108.8123, 211.0121],
    "Averitec": [1.0090, 2.0153, 4.7358, 9.2638]
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
    ax.set_xticklabels(population_sizes)
    ax.set_xlabel("Population Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Create updated plots with distinct colors
plot_metric(accuracy_data, "Accuracy", "Accuracy vs Population Size")
plot_metric(coverage_data, "Coverage", "Coverage vs Population Size")
plot_metric(runtime_data, "Runtime (s)", "Runtime vs Population Size")
