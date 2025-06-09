import matplotlib.pyplot as plt

# Values for A (Antigen Based)
avg_A = 0.1997
max_A = 0.2660
min_A = 0.1261
std_A = 0.0409

# Values for B (Random)
avg_B = 0.2093
max_B = 0.2530
min_B = 0.1538
std_B = 0.0306

# Values for C (Hybrid)
avg_C = 0.2202
max_C = 0.2687
min_C = 0.1665
std_C = 0.0319

# Values for D (Heuristic)
avg_D = 0.2360
max_D = 0.2724
min_D = 0.1745
std_D = 0.0278

# Setup
fig, ax = plt.subplots(figsize=(12, 6))
positions = [1, 2, 3, 4]
box_width = 0.2

def draw_stat_box(ax, x, avg, std, min_val, max_val, label=None):
    box_bottom = avg - std
    box_top = avg + std

    # Whisker: min to bottom of box
    ax.vlines(x, min_val, box_bottom, color="black", linewidth=1.5)
    ax.hlines(min_val, x - box_width / 2, x + box_width / 2, color="black", linewidth=1.5)

    # Std dev box
    ax.add_patch(
        plt.Rectangle(
            (x - box_width / 2, box_bottom),
            box_width,
            box_top - box_bottom,
            facecolor="skyblue",
            edgecolor="black",
            label=label,
        )
    )

    # Whisker: top of box to max
    ax.vlines(x, box_top, max_val, color="black", linewidth=1.5)
    ax.hlines(max_val, x - box_width / 2, x + box_width / 2, color="black", linewidth=1.5)

    # Average line
    ax.hlines(avg, x - box_width / 2, x + box_width / 2, color="blue", linewidth=2)

# Draw for all methods
draw_stat_box(ax, positions[0], avg_A, std_A, min_A, max_A, label="Std Dev")
draw_stat_box(ax, positions[1], avg_B, std_B, min_B, max_B)
draw_stat_box(ax, positions[2], avg_C, std_C, min_C, max_C)
draw_stat_box(ax, positions[3], avg_D, std_D, min_D, max_D)

# Formatting
ax.set_xlim(0.5, 4.5)
ax.set_xticks(positions)
ax.set_xticklabels(["0.001", "0.002", "0.005", "0.01"])
ax.set_ylabel("Accuracy")
ax.set_title("Population Size - Politifact")

# Only show one legend entry
ax.plot([], [], color="blue", linewidth=2, label="Average")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
