import matplotlib.pyplot as plt

# Values for A (Antigen Based)
avg_A = 0.2300
max_A = 0.2673
min_A = 0.1644
std_A = 0.0316

# Values for B (Random)
avg_B = 0.2183
max_B = 0.2690
min_B = 0.1696
std_B = 0.0374

# Values for C (Hybrid)
avg_C = 0.2292
max_C = 0.2692
min_C = 0.1730
std_C = 0.0329

# Values for D (Heuristic)
avg_D = 0.2356
max_D = 0.2721
min_D = 0.1640
std_D = 0.0362

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
ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"])
ax.set_ylabel("Accuracy")
ax.set_title("Leaking - Politifact")

# Only show one legend entry
ax.plot([], [], color="blue", linewidth=2, label="Average")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
