# Re-import matplotlib since the code execution environment was reset
import matplotlib.pyplot as plt

# Values for A
avg_A = 0.4336
max_A = 0.6471
min_A = 0.1111
std_A = 0.1587

# Values for B
avg_B = 0.5882
max_B = 0.6608
min_B = 0.4614
std_B = 0.0651

# Setup
fig, ax = plt.subplots(figsize=(8, 6))  # Wider for two plots

positions = [1, 2]
box_width = 0.2


def draw_stat_box(ax, x, avg, std, min_val, max_val, label=None):
    box_bottom = avg - std
    box_top = avg + std

    # Whisker: min to bottom of box
    ax.vlines(x, min_val, box_bottom, color="black", linewidth=1.5)
    ax.hlines(
        min_val, x - box_width / 2, x + box_width / 2, color="black", linewidth=1.5
    )

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
    ax.hlines(
        max_val, x - box_width / 2, x + box_width / 2, color="black", linewidth=1.5
    )

    # Average line
    ax.hlines(avg, x - box_width / 2, x + box_width / 2, color="blue", linewidth=2)


# Draw for A and B
draw_stat_box(ax, positions[0], avg_A, std_A, min_A, max_A, label="Std Dev")
draw_stat_box(ax, positions[1], avg_B, std_B, min_B, max_B)

# Formatting
ax.set_xlim(0.5, 2.5)
ax.set_xticks(positions)
ax.set_xticklabels(["Antige Based", "Random"])
ax.set_ylabel("Accuracy")
ax.set_title("Initialisation - Averitec")

# Only show one legend entry
ax.plot([], [], color="blue", linewidth=2, label="Average")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
