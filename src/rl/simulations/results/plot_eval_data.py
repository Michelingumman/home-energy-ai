import matplotlib.pyplot as plt
import pandas as pd

# 1) Point at your evaluation CSV, not the consumption CSV:

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

csv_path = "C:/_Projects/home-energy-ai/src/rl/simulations/results/evaluation_data_short_term_agent_final.csv"
# 1) Load your evaluation CSV
df = pd.read_csv(csv_path, parse_dates=['timestamps'], index_col='timestamps')

# 2) Pick the columns you want to overlay
cols = [
    "reward",
    "total_cost_cumulative",
    "grid_cost_term",
    "peak_penalty_term",
    "battery_cost_term",
    "arbitrage_bonus_term",
    "soc_action_penalty_term"
]

# 3) Create a host axes
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)   # leave space on the right for extra y-axes

# 4) Create one parasite (twin) axes per extra series
parasites = []
for i in range(len(cols)-1):
    p = host.twinx()   # share the same x-axis
    # move the spine out to the right by 60px per axis
    offset = 60 * (i+1)
    p.axis['right'] = p.new_fixed_axis(loc='right', offset=(offset, 0))
    p.axis['right'].toggle(all=True)
    parasites.append(p)

# 5) Plot each series on its own axis
axes = [host] + parasites
colors = plt.cm.tab10.colors

for ax, col, c in zip(axes, cols, colors):
    ax.plot(df.index, df[col], color=c, label=col)
    ax.set_ylabel(col, color=c)
    ax.axis['right'].label.set_color(c) if ax is not host else ax.yaxis.label.set_color(c)
    ax.tick_params(axis='y', colors=c)

# 6) Legend & formatting
lines, labels = [], []
for ax in axes:
    l, lab = ax.get_legend_handles_labels()
    lines += l; labels += lab
host.legend(labels, loc='upper left')
host.set_xlabel("Timestamp")
plt.gcf().autofmt_xdate()
plt.title("All series, independent y-scales")
plt.show()
