import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

try:
    data = json.loads(open("benchmarkResults.json").read())
except FileNotFoundError:
    print("benchmarkResults.json not found")
    exit()

fig, axs = plt.subplots(1, len(data), figsize=(len(data) * 5, 5))

for ax, (category, methods) in zip(axs, data.items()):
    df = pd.DataFrame(list(methods.items()), columns=["Method", "FPS"])
    ax.axis("off")
    tbl = table(ax, df, loc="center")
    ax.set_title(category)

plt.tight_layout()
plt.savefig("tables.png")
