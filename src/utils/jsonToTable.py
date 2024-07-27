import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open("benchmarkResults.json", "r") as file:
    data = json.load(file)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
fig.suptitle("Benchmark Results", fontsize=16)

ax1 = fig.add_subplot(gs[:, 0])
systemInfo = pd.DataFrame.from_dict(
    data["System Info"], orient="index", columns=["Value"]
)
systemInfo.index.name = "Property"
systemInfo = systemInfo.reset_index()

versionInfo = pd.DataFrame(
    {
        "Property": ["Version", "Testing Methodology"],
        "Value": [data["Version"], data["Testing Methodology"]],
    }
)
systemInfo = pd.concat([versionInfo, systemInfo]).reset_index(drop=True)

ax1.axis("off")
ax1.set_title("System Information")
table1 = ax1.table(
    cellText=systemInfo.values,
    colLabels=systemInfo.columns,
    cellLoc="left",
    loc="center",
)
table1.auto_set_font_size(False)
table1.set_fontsize(9)
table1.scale(1, 1.5)

ax2 = fig.add_subplot(gs[0, 1])
upscaleData = data["Results"]["Upscale"]
upscaleDf = pd.DataFrame.from_dict(
    upscaleData,
    orient="index",
    columns=["Highest FPS", "Highest FPS Value", "Average FPS", "Average FPS Value"],
)
upscaleDf = upscaleDf.drop(["Highest FPS", "Average FPS"], axis=1)
upscaleDf.index.name = "Model"
upscaleDf = upscaleDf.reset_index()
upscaleDf["Model"] = upscaleDf["Model"].str.title()

ax2.axis("off")
ax2.set_title("Upscale Results")
table2 = ax2.table(
    cellText=upscaleDf.values, colLabels=upscaleDf.columns, cellLoc="left", loc="center"
)
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.5)

ax3 = fig.add_subplot(gs[1, 1])
interpolateData = data["Results"]["Interpolate"]
interpolateDf = pd.DataFrame.from_dict(
    interpolateData,
    orient="index",
    columns=["Highest FPS", "Highest FPS Value", "Average FPS", "Average FPS Value"],
)
interpolateDf = interpolateDf.drop(["Highest FPS", "Average FPS"], axis=1)
interpolateDf.index.name = "Model"
interpolateDf = interpolateDf.reset_index()
interpolateDf["Model"] = interpolateDf["Model"].str.title()

ax3.axis("off")
ax3.set_title("Interpolate Results")
table3 = ax3.table(
    cellText=interpolateDf.values,
    colLabels=interpolateDf.columns,
    cellLoc="left",
    loc="center",
)
table3.auto_set_font_size(False)
table3.set_fontsize(9)
table3.scale(1, 1.5)

plt.tight_layout()
plt.savefig("benchmarkResults.png", dpi=300, bbox_inches="tight")

plt.show()
