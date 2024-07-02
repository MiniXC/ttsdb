from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset, TarDataset
from ttsdb.benchmarks.external.pesq import PESQBenchmark
from ttsdb.benchmarks.external.wv_mos import WVMOSBenchmark
from ttsdb.benchmarks.external.utmos import UTMOSBenchmark
from ttsdb.benchmarks.trainability.kaldi import KaldiBenchmark
from ttsdb.benchmarks.benchmark import Benchmark

datasets = sorted(list(Path("data").rglob("*.tar.gz")))
datasets = [TarDataset(x) for x in datasets]

benchmarks = [
    "hubert",
    "wav2vec2",
    "w2v2",
    "whisper",
    "mpm",
    "pitch",
    "wespeaker",
    "dvector",
    "hubert_token",
    "voicefixer",
    "wada_snr",
]

name = "parler.tar.gz"
# check if pickled file exists
if Path(f"{name}_dist.pkl").exists():
    with open(f"{name}_dist.pkl", "rb") as f:
        dist = pickle.load(f)
else:
    benchmark_suite = BenchmarkSuite(
        datasets, 
        benchmarks=benchmarks,
        write_to_file="results.csv",
    )
    benchmark_suite.run()
    dist = benchmark_suite.get_benchmark_distribution("Hubert", name, 2)
    with open(f"{name}_dist.pkl", "wb") as f:
        pickle.dump(dist, f)

df = pd.DataFrame()

names = []
x = []
y = []
name = "Synthetic"
for i, val in enumerate(dist["benchmark_distribution"]):
    names.append(name)
    x.append(val[0])
    y.append(val[1])
for i, val in enumerate(dist["reference_distribution"]):
    names.append("Reference")
    x.append(val[0])
    y.append(val[1])
for i, val in enumerate(dist["noise_distribution"]):
    names.append("Noise")
    x.append(val[0])
    y.append(val[1])

df["Dataset"] = names
df["PC1"] = x
df["PC2"] = y

name = "voicecraft2.tar.gz"
if Path(f"{name}_dist.pkl").exists():
    with open(f"{name}_dist.pkl", "rb") as f:
        dist = pickle.load(f)
else:
    benchmark_suite = BenchmarkSuite(
        datasets, 
        benchmarks=benchmarks,
        write_to_file="results.csv",
    )
    benchmark_suite.run()
    dist = benchmark_suite.get_benchmark_distribution("Hubert", name, 2)
    with open(f"{name}_dist.pkl", "wb") as f:
        pickle.dump(dist, f)

df_1 = pd.DataFrame()

names = []
x = []
y = []
name = "Synthtic"
for i, val in enumerate(dist["benchmark_distribution"]):
    names.append(name)
    x.append(val[0])
    y.append(val[1])
for i, val in enumerate(dist["reference_distribution"]):
    names.append("Reference")
    x.append(val[0])
    y.append(val[1])
for i, val in enumerate(dist["noise_distribution"]):
    names.append("Noise")
    x.append(val[0])
    y.append(val[1])

df_1["Dataset"] = names
df_1["PC1"] = x
df_1["PC2"] = y

# make appropriate for a paper figure
sns.set_context("talk")
sns.set_palette("tab10")
# font size
plt.rcParams.update({'font.size': 14})
# font family
plt.rcParams.update({'font.family': 'serif'})
# marker size
plt.rcParams.update({'lines.markersize': 10})
# line width
plt.rcParams.update({'lines.linewidth': 6})

# Set up the figure=
fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=2, sharey=True)

# ax.set_aspect("equal")

# Draw a contour plot to represent each bivariate density
kde1 = sns.kdeplot(
    data=df,
    x="PC1", y="PC2",
    hue="Dataset",
    style="Dataset",
    levels=5,
    bw_adjust=2,
    alpha=0.5,
    fill=True,
    ax=ax[0],
    legend=True,
)

kde2 = sns.kdeplot(
    data=df_1,
    x="PC1", y="PC2",
    hue="Dataset",
    style="Dataset",
    levels=5,
    bw_adjust=2,
    alpha=0.5,
    fill=True,
    ax=ax[1],
    legend=False,
)


ax[0].set_xlim(-4, 4)
ax[0].set_ylim(-4, 4)
ax[1].set_xlim(-4, 4)
ax[1].set_ylim(-4, 4)

# legend below plot
sns.move_legend(ax[0], "lower center", bbox_to_anchor=(1.19, -1.18), ncol=3)
# make sure the legend is not cut off
plt.tight_layout()

plt.savefig("test.svg", bbox_inches='tight')


# share x axis and legend
# sns.kdeplot(
#     data=df,
#     x="PC1", y="PC2",
#     hue="Dataset",
#     style="Dataset",
#     levels=5,
#     bw_adjust=2,
#     alpha=0.5,
#     fill=True,
# )

# sns.kdeplot(
#     data=df_1,
#     x="PC1", y="PC2",
#     hue="Dataset",
#     style="Dataset",
#     levels=5,
#     bw_adjust=2,
#     alpha=0.5,
#     fill=True,
# )

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.savefig("plot_parler.svg")