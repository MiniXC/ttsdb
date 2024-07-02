from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset, TarDataset
from ttsdb.benchmarks.external.pesq import PESQBenchmark
from ttsdb.benchmarks.external.wv_mos import WVMOSBenchmark
from ttsdb.benchmarks.external.utmos import UTMOSBenchmark
from ttsdb.benchmarks.benchmark import Benchmark

# Extract the Blizzard 2008 dataset
if not Path("processed_data").exists():
    with tarfile.open("processed_data.tar.gz", "r:gz") as tar:
        tar.extractall()

# remove files starting with ._
for x in Path("processed_data").rglob("._*"):
    x.unlink()


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

datasets = [
    DirectoryDataset(Path(x), single_speaker=True)
    for x in Path("processed_data").iterdir()
    if len(x.name) == 1 and len(list(x.rglob("*.wav"))) > 0
]

with importlib.resources.path("ttsdb", "data") as dp:
    test_ds = TarDataset(dp / "reference" / "speech_blizzard2008.tar.gz", single_speaker=True)

benchmark_suite = BenchmarkSuite(
    datasets, 
    benchmarks=benchmarks,
    write_to_file="results.csv",
)

benchmark_suite.run()
name="T"
dist = benchmark_suite.get_benchmark_distribution("WeSpeaker", name, 2)

df = pd.DataFrame()

names = []
x = []
y = []
name = "T"
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

# Set up the figure
f, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")

# Draw a contour plot to represent each bivariate density
sns.kdeplot(
    data=df,
    x="PC1", y="PC2",
    hue="Dataset",
    style="Dataset",
    levels=5,
    bw_adjust=2,
    alpha=0.5,
    fill=True,
)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
plt.savefig("plot_i.svg")