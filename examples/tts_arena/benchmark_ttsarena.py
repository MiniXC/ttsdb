from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset, TarDataset
from ttsdb.benchmarks.external.pesq import PESQBenchmark
from ttsdb.benchmarks.external.wv_mos import WVMOSBenchmark
from ttsdb.benchmarks.external.utmos import UTMOSBenchmark
from ttsdb.benchmarks.trainability.kaldi import KaldiBenchmark
from ttsdb.benchmarks.benchmark import Benchmark

datasets = sorted(list(Path(".").rglob("*.tar.gz")))
datasets = [TarDataset(x) for x in datasets]

benchmarks = [
    "mfcc",
    "hubert",
    "w2v2",
    "whisper",
    "mpm",
    "pitch",
    "wespeaker",
    "hubert_token",
    "allosaurus",
    "voicefixer",
    "wada_snr",
]

with importlib.resources.path("ttsdb", "data") as dp:
    test_ds = TarDataset(dp / "reference" / "speech_libritts_test.tar.gz", single_speaker=True)

benchmark_suite = BenchmarkSuite(
    datasets, 
    benchmarks=benchmarks,
    write_to_file="results.csv", 
    hubert_token={"cluster_dataset": test_ds},
)

df = benchmark_suite.run()

# sort datasets
datasets = sorted(datasets, key=lambda x: x.name)

def run_external_benchmark(benchmark: Benchmark, datasets: list):
    if Path(f"{benchmark.name.lower()}.csv").exists():
        return pd.read_csv(f"{benchmark.name.lower()}.csv")
    df = pd.DataFrame()
    names = []
    scores = []
    for d in datasets:
        score = np.mean(benchmark._get_distribution(d))
        names.append(d.name)
        scores.append(score)
    df["dataset"] = names
    df["score"] = scores
    df.to_csv(f"{benchmark.name.lower()}.csv", index=False)
    return df

# pesq_df = run_external_benchmark(PESQBenchmark(datasets[5]), datasets)
# pesq_df["benchmark_name"] = "pesq"
wvmos_df = run_external_benchmark(WVMOSBenchmark(), datasets)
wvmos_df["benchmark_name"] = "wvmos"
utmos_df = run_external_benchmark(UTMOSBenchmark(), datasets)
utmos_df["benchmark_name"] = "utmos"
# kaldi_df = run_external_benchmark(KaldiBenchmark(verbose=True, test_set=test_ds), datasets)
# kaldi_df["benchmark_name"] = "kaldi"

gt_mos_df = pd.read_csv("gt_mos.csv")
gt_mos_df["benchmark_name"] = "gt_mos"
# normalize the scores
gt_mos_df["score"] = (gt_mos_df["score"] - gt_mos_df["score"].min()) / (gt_mos_df["score"].max() - gt_mos_df["score"].min())


# print systems ordered by score
print(gt_mos_df.sort_values("score"))

# merge the dataframes
df["benchmark_type"] = "ttsdb"
# pesq_df["benchmark_type"] = "external"
wvmos_df["benchmark_type"] = "external"
utmos_df["benchmark_type"] = "external"
# kaldi_df["benchmark_type"] = "external"
gt_mos_df["benchmark_type"] = "mos"
df = pd.concat([df, wvmos_df, utmos_df, gt_mos_df])

# remove parlertts and vokan
# df = df[~df["dataset"].str.contains("gpt")]
# gt_mos_df = gt_mos_df[~gt_mos_df["dataset"].str.contains("gpt")]
df = df[~df["dataset"].str.contains("meta")]
gt_mos_df = gt_mos_df[~gt_mos_df["dataset"].str.contains("meta")]
df = df[~df["dataset"].str.contains("melo")]
gt_mos_df = gt_mos_df[~gt_mos_df["dataset"].str.contains("melo")]

# remove VoiceFixer and WadaSNR benchmarks
df = df[~df["benchmark_name"].str.contains("VoiceFixer")]
df = df[~df["benchmark_name"].str.contains("WadaSNR")]
df = df[~df["benchmark_name"].str.contains("Allosaurus")]

# compute the correlations
corrs = []

# compute the correlations with statsmodels
X = df[df["benchmark_type"] == "ttsdb"]
X = X.pivot(index="dataset", columns="benchmark_name", values="score")
X = X.sort_values("dataset")
# remove index
X = X.reset_index()

def normalize_min_max(values):
  min_val = values.min()
  max_val = values.max()
  vals = (values - min_val) / (max_val - min_val)
  return vals

# apply to all columns except dataset
x_ds = X["dataset"]
X["dataset"] = x_ds

# print systems ordered by harmonic mean
X_mean = X
# mean of all columns except dataset
X_mean["mean"] = X_mean.drop("dataset", axis=1).apply(np.mean, axis=1)
print(X_mean.sort_values("mean"))

X = X.drop("dataset", axis=1)

y = df[df["benchmark_name"] == "gt_mos"]
# remove parlertts and vokan
y = y.sort_values("dataset")
y = y.reset_index()
y = y["score"]



X_mean = X.apply(np.mean, axis=1)
# min_max normalize
X_mean = (X_mean - X_mean.min()) / (X_mean.max() - X_mean.min())
# get correlation with harmonic mean
print(y.shape, X_mean.shape)
corr, p = spearmanr(y, X_mean)
# print systems ordered by harmonic mean
print(f"mean: {corr:.3f} ({p:.3f})")

from matplotlib import pyplot as plt
import seaborn as sns

# plot the scatter plot
plt.figure(figsize=(10, 10))
sns.scatterplot(x=y, y=X_mean)
plt.xlabel("Ground truth elo")
plt.ylabel("Harmonic mean of benchmark scores")
plt.savefig("scatter.png")

for b in df["benchmark_name"].unique():
    bdf = df[df["benchmark_name"] == b]
    mosdf = df[df["benchmark_name"] == "gt_mos"]
    hmean_score = X_mean
    # sort both dataframes by dataset name
    mosdf = mosdf.sort_values("dataset")
    bdf = bdf.sort_values("dataset")
    assert (mosdf["dataset"].values == bdf["dataset"].values).all()
    if b == "gt_mos":
        continue
    bdf_score = bdf["score"]
    bdf["score"] = (bdf_score - bdf_score.min()) / (bdf_score.max() - bdf_score.min())
    corr, p = spearmanr(mosdf["score"], bdf["score"])
    corrs.append((b, corr, p))
    print(f"{b}: {corr:.3f} ({p:.3f})")
    # get correlation with harmonic mean
    hmean_corr, hmean_p = spearmanr(bdf["score"], hmean_score)
    # print(f"{b} harmonic mean: {hmean_corr:.3f} ({hmean_p:.3f})")

# save the correlations
corrs_df = pd.DataFrame(corrs, columns=["benchmark", "corr", "p"])
corrs_df.to_csv("correlations.csv", index=False)
