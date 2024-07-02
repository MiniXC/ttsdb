from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
df = benchmark_suite.get_aggregated_results()

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

pesq_df = run_external_benchmark(PESQBenchmark(datasets[0]), datasets)
pesq_df["benchmark_category"] = "pesq"
wvmos_df = run_external_benchmark(WVMOSBenchmark(), datasets)
wvmos_df["benchmark_category"] = "wvmos"
utmos_df = run_external_benchmark(UTMOSBenchmark(), datasets)
utmos_df["benchmark_category"] = "utmos"

gt_mos_df = pd.read_csv("gt_mos.csv")
gt_mos_df["benchmark_category"] = "gt_mos"

# merge the dataframes
df["benchmark_type"] = "ttsdb"
pesq_df["benchmark_type"] = "external"
wvmos_df["benchmark_type"] = "external"
utmos_df["benchmark_type"] = "external"
gt_mos_df["benchmark_type"] = "mos"
df = pd.concat([df, pesq_df, wvmos_df, utmos_df, gt_mos_df])

# compute the correlations
corrs = []
for b in df["benchmark_category"].unique():
    bdf = df[df["benchmark_category"] == b]
    mosdf = df[df["benchmark_category"] == "gt_mos"]
    # sort both dataframes by dataset name
    mosdf = mosdf.sort_values("dataset")
    bdf = bdf.sort_values("dataset")
    # remove "A" dataset
    # mosdf = mosdf[mosdf["dataset"] != "A"]
    # bdf = bdf[bdf["dataset"] != "A"]
    assert (mosdf["dataset"].values == bdf["dataset"].values).all()
    if b == "gt_mos":
        continue
    # normalize the data
    bdf_score = bdf["score"]
    bdf["score"] = (bdf_score - bdf_score.min()) / (bdf_score.max() - bdf_score.min())
    corr, p = spearmanr(mosdf["score"], bdf["score"])
    corrs.append((b, corr, p))
    print(f"{b}: {corr:.3f} ({p:.3f})")


# compute the correlations with statsmodels
X = df[df["benchmark_type"] == "ttsdb"]
# remove "A" dataset
# X = X[X["dataset"] != "A"]
X = X.pivot(index="dataset", columns="benchmark_category", values="score")
X = X.sort_values("dataset")
X.drop("OVERALL", axis=1, inplace=True)
# remove index
X = X.reset_index()
X_mean = X
X_mean["mean"] = X_mean.drop("dataset", axis=1).apply(np.mean, axis=1)
print(X_mean.sort_values("mean"))
X = X.drop("dataset", axis=1)
# normalize the data per column


y = df[df["benchmark_category"] == "gt_mos"]
# remove "A" dataset
# y = y[y["dataset"] != "A"]
y = y.sort_values("dataset")
y = y.reset_index()
y = y["score"]

# mean
X_mean = X.apply(np.mean, axis=1)
# calculate the correlation
corr, p = spearmanr(X_mean, y)

print(f"mean: {corr:.3f} ({p:.3f})")


# save the correlations
corrs_df = pd.DataFrame(corrs, columns=["benchmark", "corr", "p"])
corrs_df.to_csv("correlations.csv", index=False)
