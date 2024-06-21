from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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

pesq_df = run_external_benchmark(PESQBenchmark(datasets[0]), datasets)
pesq_df["benchmark_name"] = "pesq"
wvmos_df = run_external_benchmark(WVMOSBenchmark(), datasets)
wvmos_df["benchmark_name"] = "wvmos"
utmos_df = run_external_benchmark(UTMOSBenchmark(), datasets)
utmos_df["benchmark_name"] = "utmos"

gt_mos_df = pd.read_csv("gt_mos.csv")
gt_mos_df["benchmark_name"] = "gt_mos"

# merge the dataframes
df["benchmark_type"] = "ttsdb"
pesq_df["benchmark_type"] = "external"
wvmos_df["benchmark_type"] = "external"
utmos_df["benchmark_type"] = "external"
gt_mos_df["benchmark_type"] = "mos"
df = pd.concat([df, pesq_df, wvmos_df, utmos_df, gt_mos_df])

# compute the correlations
corrs = []
for b in df["benchmark_name"].unique():
    bdf = df[df["benchmark_name"] == b]
    mosdf = df[df["benchmark_name"] == "gt_mos"]
    # sort both dataframes by dataset name
    mosdf = mosdf.sort_values("dataset")
    bdf = bdf.sort_values("dataset")
    # remove "A" dataset
    assert (mosdf["dataset"].values == bdf["dataset"].values).all()
    if b == "gt_mos":
        continue
    # normalize the data
    bdf_score = bdf["score"]
    bdf["score"] = (bdf_score - bdf_score.min()) / (bdf_score.max() - bdf_score.min())
    corr, p = pearsonr(mosdf["score"], bdf["score"])
    corrs.append((b, corr, p))
    print(f"{b}: {corr:.3f} ({p:.3f})")


dfx = df[df["benchmark_type"] == "ttsdb"]
dfx = dfx.pivot(index="dataset", columns="benchmark_name", values="score")
dfx = dfx.sort_values("dataset")
# make columns lowercase and replace space with underscore
dfx.columns = [x.lower().replace(" ", "_") for x in dfx.columns]
cols = '+'.join(dfx.columns)
y = df[df["benchmark_name"] == "gt_mos"]
y = y.sort_values("dataset")
dfx["gt_mos"] = y["score"].values

y, X = dmatrices(f"gt_mos ~ {cols}", data=dfx, return_type='dataframe')

#create DataFrame to hold VIF values
vif_df = pd.DataFrame()
vif_df['variable'] = X.columns

#calculate VIF for each predictor variable 
vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#view VIF for each predictor variable 
print(vif_df)

# compute the correlations with statsmodels
X = df[df["benchmark_type"] == "ttsdb"]
X = X.pivot(index="dataset", columns="benchmark_name", values="score")
X = X.sort_values("dataset")
# remove index
X = X.reset_index()
X = X.drop("dataset", axis=1)
# normalize the data per column

def normalize_min_max(values):
  min_val = values.min()
  max_val = values.max()
  vals = (values - min_val) / (max_val - min_val)
  return vals

X = X.apply(normalize_min_max, axis=0)

y = df[df["benchmark_name"] == "gt_mos"]
y = y.sort_values("dataset")
y = y.reset_index()
y = y["score"]

# mean
X_mean = X.apply(np.mean, axis=1)
# calculate the correlation
corr, p = pearsonr(X_mean, y)

print(f"mean: {corr:.3f} ({p:.3f})")



from sklearn.linear_model import Ridge

# fit the model
model = Ridge()
model.fit(X, y)
# set negative coefs to zero
model.coef_[model.coef_ < 0] = 0
# save model
import joblib
joblib.dump(model, "ridge.joblib")
# make predictions
yhat = model.predict(X)
# calculate the correlation
corr, p = pearsonr(y, yhat)
print(f"ridge: {corr:.3f} ({p:.3f})")
corrs.append(("ridge", corr, p))
print(model.coef_)


# save the correlations
corrs_df = pd.DataFrame(corrs, columns=["benchmark", "corr", "p"])
corrs_df.to_csv("correlations.csv", index=False)
