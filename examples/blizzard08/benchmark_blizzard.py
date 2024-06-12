from pathlib import Path
import tarfile

import pandas as pd
import numpy as np

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset
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
    "kaldi",
    "mfcc",
    "hubert",
    "w2v2",
    "whisper",
    "mpm",
    "pitch",
    "wespeaker",
    "allosaurus",
    "voicefixer",
    "wada_snr",
]

datasets = [
    DirectoryDataset(Path(x), single_speaker=True)
    for x in Path("processed_data").iterdir()
    if len(x.name) == 1 and len(list(x.rglob("*.wav"))) > 0
]

benchmark_suite = BenchmarkSuite(
    datasets, 
    benchmarks=benchmarks,
    write_to_file="results.csv", 
    kaldi={"verbose": True},
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

run_external_benchmark(PESQBenchmark(datasets[0]), datasets)
run_external_benchmark(WVMOSBenchmark(), datasets)
run_external_benchmark(UTMOSBenchmark(), datasets)