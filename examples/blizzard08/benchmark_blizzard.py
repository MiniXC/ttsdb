from pathlib import Path
import tarfile

import pandas as pd
import numpy as np

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset
from ttsdb.benchmarks.external.pesq import PESQBenchmark

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

# external benchmarks
pesq_df = pd.DataFrame()
dataset = sorted(datasets, key=lambda x: x.name)
pesq = PESQBenchmark(datasets[0])
pesq_names = []
pesq_scores = []
for d in dataset:
    score = np.mean(pesq._get_distribution(d))
    pesq_names.append(d.name)
    pesq_scores.append(score)
pesq_df["name"] = pesq_names
pesq_df["pesq"] = pesq_scores
pesq_df.to_csv("pesq.csv", index=False)