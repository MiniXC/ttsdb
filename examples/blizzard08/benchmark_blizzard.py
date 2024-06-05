from pathlib import Path

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset

datasets = [
    DirectoryDataset(Path(x))
    for x in Path("processed_data").iterdir()
    if len(x.name) == 1 and len(list(x.rglob("*.wav"))) > 0
]

benchmark_suite = BenchmarkSuite(datasets, n_test_splits=2, n_samples_per_split=84)

df = benchmark_suite.run()
