from pathlib import Path
import tarfile

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset

# Extract the Blizzard 2008 dataset
if not Path("processed_data").exists():
    with tarfile.open("processed_data.tar.gz", "r:gz") as tar:
        tar.extractall()

# remove files starting with ._
for x in Path("processed_data").rglob("._*"):
    x.unlink()

datasets = [
    DirectoryDataset(Path(x), single_speaker=True)
    for x in Path("processed_data").iterdir()
    if len(x.name) == 1 and len(list(x.rglob("*.wav"))) > 0
]

benchmark_suite = BenchmarkSuite(datasets, write_to_file="results.csv", kaldi={"verbose": True})

df = benchmark_suite.run()
