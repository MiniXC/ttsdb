# SPDX-FileCopyrightText: 2024-present Christoph Minixhofer <christoph.minixhofer@gmail.com>
#
# SPDX-License-Identifier: MIT
from typing import List, Optional
import importlib.resources
from time import time
from pathlib import Path
import pickle
import gzip

import pandas as pd
from transformers import logging
import numpy as np
from sklearn.decomposition import PCA

from ttsdb.benchmarks.environment.voicefixer import VoiceFixerBenchmark
from ttsdb.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsdb.benchmarks.general.hubert import HubertBenchmark
from ttsdb.benchmarks.general.wav2vec2 import Wav2Vec2Benchmark
from ttsdb.benchmarks.general.wavlm import WavLMBenchmark
from ttsdb.benchmarks.intelligibility.w2v2_wer import Wav2Vec2WERBenchmark
from ttsdb.benchmarks.intelligibility.whisper_wer import WhisperWERBenchmark
from ttsdb.benchmarks.prosody.mpm import MPMBenchmark
from ttsdb.benchmarks.prosody.pitch import PitchBenchmark
from ttsdb.benchmarks.prosody.hubert_token import HubertTokenBenchmark
from ttsdb.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
from ttsdb.benchmarks.speaker.dvector import DVectorBenchmark
from ttsdb.benchmarks.external.wv_mos import WVMOSBenchmark
from ttsdb.benchmarks.trainability.kaldi import KaldiBenchmark
from ttsdb.benchmarks.benchmark import BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset, TarDataset

# we do this to avoid "some weights of the model checkpoint at ... were not used when initializing" warnings
logging.set_verbosity_error()


benchmark_dict = {
    "hubert": HubertBenchmark(),
    "wav2vec2": Wav2Vec2Benchmark(),
    "wavlm": WavLMBenchmark(),
    "w2v2": Wav2Vec2WERBenchmark(),
    "whisper": WhisperWERBenchmark(),
    "mpm": MPMBenchmark(),
    "pitch": PitchBenchmark(),
    "wespeaker": WeSpeakerBenchmark(),
    "dvector": DVectorBenchmark(),
    "hubert_token": HubertTokenBenchmark(),
    "voicefixer": VoiceFixerBenchmark(),
    "wada_snr": WadaSNRBenchmark(),
}

DEFAULT_BENCHMARKS = [
    "hubert",
    "wav2vec2",
    "wavlm",
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

class DataDistribution:

    def __init__(self, dataset: Dataset = None, benchmarks: List[str] = DEFAULT_BENCHMARKS):
        self.benchmarks = benchmarks
        self.benchmark_objects = {
            benchmark: benchmark_dict[benchmark] for benchmark in benchmarks
        }
        self.benchmark_results = {}
        if dataset is not None:
            self.dataset = dataset
            self.run()
    
    def run(self):
        for benchmark in self.benchmark_objects:
            print(f"Running {benchmark} on {self.dataset.root_dir}")
            bench = self.benchmark_objects[benchmark]
            dist = bench.get_distribution(self.dataset)
            if bench.dimension == BenchmarkDimension.N_DIMENSIONAL:
                # compute mu and sigma and store as tuple
                mu = np.mean(dist, axis=0)
                sigma = np.cov(dist, rowvar=False)
                dist = (mu, sigma)
            self.benchmark_results[benchmark] = dist

    def get_distribution(benchmark_name: str) -> np.ndarray:
        if benchmark_name not in self.benchmark_results:
            self.run()
        return self.benchmark_results[benchmark_name]

    def to_pickle(self, path: str):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.benchmark_results, f)

    @staticmethod
    def from_pickle(path: str):
        with gzip.open(path, "rb") as f:
            benchmark_results = pickle.load(f)
        obj = DataDistribution()
        obj.benchmark_results = benchmark_results
        return obj

with importlib.resources.path("ttsdb", "data") as data_path:
    REFERENCE_DISTS = [
        DataDistribution.from_pickle(f"src/ttsdb/data/reference_{name}.pkl.gz") for name in [
            "speech_blizzard2008",
            "speech_blizzard2013",
            "speech_common_voice",
            "speech_libritts_test",
            "speech_libritts_r_test",
            "speech_lj_speech",
            "speech_vctk",
        ]
    ]

    NOISE_DISTS = [
        DataDistribution.from_pickle(f"src/ttsdb/data/noise_{name}.pkl.gz") for name in [
            "esc50",
            "noise_all_ones",
            "noise_all_zeros",
            "noise_normal_distribution",
            "noise_uniform_distribution",
        ]
    ]

class BenchmarkSuite:

    def __init__(
        self,
        datasets: List[Dataset],
        benchmarks: List[str] = DEFAULT_BENCHMARKS,
        print_results: bool = True,
        skip_errors: bool = False,
        noise_distributions: List[DataDistribution] = NOISE_DISTS,
        reference_distributions: List[DataDistribution] = REFERENCE_DISTS,
        write_to_file: str = None,
    ):
        self.benchmarks = benchmarks
        self.benchmark_objects = [
            benchmark_dict[benchmark] for benchmark in benchmarks
        ]
        # sort by category and then by name
        self.benchmark_objects = sorted(
            self.benchmark_objects, key=lambda x: (x.category.value, x.name)
        )
        self.datasets = datasets
        self.datasets = sorted(self.datasets, key=lambda x: x.name)
        self.database = pd.DataFrame(
            columns=["benchmark_name", "benchmark_category", "dataset", "score", "ci", "time_taken", "noise_dataset", "reference_dataset"]
        )
        self.print_results = print_results
        self.skip_errors = skip_errors
        self.noise_distributions = noise_distributions
        self.reference_distributions = reference_distributions
        self.write_to_file = write_to_file
        if Path(write_to_file).exists():
            self.database = pd.read_csv(write_to_file, index_col=0)
            self.database = self.database.reset_index()

    def run(self) -> pd.DataFrame:
        for benchmark in self.benchmark_objects:
            for dataset in self.datasets:
                # empty lines for better readability
                print("\n")
                print(f"{'='*80}")
                print(f"Benchmark Category: {benchmark.category.name}")
                print(f"Running {benchmark.name} on {dataset.root_dir}")
                try:
                    # check if it's in the database
                    if (
                        (self.database["benchmark_name"]==benchmark.name)
                        & (self.database["dataset"]==dataset.name)
                    ).any():
                        print(f"Skipping {benchmark.name} on {dataset.name} as it's already in the database")
                        continue
                    start = time()
                    score = benchmark.compute_score(dataset, self.reference_distributions, self.noise_distributions)
                    time_taken = time() - start
                except Exception as e:
                    if self.skip_errors:
                        print(f"Error: {e}")
                        score = (np.nan, np.nan)
                        time_taken = np.nan
                    else:
                        raise e
                result = {
                    "benchmark_name": [benchmark.name],
                    "benchmark_category": [benchmark.category.value],
                    "dataset": [dataset.name],
                    "score": [score[0]],
                    "ci": [score[1]],
                    "time_taken": [time_taken],
                    "noise_dataset": [score[2][0]],
                    "reference_dataset": [score[2][1]],
                }
                if self.print_results:
                    print(result)
                self.database = pd.concat(
                    [
                        self.database,
                        pd.DataFrame(
                            result
                        ),
                    ],
                    ignore_index=True,
                )
                if self.write_to_file is not None:
                    self.database["score"] = self.database["score"].astype(float)
                    self.database = self.database.sort_values(["benchmark_category", "benchmark_name", "score"], ascending=[True, True, False])
                    self.database.to_csv(self.write_to_file, index=False)
        return self.database

    def get_aggregated_results(self) -> pd.DataFrame:
        def concat_text(x):
            return ", ".join(x)
        df = self.database.copy()
        df["benchmark_category"] = df["benchmark_category"].apply(lambda x: BenchmarkCategory(x).name)
        df = df.groupby([
            "benchmark_category",
            "dataset",
        ]).agg(
            {
                "score": ["mean"],
                "ci": ["mean"],
                "time_taken": ["mean"],
                "noise_dataset": [concat_text],
                "reference_dataset": [concat_text],
                "benchmark_name": [concat_text],
            }
        ).reset_index()
        # remove multiindex
        df.columns = [x[0] for x in df.columns.ravel()]
        # drop the benchmark_name column
        df = df.drop("benchmark_name", axis=1)
        # replace benchmark_category number with string
        return df

    def get_benchmark_distribution(self, benchmark_name: str, dataset_name: str, pca_components: Optional[int] = None) -> dict:
        benchmark = [x for x in self.benchmark_objects if x.name == benchmark_name][0]
        dataset = [x for x in self.datasets if x.name == dataset_name][0]
        closest_noise = self.database[
            (self.database["benchmark_name"] == benchmark_name)
            & (self.database["dataset"] == dataset_name)
        ]["noise_dataset"].values[0]
        closest_noise = [x for x in self.noise_distributions if x.name == closest_noise][0]
        other_noise = [x for x in self.noise_distributions if x.name != closest_noise.name][0]
        closest_reference = self.database[
            (self.database["benchmark_name"] == benchmark_name)
            & (self.database["dataset"] == dataset_name)
        ]["reference_dataset"].values[0]
        closest_reference = [x for x in self.reference_distributions if x.name == closest_reference][0]
        other_reference = [x for x in self.reference_distributions if x.name != closest_reference.name][0]
        result = {
            "benchmark_distribution": benchmark.get_distribution(dataset),
            "noise_distribution": benchmark.get_distribution(closest_noise),
            "reference_distribution": benchmark.get_distribution(closest_reference),
            "other_noise_distribution": benchmark.get_distribution(other_noise),
            "other_reference_distribution": benchmark.get_distribution(other_reference),
        }
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            # fit on all except the benchmark distribution
            pca.fit(np.vstack([v for k, v in result.items() if k != "benchmark_distribution"]))
            result = {k: pca.transform(v) for k, v in result.items()}
        return result