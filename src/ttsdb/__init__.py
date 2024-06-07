# SPDX-FileCopyrightText: 2024-present Christoph Minixhofer <christoph.minixhofer@gmail.com>
#
# SPDX-License-Identifier: MIT
from typing import List, Tuple

import pandas as pd

from ttsdb.benchmarks.environment.voicefixer import VoiceFixerBenchmark
from ttsdb.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsdb.benchmarks.general.hubert import HubertBenchmark
from ttsdb.benchmarks.general.mfcc import MFCCBenchmark
from ttsdb.benchmarks.intelligibility.w2v2_wer import Wav2Vec2WERBenchmark
from ttsdb.benchmarks.intelligibility.whisper_wer import WhisperWERBenchmark
from ttsdb.benchmarks.phonetics.allosaurus import AllosaurusBenchmark
from ttsdb.benchmarks.prosody.mpm import MPMBenchmark
from ttsdb.benchmarks.prosody.pitch import PitchBenchmark
from ttsdb.benchmarks.speaker.xvector import XVectorBenchmark
from ttsdb.benchmarks.trainability.kaldi import KaldiBenchmark
from ttsdb.util.dataset import Dataset

benchmark_dict = {
    "mfcc": MFCCBenchmark,
    "hubert": HubertBenchmark,
    "w2v2": Wav2Vec2WERBenchmark,
    "whisper": WhisperWERBenchmark,
    "mpm": MPMBenchmark,
    "pitch": PitchBenchmark,
    "xvector": XVectorBenchmark,
    "allosaurus": AllosaurusBenchmark,
    "voicefixer": VoiceFixerBenchmark,
    "wada_snr": WadaSNRBenchmark,
    "kaldi": KaldiBenchmark,
}

DEFAULT_BENCHMARKS = [
    "mfcc",
    "hubert",
    "w2v2",
    "whisper",
    "mpm",
    "pitch",
    "xvector",
    "allosaurus",
    "voicefixer",
    "wada_snr",
]


class BenchmarkSuite:

    def __init__(
        self,
        datasets: List[Dataset],
        benchmarks: List[str] = DEFAULT_BENCHMARKS,
        n_test_splits: int = None,
        n_samples_per_split: int = None,
    ):
        self.benchmarks = benchmarks
        self.benchmark_objects = [
            benchmark_dict[benchmark]() for benchmark in benchmarks
        ]
        # sort by category and then by name
        self.benchmark_objects = sorted(
            self.benchmark_objects, key=lambda x: (x.category.value, x.name)
        )
        self.datasets = datasets
        self.database = pd.DataFrame(
            columns=["benchmark_name", "benchmark_category", "dataset", "score", "ci"]
        )
        self.n_test_splits = n_test_splits
        self.n_samples_per_split = n_samples_per_split

    def run(self) -> pd.DataFrame:
        for benchmark in self.benchmark_objects:
            for dataset in self.datasets:
                # empty lines for better readability
                print("\n")
                print(f"{'='*80}")
                print(f"Benchmark Category: {benchmark.category.value}")
                print(f"Running {benchmark.name} on {dataset.root_dir}")
                if (
                    self.n_test_splits is not None
                    and self.n_samples_per_split is not None
                ):
                    score = benchmark.compute_score(
                        dataset,
                        n_test_splits=self.n_test_splits,
                        n_samples_per_split=self.n_samples_per_split,
                    )
                else:
                    score = benchmark.compute_score(dataset)
                self.database = pd.concat(
                    [
                        self.database,
                        pd.DataFrame(
                            {
                                "benchmark_name": [benchmark.name],
                                "benchmark_category": [benchmark.category.value],
                                "dataset": [dataset.name],
                                "score": [score[0]],
                                "ci": [score[1]],
                            }
                        ),
                    ]
                )
        return self.database
