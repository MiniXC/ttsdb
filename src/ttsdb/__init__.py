# SPDX-FileCopyrightText: 2024-present Christoph Minixhofer <christoph.minixhofer@gmail.com>
#
# SPDX-License-Identifier: MIT
from typing import List
import importlib.resources
from time import time
from pathlib import Path

import pandas as pd
from transformers import logging
import numpy as np

from ttsdb.benchmarks.environment.voicefixer import VoiceFixerBenchmark
from ttsdb.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsdb.benchmarks.general.hubert import HubertBenchmark
from ttsdb.benchmarks.general.mfcc import MFCCBenchmark
from ttsdb.benchmarks.intelligibility.w2v2_wer import Wav2Vec2WERBenchmark
from ttsdb.benchmarks.intelligibility.whisper_wer import WhisperWERBenchmark
from ttsdb.benchmarks.phonetics.allosaurus import AllosaurusBenchmark
from ttsdb.benchmarks.prosody.mpm import MPMBenchmark
from ttsdb.benchmarks.prosody.pitch import PitchBenchmark
from ttsdb.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
from ttsdb.benchmarks.trainability.kaldi import KaldiBenchmark
from ttsdb.util.dataset import Dataset, TarDataset

# we do this to avoid "some weights of the model checkpoint at ... were not used when initializing" warnings
logging.set_verbosity_error()


benchmark_dict = {
    "mfcc": MFCCBenchmark,
    "hubert": HubertBenchmark,
    "w2v2": Wav2Vec2WERBenchmark,
    "whisper": WhisperWERBenchmark,
    "mpm": MPMBenchmark,
    "pitch": PitchBenchmark,
    "wespeaker": WeSpeakerBenchmark,
    "allosaurus": AllosaurusBenchmark,
    "voicefixer": VoiceFixerBenchmark,
    "wada_snr": WadaSNRBenchmark,
    "kaldi": KaldiBenchmark,
}

DEFAULT_BENCHMARKS = [
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

with importlib.resources.path("ttsdb", "data") as data_path:
    REFERENCE_DATASETS = [
        data_path / "reference/speech_blizzard2008.tar.gz",
        data_path / "reference/speech_blizzard2013.tar.gz",
        data_path / "reference/speech_common_voice.tar.gz",
        data_path / "reference/speech_libritts_test.tar.gz",
        data_path / "reference/speech_lj_speech.tar.gz",
        data_path / "reference/speech_vctk.tar.gz",
    ]
    REFERENCE_DATASETS = [TarDataset(x, single_speaker=True) for x in REFERENCE_DATASETS]

    NOISE_DATASETS = [
        data_path / "noise/esc50.tar.gz",
        data_path / "noise/noise_all_ones.tar.gz",
        data_path / "noise/noise_all_zeros.tar.gz",
        data_path / "noise/noise_normal_distribution.tar.gz",
        data_path / "noise/noise_uniform_distribution.tar.gz",
    ]
    NOISE_DATASETS = [TarDataset(x, single_speaker=True) for x in NOISE_DATASETS]
    # we need this for kaldis benchmark, to avoid unnecessary computation
    for dataset in NOISE_DATASETS:
        dataset.is_noise_dataset = True

class BenchmarkSuite:

    def __init__(
        self,
        datasets: List[Dataset],
        benchmarks: List[str] = DEFAULT_BENCHMARKS,
        print_results: bool = True,
        skip_errors: bool = False,
        noise_datasets: List[Dataset] = NOISE_DATASETS,
        reference_datasets: List[Dataset] = REFERENCE_DATASETS,
        write_to_file: str = None,
        **kwargs,
    ):
        self.benchmarks = benchmarks
        self.benchmark_objects = []
        for benchmark in benchmarks:
            kwargs_for_benchmark = kwargs.get(benchmark, {})
            self.benchmark_objects.append(benchmark_dict[benchmark](**kwargs_for_benchmark))
        # sort by category and then by name
        self.benchmark_objects = sorted(
            self.benchmark_objects, key=lambda x: (x.category.value, x.name)
        )
        self.datasets = datasets
        self.datasets = sorted(self.datasets, key=lambda x: x.name)
        self.database = pd.DataFrame(
            columns=["benchmark_name", "benchmark_category", "dataset", "score", "ci", "time_taken"]
        )
        self.print_results = print_results
        self.skip_errors = skip_errors
        self.noise_datasets = noise_datasets
        self.reference_datasets = reference_datasets
        self.write_to_file = write_to_file
        if Path(write_to_file).exists():
            self.database = pd.read_csv(write_to_file)

    def run(self) -> pd.DataFrame:
        for benchmark in self.benchmark_objects:
            for dataset in self.datasets:
                # empty lines for better readability
                print("\n")
                print(f"{'='*80}")
                print(f"Benchmark Category: {benchmark.category.value}")
                print(f"Running {benchmark.name} on {dataset.root_dir}")
                try:
                    # check if it's in the database
                    if (
                        self.database["benchmark_name"].str.contains(benchmark.name)
                        & self.database["dataset"].str.contains(dataset.name)
                    ).any():
                        print(f"Skipping {benchmark.name} on {dataset.name} as it's already in the database")
                        continue
                    start = time()
                    score = benchmark.compute_score(dataset, self.reference_datasets, self.noise_datasets)
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
                }
                if self.print_results:
                    print(result)
                self.database = pd.concat(
                    [
                        self.database,
                        pd.DataFrame(
                            result
                        ),
                    ]
                )
                if self.write_to_file is not None:
                    self.database.to_csv(self.write_to_file, index=False)
        return self.database
