# SPDX-FileCopyrightText: 2024-present Christoph Minixhofer <christoph.minixhofer@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib.resources

from ttsdb.benchmarks.benchmark import BenchmarkCategory, BenchmarkDimension
from ttsdb.benchmarks.general.mfcc import MFCCBenchmark
from ttsdb.benchmarks.general.hubert import HubertBenchmark
from ttsdb.util.dataset import TarDataset

with importlib.resources.path("ttsdb", "data") as data_path:
    test_dataset = data_path / "libritts_test.tar.gz"
    dev_dataset = data_path / "libritts_dev.tar.gz"
    noise_dataset = data_path / "noise.tar.gz"
    test_tar_dataset = TarDataset(test_dataset).sample(100)
    dev_tar_dataset = TarDataset(dev_dataset).sample(100)
    noise_tar_dataset = TarDataset(noise_dataset).sample(100)


# def test_mfcc_compute_distance():
#     benchmark = MFCCBenchmark()
#     result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
#     assert int(result) == 242


# def test_mfcc_compute_score():
#     benchmark = MFCCBenchmark()
#     result = benchmark.compute_score(dev_tar_dataset)
#     assert result[0] == 99.87295897554043


def test_hubert_compute_distance():
    benchmark = HubertBenchmark()
    result = benchmark.compute_distance(test_tar_dataset, dev_tar_dataset)
    assert result == 2.141043000170157


def test_hubert_compute_score():
    benchmark = HubertBenchmark()
    result = benchmark.compute_score(dev_tar_dataset)
    print(result)
