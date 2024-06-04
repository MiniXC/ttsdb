"""
This file contains the Benchmark abstract class.
"""

from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import importlib.resources
import json

import numpy as np

from ttsdb.util.dataset import Dataset, TarDataset
from ttsdb.util.cache import cache, load_cache, check_cache
from ttsdb.util.distances import wasserstein_distance, frechet_distance

N_TEST_DATASET_SPLITS = 10
N_SAMPLES_PER_SPLIT = 100


class BenchmarkCategory(Enum):
    """
    Enum class for the different categories of benchmarks.
    """

    OVERALL = 1
    PROSODY = 2
    ENVIRONMENT = 3
    SPEAKER = 4
    PRONUNCIATION = 5
    INTELLIGIBILITY = 6
    TRAINABILITY = 7


class BenchmarkDimension(Enum):
    """
    Enum class for the different dimensions of benchmarks.
    """

    ONE_DIMENSIONAL = 1
    N_DIMENSIONAL = 2


class Benchmark(ABC):
    """
    Abstract class for a benchmark.
    """

    def __init__(
        self,
        name: str,
        category: BenchmarkCategory,
        dimension: BenchmarkDimension,
        description: str,
        **kwargs,
    ):
        self.name = name
        self.category = category
        self.dimension = dimension
        self.description = description
        self.kwargs = kwargs

    def get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        If the benchmark is one-dimensional, the method should return a
        numpy array with the values of the benchmark for each sample in the dataset.
        If the benchmark is n-dimensional, the method should return a numpy array
        with the values of the benchmark for each sample in the dataset, where each
        row corresponds to a sample and each column corresponds to a dimension of the benchmark.
        """
        ds_hash = hash(dataset)
        benchmark_hash = hash(self)
        cache_name = f"{ds_hash}_{benchmark_hash}"
        if check_cache(cache_name):
            return load_cache(cache_name)
        distribution = self._get_distribution(dataset)
        cache(distribution, cache_name)
        return distribution

    @abstractmethod
    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.category.name}/{self.name}"

    def __repr__(self):
        return f"{self.category.name}/{self.name}"

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(self.name.encode())
        h.update(self.category.name.encode())
        h.update(self.dimension.name.encode())
        h.update(self.description.encode())
        # convert the kwargs to strings
        kwargs_str = {
            k: str(v) if not isinstance(v, dict) else json.dumps(v, sort_keys=True)
            for k, v in self.kwargs.items()
        }
        h.update(json.dumps(kwargs_str, sort_keys=True).encode())
        return int(h.hexdigest(), 16)

    def compute_distance(self, one_dataset: Dataset, other_dataset: Dataset) -> float:
        """
        Compute the distance between the distributions of the benchmark in two datasets.
        """
        one_distribution = self.get_distribution(one_dataset)
        other_distribution = self.get_distribution(other_dataset)
        if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
            return wasserstein_distance(one_distribution, other_distribution)
        elif self.dimension == BenchmarkDimension.N_DIMENSIONAL:
            return frechet_distance(one_distribution, other_distribution)
        else:
            raise ValueError("Invalid benchmark dimension")

    def compute_score(self, dataset: Dataset) -> float:
        """
        Compute the score of the benchmark on a dataset.
        """
        # we calculate the distributions
        # for the test and noise datasets
        # the score is from 0 to 100
        # where 100 is equal distributions
        # and 0 is equal to the noise dataset
        with importlib.resources.path("ttsdb", "data") as data_path:
            noise_dataset = data_path / "noise.tar.gz"
            noise_tar_dataset = TarDataset(noise_dataset).sample(N_SAMPLES_PER_SPLIT)
            test_dataset = data_path / "libritts_test.tar.gz"
            test_tar_datasets = [
                TarDataset(test_dataset).sample(N_SAMPLES_PER_SPLIT, seed=i)
                for i in range(N_TEST_DATASET_SPLITS)
            ]
        noise_scores = []
        for test_ds in test_tar_datasets:
            score = self.compute_distance(test_ds, noise_tar_dataset)
            noise_scores.append(score)
        noise_scores = np.array(noise_scores)

        dataset_scores = []
        for test_ds in test_tar_datasets:
            score = self.compute_distance(test_ds, dataset)
            dataset_scores.append(score)
        dataset_scores = np.array(dataset_scores)

        scores = 100 * (1 - dataset_scores / noise_scores)
        confidence_interval = 1.96 * np.std(scores) / np.sqrt(N_TEST_DATASET_SPLITS)
        return np.mean(scores), confidence_interval
