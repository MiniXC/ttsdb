import librosa
import numpy as np

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset


class MFCCBenchmark(Benchmark):
    """
    Benchmark class for the Mel-Frequency Cepstral Coefficients (MFCC) benchmark.
    """

    def __init__(
        self,
        **mfcc_kwargs,
    ):
        super().__init__(
            name="MFCC",
            category=BenchmarkCategory.OVERALL,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="The Mel-Frequency Cepstral Coefficients (MFCC) benchmark.",
            **mfcc_kwargs,
        )
        self.sample_rate = None

    def get_mfcc(self, wav) -> np.ndarray:
        """
        Get the MFCC of a wav file.

        Args:
            wav (np.ndarray): The wav file.

        Returns:
            np.ndarray: The MFCC of the wav file.
        """
        return librosa.feature.mfcc(y=wav, sr=self.sample_rate, **self.kwargs).T

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the MFCC benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the MFCC benchmark.
        """
        wavs = [
            wav
            for wav, _, _ in tqdm(
                dataset, desc=f"loading wavs for {self.name} {dataset}"
            )
        ]
        self.sample_rate = dataset.sample_rate
        mfccs = process_map(
            self.get_mfcc, wavs, desc=f"computing mfccs for {self.name} {dataset}"
        )
        return np.concatenate(mfccs)
