import tempfile

import torch
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from alignments.aligners.mfa import MFAligner

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset
from ttsdb.util.kaldi import install_kaldi

class MFADurationBenchmark(Benchmark):
    """
    Benchmark class for the Masked Prosody Model (MPM) benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="MFA Duration",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The phone durations using montreal forced aligner.",
        )
        install_kaldi()
        self.aligner = MFAligner()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the MPM benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the MPM benchmark.
        """
        lengths = []
        for wav, text, _ in tqdm(dataset, desc="loading masked prosody model representations"):
            wav_path, txt_path = None, None
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, dataset.sample_rate)
                wav_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".txt") as f:
                f = open(f.name, "w")
                f.write(text)
                txt_path = f.name
            alignment = self.aligner.align_one(wav_path, txt_path)
            print(alignment)
