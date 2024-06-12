import tempfile
import os

import wespeaker
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset


class WeSpeakerBenchmark(Benchmark):
    """
    Benchmark class for the WeSpeaker benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="WeSpeaker",
            category=BenchmarkCategory.SPEAKER,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="The speaker embeddings using WeSpeaker.",
        )
        self.model = wespeaker.load_model('english')

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the WeSpeaker benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the WeSpeaker benchmark.
        """
        embeddings = []
        for wav, _, _ in tqdm(dataset, desc=f"computing embeddings for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                embedding = self.model.extract_embedding(f.name).numpy()
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        return embeddings
