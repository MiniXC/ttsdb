import tempfile
import os

from pyannote.audio import Model, Inference
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset


class XVectorBenchmark(Benchmark):
    """
    Benchmark class for the XVector benchmark.
    """

    def __init__(
        self,
        window_duration: float = 3.0,
        window_step: float = 0.5,
    ):
        super().__init__(
            name="XVector",
            category=BenchmarkCategory.SPEAKER,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The speaker embeddings of XVector.",
            window_duration=window_duration,
            window_step=window_step,
        )
        model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.model = Inference(model, window="sliding", duration=3.0, step=1.0)

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the XVector benchmark.

        Args:
            dataset (Dataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the XVector benchmark.
        """
        embeddings = []
        for wav, _, _ in tqdm(dataset, desc=f"computing embeddings for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                embedding = Inference(self.model)
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        return embeddings
