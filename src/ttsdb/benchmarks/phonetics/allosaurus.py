import tempfile

from allosaurus.app import read_recognizer
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset


class AllosaurusBenchmark(Benchmark):
    """
    Benchmark class for the Allosaurus benchmark.
    """

    def __init__(
        self,
        num_phones: int = 100,
    ):
        super().__init__(
            name="Allosaurus",
            category=BenchmarkCategory.PHONETICS,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The phone counts of Allosaurus.",
            num_phones=num_phones,
        )
        self.model = read_recognizer()
        self.num_phones = num_phones

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Compute the phone distribution of the Allosaurus model.

        Args:
            dataset (Dataset): The dataset to extract the phones for.

        Returns:
            float: The phone distribution of the Allosaurus model.
        """
        phone_dict = {}
        for wav, _, _ in tqdm(dataset, desc=f"computing WER for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                sf.write(f.name, wav, 16000)
                phones = self.model.recognize(f.name)
            for phone in phones:
                if phone in phone_dict:
                    phone_dict[phone] += 1
                else:
                    phone_dict[phone] = 1
        result = np.array(list(phone_dict.values()))
        result = result / np.sum(result)
        # sort descending
        result = result[np.argsort(result)[::-1]]
        # only take the first N phones
        result = result[: self.num_phones]
        if len(result) < self.num_phones:
            result = np.pad(result, (0, self.num_phones - len(result)))
        return result
