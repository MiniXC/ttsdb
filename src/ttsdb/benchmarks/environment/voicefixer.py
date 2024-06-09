import tempfile

from voicefixer import VoiceFixer
from simple_hifigan import Synthesiser
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset


class VoiceFixerBenchmark(Benchmark):
    """
    Benchmark class for the VoiceFixer benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="VoiceFixer",
            category=BenchmarkCategory.PHONETICS,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="The phone counts of VoiceFixer.",
        )
        self.model = VoiceFixer()
        self.synthesiser = Synthesiser()

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Compute the Word Error Rate (WER) distribution of the VoiceFixer model.

        Args:
            dataset (Dataset): The dataset to compute the WER on.

        Returns:
            float: The Word Error Rate (WER) distribution of the VoiceFixer model.
        """
        mel_diffs = []
        for wav, _, _ in tqdm(dataset, desc=f"computing noise for {self.name}"):
            if dataset.sample_rate != 16000:
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                # take the 16000 samples from the middle
                num_secs = 2
                num_samples = num_secs * 16000
                if len(wav) > num_samples:
                    wav = wav[int(len(wav) / 2) - num_samples // 2 : int(len(wav) / 2) + num_samples // 2]
                sf.write(f.name, wav, num_samples)
                with tempfile.NamedTemporaryFile(suffix=".wav") as f_out:
                    self.model.restore(f.name, f_out.name)
                    wav_out, _ = librosa.load(f_out.name, sr=16000)
            wav = wav / np.max(np.abs(wav))
            wav_out = wav_out / np.max(np.abs(wav_out))
            mel = self.synthesiser.wav_to_mel(wav, 16000)[0].T
            mel_out = self.synthesiser.wav_to_mel(wav_out, 16000)[0].T
            if mel_out.shape[0] > mel.shape[0]:
                mel_out = mel_out[: mel.shape[0]]
            elif mel_out.shape[0] < mel.shape[0]:
                mel = mel[: mel_out.shape[0]]
            mel_diff = mel - mel_out
            mel_diffs.append(mel_diff)
        mel_diffs = np.vstack(mel_diffs)
        return mel_diffs
