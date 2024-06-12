from random import sample

import torchaudio
import torch
import torch.nn as nn
import requests
import numpy as np

import lightning_module
from change_sample_rate import ChangeSampleRate
from ttsdb.util.cache import CACHE_DIR

CHECKPOINT_URL = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt?download=true"
CHECKPOINT_PATH = CACHE_DIR / "utmos.ckpt"

if not CHECKPOINT_PATH.exists():
    print("Downloading UTMOS checkpoint...")
    response = requests.get(CHECKPOINT_URL)
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(response.content)

class UTMOSBenchmark(Benchmark):
    """
    Benchmark class for the UTMOS benchmark.
    """

    def __init__(
        self,
    ):
        super().__init__(
            name="UTMOS",
            category=BenchmarkCategory.EXTERNAL,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="The UTMOS benchmark.",
        )
        self.model = model = lightning_module.BaselineLightningModule.load_from_checkpoint(CHECKPOINT_PATH)
        self.model.eval()


    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the UTMOS benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the UTMOS benchmark.
        """
        scores = []
        for wav, _, _ in dataset:
            batch = torch.tensor(wav).unsqueeze(0).repeat(10, 1, 1)
            csr = ChangeSampleRate(sr, 16_000)
            out_wavs = csr(wav)
            batch = {
                'wav': out_wavs,
                'domains': torch.tensor([0]),
                'judge_id': torch.tensor([288])
            }
            with torch.no_grad():
                output = model(batch)
            score = output.mean(dim=1).squeeze().detach().numpy() * 2 + 3
            scores.append(score)
        return scores