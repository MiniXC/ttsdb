from typing import Union, List
import importlib.resources

import numpy as np
import torch
from transformers import Wav2Vec2Processor, HubertModel
from tqdm import tqdm
import librosa
from sklearn.cluster import KMeans

from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import Dataset, TarDataset
from ttsdb.util.cache import cache, load_cache, check_cache, hash_md5

with importlib.resources.path("ttsdb", "data") as dp:
    TEST_DS = []
    TEST_DS.append(TarDataset(dp / "reference" / "speech_libritts_test.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_libritts_r_test.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_lj_speech.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_vctk.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_blizzard2008.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_blizzard2013.tar.gz", single_speaker=True))
    TEST_DS.append(TarDataset(dp / "reference" / "speech_common_voice.tar.gz", single_speaker=True))

class HubertTokenBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
        hubert_model: str = "facebook/hubert-base-ls960",
        hubert_layer: Union[int, str] = 7, 
        cluster_num: int = 100,
        cluster_seed: int = 42,
        cluster_datasets: List[Dataset] = TEST_DS,
    ):
        super().__init__(
            name="HubertToken",
            category=BenchmarkCategory.PROSODY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Hubert hidden states.",
            hubert_model=hubert_model,
            hubert_layer=hubert_layer,
            cluster_num=cluster_num,
            cluster_seed=cluster_seed,
            cluster_datasets=cluster_datasets,
        )
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model_layer = hubert_layer
        self.kmeans = self.create_clusters(cluster_num, cluster_seed, cluster_datasets)

    def create_clusters(self, cluster_num: int, cluster_seed: int, cluster_datasets: Dataset) -> KMeans:
        """
        Create clusters for the Hubert benchmark.
        """
        cache_id = self.__hash__()
        print(f"cache_id: {cache_id}")
        if check_cache(cache_id):
            cluster_centres = load_cache(cache_id)
            kmeans = KMeans(n_clusters=cluster_num, random_state=cluster_seed)
            dummy = np.zeros((100, 768))
            kmeans.fit(dummy)
            kmeans.cluster_centers_ = cluster_centres
            return kmeans
        wavs = []
        for ds in tqdm(cluster_datasets, desc=f"loading wavs for {self.name}"):
            wavs.extend([(wav, ds.sample_rate) for wav, _, _ in ds])
        embeddings = []
        for wav in tqdm(wavs):
            embeddings.append(self.get_embedding(wav[0], wav[1]))
        embeddings = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=cluster_num, random_state=cluster_seed).fit(embeddings)
        cache(kmeans.cluster_centers_, cache_id)
        return kmeans

    def get_embedding(self, wav, sr) -> np.ndarray:
        """
        Get the embedding of a wav file.

        Args:
            wav (np.ndarray): The wav file.

        Returns:
            np.ndarray: The embedding of the wav file.
        """
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        input_values = self.processor(
            wav, return_tensors="pt", sampling_rate=sr
        ).input_values
        with torch.no_grad():
            features = self.model(input_values, output_hidden_states=True).hidden_states
        features = features[self.model_layer].detach().cpu().numpy()[0]
        return features

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of the Hubert benchmark.

        Args:
            dataset (DirectoryDataset): The dataset to get the distribution from.

        Returns:
            np.ndarray: The distribution of the Hubert benchmark.
        """
        wavs = [
            wav
            for wav, _, _ in tqdm(
                dataset, desc=f"loading wavs for {self.name} {dataset}"
            )
        ]
        lengths = []
        for wav in tqdm(wavs):
            wav_emb = self.get_embedding(wav, dataset.sample_rate)
            cluster = self.kmeans.predict(wav_emb)
            # the lengths are the number of times each cluster is repeated in a row
            current_length = 1
            for i in range(1, len(cluster)):
                if cluster[i] == cluster[i-1]:
                    current_length += 1
                else:
                    lengths.append(current_length)
                    current_length = 1
            lengths.append(current_length)
        return np.array(lengths)
