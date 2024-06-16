from typing import Union
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
    TEST_DS = TarDataset(dp / "libritts_test.tar.gz")

class HubertTokenBenchmark(Benchmark):
    """
    Benchmark class for the Hubert benchmark.
    """

    def __init__(
        self,
        hubert_model: str = "facebook/hubert-base-ls960",
        hubert_layer: Union[int, str] = 8, 
        cluster_num: int = 100,
        cluster_seed: int = 42,
        cluster_dataset: Dataset = TEST_DS,
    ):
        super().__init__(
            name="HubertTokens",
            category=BenchmarkCategory.OVERALL,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Hubert hidden states.",
            hubert_model=hubert_model,
            hubert_layer=hubert_layer,
            cluster_num=cluster_num,
            cluster_seed=cluster_seed,
            cluster_dataset=cluster_dataset,
        )
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model_layer = hubert_layer
        self.kmeans = self.create_clusters(cluster_num, cluster_seed, cluster_dataset)

    def create_clusters(self, cluster_num: int, cluster_seed: int, cluster_dataset: Dataset) -> KMeans:
        """
        Create clusters for the Hubert benchmark.
        """
        cache_id = hash_md5(self.__cache__())
        if check_cache(cache_id):
            cluster_centres = load_cache(cache_id)
            kmeans = KMeans(n_clusters=cluster_num, random_state=cluster_seed)
            dummy = np.zeros((100, 768))
            kmeans.fit(dummy)
            kmeans.cluster_centers_ = cluster_centres
            return kmeans
        wavs = [
            wav
            for wav, _, _ in tqdm(
                cluster_dataset, desc=f"loading wavs for {self.name} {cluster_dataset}"
            )
        ]
        embeddings = []
        for wav in tqdm(wavs):
            embeddings.append(self.get_embedding(wav, cluster_dataset.sample_rate))
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
        if isinstance(self.model_layer, int):
            features = features[self.model_layer].detach().cpu().numpy()[0]
        else:
            layer_num = features.shape[0]
            features_new = []
            for i in range(layer_num):
                features_new.append(features[i].detach().cpu().numpy()[0])
            features = np.concatenate(features_new, axis=0)
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
