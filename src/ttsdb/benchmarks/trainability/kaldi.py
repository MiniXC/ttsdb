from dataclasses import dataclass, fields
from multiprocessing import cpu_count
import os
import platform
from subprocess import run
import subprocess
from typing import List
import importlib.resources
from pathlib import Path
from enum import Enum
import re

import librosa
from numpy import ndarray
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm

from ttsdb.util.cache import CACHE_DIR
from ttsdb.benchmarks.benchmark import Benchmark, BenchmarkCategory, BenchmarkDimension
from ttsdb.util.dataset import TarDataset, Dataset

cpus = min(16, cpu_count())

KALDI_PATH = os.getenv("TTSDB_KALDI_PATH", CACHE_DIR / "kaldi")
with importlib.resources.path("ttsdb", "data") as data_path:
    TEST_DS = TarDataset(data_path / "libritts_test.tar.gz").sample(100)
DS_CACHE = CACHE_DIR / "kaldi_datasets"


def run_commands(commands: List[str], directory: str = None):
    """
    Run a list of commands.
    """
    for command in commands:
        if directory:
            run(command, shell=True, check=True, cwd=directory)
        run(command, shell=True, check=True)


def run_command(command: str, directory: str = None, suppress_output: bool = False):
    """
    Run a command.
    """
    if suppress_output:
        if directory:
            run(
                command,
                shell=True,
                check=True,
                cwd=directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif directory:
        run(command, shell=True, check=True, cwd=directory)
    else:
        run(command, shell=True, check=True)


def install_kaldi(kaldi_path: str):
    """
    Install Kaldi.
    """
    if not kaldi_path.exists():
        yn_install_kaldi = input(
            f"TTSDB_KALDI_PATH is not set. Do you want to install Kaldi to {kaldi_path}? (y/n) "
        )
    else:
        yn_install_kaldi = input(f"Overwrite Kaldi at {kaldi_path}? (y/n) ")
        if yn_install_kaldi.lower() == "n":
            return
        run_command(f"rm -rf {kaldi_path}")
    kaldi_path = kaldi_path.resolve()
    if yn_install_kaldi.lower() == "y":
        run(
            f"git clone https://github.com/kaldi-asr/kaldi.git {KALDI_PATH}",
            shell=True,
            check=True,
        )
        run(
            f"cd {kaldi_path} && git checkout d136b18346bee14166b950029405314401fc4a8b",
            shell=True,
            check=True,
        )
        try:
            is_osx = platform.system() == "Darwin"
            if not is_osx:
                run_commands(
                    [
                        f"cd {kaldi_path}/tools && \
                            sed -i 's/python2.7/python3/' extras/check_dependencies.sh"
                    ]
                )
            else:
                run_commands(
                    [
                        f"cd {kaldi_path}/tools && \
                            sed -i '' 's/python2.7/python3/' extras/check_dependencies.sh"
                    ]
                )
            run_commands(
                [
                    f"cd {kaldi_path}/tools && ./extras/check_dependencies.sh",
                    f"cd {kaldi_path}/tools && make -j {cpus}",
                    f"cd {kaldi_path}/src && ./configure --shared",
                    f"cd {kaldi_path}/src && make depend -j {cpus}",
                    f"cd {kaldi_path}/src && make -j {cpus}",
                ]
            )
        except Exception as e:
            print(f"Error installing Kaldi: {e}")
            # remove kaldi
            run(f"rm -rf {kaldi_path}", shell=True, check=True)
            raise e


def dataset_to_kaldi(dataset: Dataset, egs_path: Path) -> Path:
    """
    Convert a dataset to Kaldi format.
    """
    ds_hash = hash(dataset)
    dest = DS_CACHE / f"{ds_hash}"
    if (
        dest.exists()
        and len(list(dest.rglob("*.wav"))) == len(dataset)
        and len(list(dest.rglob("*.txt"))) == len(dataset)
        and (dest / "spk2utt").exists()
        and (dest / "text").exists()
        and (dest / "utt2spk").exists()
        and (dest / "wav.scp").exists()
        and (dest / "feats.scp").exists()
    ):
        return dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    data_dict = {"speaker": [], "id": [], "wav": [], "text": []}
    for idx, (wav, text, speaker) in tqdm(
        enumerate(dataset), desc=f"converting {dataset} to kaldi format"
    ):
        wav_path = dest / f"{speaker}_{idx}.wav"
        text_path = dest / f"{speaker}_{idx}.txt"
        # resample wav to 16kHz
        if not Path(wav_path).exists():
            wav = librosa.resample(wav, orig_sr=dataset.sample_rate, target_sr=16000)
            sf.write(wav_path, wav, 16000)
        if not Path(text_path).exists():
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
        data_dict["speaker"].append(speaker)
        data_dict["id"].append(f"{speaker}_{idx}")
        data_dict["wav"].append(str(wav_path))
        data_dict["text"].append(text)
    df = pd.DataFrame(data_dict)
    # spk2utt
    with open(dest / "spk2utt", "w", encoding="utf-8") as spk2utt:
        for spk in sorted(df["speaker"].unique()):
            utts = df[df["speaker"] == spk]["id"].unique()
            utts = sorted([f"{utt}" for utt in utts])
            spk2utt.write(f'{spk} {" ".join(utts)}\n')
    # text
    with open(dest / "text", "w", encoding="utf-8") as text_file:
        for utt in sorted(df["id"].unique()):
            text = df[df["id"] == utt]["text"].values[0]
            text_file.write(f"{utt} {text}\n")
    # utt2spk
    with open(dest / "utt2spk", "w", encoding="utf-8") as utt2spk:
        for spk in sorted(df["speaker"].unique()):
            utts = df[df["speaker"] == spk]["id"].unique()
            for utt in sorted(utts):
                utt2spk.write(f"{utt} {spk}\n")
    # wav.scp
    with open(dest / "wav.scp", "w", encoding="utf-8") as wavscp:
        for utt in sorted(df["id"].unique()):
            wav = df[df["id"] == utt]["wav"].values[0]
            wav = Path(wav).resolve()
            wavscp.write(f"{utt} sox {wav} -t wav -c 1 -b 16 -t wav - rate 16000 |\n")
    run_command(
        f"steps/make_mfcc.sh --cmd run.pl --nj {cpus} {dest} {dest} mfccs",
        directory=egs_path,
    )
    run_command(
        f"steps/compute_cmvn_stats.sh {dest} {dest} mfccs",
        directory=egs_path,
    )
    return dest.resolve()


def score_model(
    egs_path: Path, model_path: Path, test_set: Path, exp_path: Path, data_path: Path
) -> np.ndarray:
    """
    Score a model.
    """
    graph_path = model_path / "graph_tgsmall"
    if not graph_path.exists():
        run_command(
            f"utils/mkgraph.sh {data_path / 'lang_nosp_test_tgsmall'} {model_path} {graph_path}",
            directory=egs_path,
        )
    graph_test_path = model_path / "graph_tgsmall_test"
    tst_path = test_set
    run_command(
        f"steps/decode.sh --nj {cpus} --cmd run.pl {graph_path} {tst_path} {graph_test_path}",
        directory=egs_path,
    )
    scoring_path = graph_test_path / "scoring_kaldi"
    run_command(
        f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
        directory=egs_path,
    )
    with open(scoring_path / "best_wer", "r") as best_wer:
        best_wer = best_wer.read()
    wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
    try:
        ins_err = int(re.findall(r"(\d+)\s+ins", best_wer)[0])
    except Exception:
        ins_err = int(re.findall(r"(\d+)\s+in", best_wer)[0])
    del_err = int(re.findall(r"(\d+)\s+del", best_wer)[0])
    try:
        sub_err = int(re.findall(r"(\d+)\s+sub", best_wer)[0])
    except Exception:
        sub_err = int(re.findall(r"(\d+)\s+ub", best_wer)[0])
    with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
        bootci = bootci.read()
    lower_wer, upper_wer = [
        round(float(c), 2)
        for c in re.findall(r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci)[0]
    ]
    ci = round((upper_wer - lower_wer) / 2, 2)
    print(
        {
            "wer": wer,
            "wer_lower": lower_wer,
            "wer_upper": upper_wer,
            "ci_width": ci,
            "ins": ins_err,
            "del": del_err,
            "sub": sub_err,
        }
    )
    per_utt_info = (scoring_path / "wer_details" / "per_utt").read_text().split("\n")
    per_utt_counts = [p.split(" ref")[-1] for p in per_utt_info if " ref" in p]
    per_utt_counts = [
        float(len([x for x in p.strip().split() if "***" not in x]))
        for p in per_utt_counts
    ]
    per_utt_sid = [
        p.split("#csid")[-1].strip().split() for p in per_utt_info if "#csid" in p
    ]
    per_utt_sid = [float(p[1]) + float(p[2]) + float(p[3]) for p in per_utt_sid]
    per_utt_wer = [x / y for x, y in zip(per_utt_sid, per_utt_counts)]
    print(per_utt_wer)
    return per_utt_wer


class KaldiStage(Enum):
    MONO = 1
    TRI1 = 2
    TRI2B = 3
    TRI3B = 4


class KaldiBenchmark(Benchmark):
    """
    Benchmark class for the Kaldi benchmark.
    """

    def __init__(
        self,
        kaldi_path: str = KALDI_PATH,
        test_set: Dataset = TEST_DS,
    ):
        super().__init__(
            name="Kaldi",
            category=BenchmarkCategory.TRAINABILITY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Kaldi WER.",
            kaldi_path=kaldi_path,
        )
        self.kaldi_path = kaldi_path
        self.egs_path = self.kaldi_path / "egs" / "librispeech" / "s5"
        # test kali installation
        try:
            run_command(
                f"{kaldi_path}/src/featbin/compute-mfcc-feats --help",
                suppress_output=True,
            )
        except Exception as e:
            print(f"Error: {e}")
            install_kaldi(kaldi_path)
        self.test_set = dataset_to_kaldi(test_set, self.egs_path)
        os.environ["KALDI_PATH"] = str(self.kaldi_path.resolve())
        from ttsdb.benchmarks.trainability.legacy_kaldi import Args, get_kaldi_wer

        self.args = Args()
        self.args.local_data_path = CACHE_DIR / "kaldi_datasets"
        self.args.local_exp_path = CACHE_DIR / "kaldi_experiments"
        # delete experiments
        if self.args.local_exp_path.exists():
            run_command(f"rm -rf {self.args.local_exp_path}")
        self.kaldi_wer = get_kaldi_wer

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        train_set = dataset_to_kaldi(dataset, self.egs_path)
        wer = self.kaldi_wer(
            self.args,
            train_set,
            self.test_set,
            cache_dir=CACHE_DIR / "kaldi_cache",
        )


kb = KaldiBenchmark()

with importlib.resources.path("ttsdb", "data") as data_path:
    dev_ds = TarDataset(data_path / "libritts_dev.tar.gz").sample(100)

kb._get_distribution(dev_ds)
