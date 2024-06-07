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
with importlib.resources.path("ttsdb", "data") as dp:
    TEST_DS = TarDataset(dp / "libritts_test.tar.gz").sample(100)


def run_commands(
    commands: List[str], directory: str = None, suppress_output: bool = False
):
    """
    Run a list of commands.
    """
    for command in commands:
        if directory:
            run_command(command, directory, suppress_output)
        else:
            run_command(command, suppress_output=suppress_output)


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
        stage: KaldiStage = KaldiStage.MONO,
        verbose: bool = False,
    ):
        super().__init__(
            name="Kaldi",
            category=BenchmarkCategory.TRAINABILITY,
            dimension=BenchmarkDimension.ONE_DIMENSIONAL,
            description="Kaldi WER.",
            kaldi_path=kaldi_path,
        )
        self.kaldi_path = kaldi_path
        self.egs_path = self.kaldi_path / "egs/librispeech/s5"
        self.data_path = CACHE_DIR / "kaldi_data"
        # test kali installation
        try:
            run_command(
                f"{kaldi_path}/src/featbin/compute-mfcc-feats --help",
                suppress_output=True,
            )
        except Exception as e:
            print(f"Error: {e}")
            self.install_kaldi(kaldi_path)
        self.test_set = self.dataset_to_kaldi(test_set, self.egs_path, self.data_path)
        self.stage = stage
        self.verbose = verbose

    def install_kaldi(self, kaldi_path: str):
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
            run_command(f"rm -rf {kaldi_path}", suppress_output=not self.verbose)
        kaldi_path = kaldi_path.resolve()
        if yn_install_kaldi.lower() == "y":
            run(
                f"git clone https://github.com/kaldi-asr/kaldi.git {KALDI_PATH}",
                shell=True,
                check=True,
            )
            run(
                f"cd {kaldi_path} && git checkout 26b9f648",
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
                        ],
                        suppress_output=not self.verbose,
                    )
                else:
                    run_commands(
                        [
                            f"cd {kaldi_path}/tools && \
                                sed -i '' 's/python2.7/python3/' extras/check_dependencies.sh"
                        ],
                        suppress_output=not self.verbose,
                    )
                run_commands(
                    [
                        f"cd {kaldi_path}/tools && ./extras/check_dependencies.sh",
                        f"cd {kaldi_path}/tools && make -j {cpus}",
                        f"cd {kaldi_path}/src && ./configure --shared",
                        f"cd {kaldi_path}/src && make depend -j {cpus}",
                        f"cd {kaldi_path}/src && make -j {cpus}",
                    ],
                    suppress_output=not self.verbose,
                )
            except Exception as e:
                print(f"Error installing Kaldi: {e}")
                # remove kaldi
                run_command(f"rm -rf {kaldi_path}", suppress_output=not self.verbose)
                raise e

    def dataset_to_kaldi(
        self, dataset: Dataset, egs_path: Path, data_path: Path
    ) -> Path:
        """
        Convert a dataset to Kaldi format.
        """
        ds_hash = hash(dataset)
        dest = data_path / f"{ds_hash}"
        dest_mfcc = dest.parent / f"{dest.name}_mfcc"
        dest_mfcc.mkdir(parents=True, exist_ok=True)
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
            wav_path = dest / f"{speaker}-{idx}.wav"
            text_path = dest / f"{speaker}-{idx}.txt"
            # resample wav to 16kHz
            if not Path(wav_path).exists():
                wav = librosa.resample(
                    wav, orig_sr=dataset.sample_rate, target_sr=16000
                )
                sf.write(wav_path, wav, 16000)
            if not Path(text_path).exists():
                with open(text_path, "w", encoding="utf-8") as f:
                    text = text.upper()
                    f.write(text)
            data_dict["speaker"].append(speaker)
            data_dict["id"].append(f"{speaker}-{idx}")
            data_dict["wav"].append(str(wav_path))
            text = text.upper()
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
                wavscp.write(
                    f"{utt} sox {wav} -t wav -c 1 -b 16 -t wav - rate 16000 |\n"
                )
        run_command(
            f"steps/make_mfcc.sh --cmd run.pl --nj {cpus} {dest} {dest_mfcc} mfccs",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        run_command(
            f"steps/compute_cmvn_stats.sh {dest} {dest_mfcc} mfccs",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        return dest.resolve()

    def score_model(
        self, egs_path: Path, model_path: Path, test_set: Path, data_path: Path
    ) -> np.ndarray:
        """
        Score a model.
        """
        graph_path = model_path / "graph_tgsmall"
        if not graph_path.exists():
            run_command(
                f"utils/mkgraph.sh {data_path / 'lang_nosp_test_tgsmall'} {model_path} {graph_path}",
                directory=egs_path,
                suppress_output=not self.verbose,
            )
        graph_test_path = model_path / "graph_tgsmall_test"
        tst_path = test_set
        run_command(
            f"steps/decode.sh --nj {cpus} --cmd run.pl {graph_path} {tst_path} {graph_test_path}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        scoring_path = graph_test_path / "scoring_kaldi"
        run_command(
            f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        with open(scoring_path / "best_wer", "r") as best_wer:
            best_wer = best_wer.read()
        wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
        ins_err = int(re.findall(r"(\d+)\s+ins", best_wer)[0])
        del_err = int(re.findall(r"(\d+)\s+del", best_wer)[0])
        sub_err = int(re.findall(r"(\d+)\s+sub", best_wer)[0])
        with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
            bootci = bootci.read()
        lower_wer, upper_wer = [
            round(float(c), 2)
            for c in re.findall(r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci)[
                0
            ]
        ]
        ci = round((upper_wer - lower_wer) / 2, 2)
        if self.verbose:
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
        per_utt_info = (
            (scoring_path / "wer_details" / "per_utt").read_text().split("\n")
        )
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
        if self.verbose:
            print(per_utt_wer)
        return per_utt_wer

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        egs_path = self.egs_path
        local_data_path = self.data_path
        train_set = self.dataset_to_kaldi(dataset, egs_path, local_data_path)
        local_exp_path = CACHE_DIR / "kaldi_exp" / str(hash(self) + hash(dataset))
        # rm exp path
        if local_exp_path.exists():
            run_command(f"rm -r {local_exp_path}")
        lm_path = local_data_path / "local" / "lm"
        # validate data for both train and test
        run_commands(
            [
                f"utils/validate_data_dir.sh {train_set}",
                f"utils/validate_data_dir.sh {self.test_set}",
                f"local/download_lm.sh www.openslr.org/resources/11 {lm_path}",
                f"local/prepare_dict.sh --stage 3 --nj {cpus} --cmd run.pl \
                    {lm_path} {lm_path} {local_data_path / 'local' / 'dict_nosp'}",
                f"utils/prepare_lang.sh {local_data_path / 'local' / 'dict_nosp'} '<UNK>' \
                    {local_data_path / 'local' / 'lang_tmp'} {local_data_path / 'lang_nosp'}",
            ],
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        if not (local_data_path / "lang_nosp_test_tgsmall").exists():
            run_command(
                f"local/format_lms.sh --src-dir {local_data_path / 'lang_nosp'} {lm_path}",
                directory=egs_path,
                suppress_output=not self.verbose,
            )
        # always remove tgmed
        if (local_data_path / "lang_nosp_test_tgmed").exists():
            run_command(
                f"rm -r {local_data_path / 'lang_nosp_test_tgmed'}",
                directory=egs_path,
                suppress_output=not self.verbose,
            )
        # mono
        mono_data_path = str(train_set)
        run_command(
            f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {mono_data_path} {local_data_path / 'lang_nosp'} {local_exp_path / 'mono'}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        if self.stage == KaldiStage.MONO:
            return self.score_model(
                egs_path,
                local_exp_path / "mono",
                self.test_set,
                local_data_path,
            )
        # tri1
        tri1_data_path = str(train_set)
        tri1_ali_path = str(local_exp_path / "tri1") + "_ali"
        run_command(
            f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {tri1_data_path} {local_data_path / 'lang_nosp'} {local_exp_path / 'mono'} {tri1_ali_path}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        run_command(
            f"steps/train_deltas.sh --boost-silence 1.25 --cmd run.pl 2000 10000 \
                {tri1_data_path} {local_data_path / 'lang_nosp'} {tri1_ali_path} {local_exp_path / 'tri1'}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        if self.stage == KaldiStage.TRI1:
            return self.score_model(
                egs_path,
                local_exp_path / "tri1",
                self.test_set,
                local_data_path,
            )
        # tri2b
        tri2b_ali_path = str(local_exp_path / "tri2b") + "_ali"
        run_command(
            f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {train_set} {local_data_path / 'lang_nosp'} {local_exp_path / 'tri1'} {tri2b_ali_path}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        run_command(
            f"steps/train_lda_mllt.sh --boost-silence 1.25 --cmd run.pl --splice-opts '--left-context=3 --right-context=3' 2500 15000 \
                {train_set} {local_data_path / 'lang_nosp'} {tri2b_ali_path} {local_exp_path / 'tri2b'}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        if self.stage == KaldiStage.TRI2B:
            return self.score_model(
                egs_path,
                local_exp_path / "tri2b",
                self.test_set,
                local_data_path,
            )
        # tri3b
        tri3b_ali_path = str(local_exp_path / "tri3b") + "_ali"
        run_command(
            f"steps/align_fmllr.sh --nj {cpus} --cmd run.pl --use-graphs true \
                {train_set} {local_data_path / 'lang_nosp'} {local_exp_path / 'tri2b'} {tri3b_ali_path}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        run_command(
            f"steps/train_sat.sh --cmd run.pl 2500 15000 {train_set} \
                {local_data_path / 'lang_nosp'} {tri3b_ali_path} {local_exp_path / 'tri3b'}",
            directory=egs_path,
            suppress_output=not self.verbose,
        )
        if self.stage == KaldiStage.TRI3B:
            return self.score_model(
                egs_path,
                local_exp_path / "tri3b",
                self.test_set,
                local_data_path,
            )
