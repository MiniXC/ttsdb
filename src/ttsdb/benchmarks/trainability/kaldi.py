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


def dataset_to_kaldi(dataset: Dataset, egs_path: Path, data_path: Path) -> Path:
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
            wav = librosa.resample(wav, orig_sr=dataset.sample_rate, target_sr=16000)
            sf.write(wav_path, wav, 16000)
        if not Path(text_path).exists():
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
        data_dict["speaker"].append(speaker)
        data_dict["id"].append(f"{speaker}-{idx}")
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
        f"steps/make_mfcc.sh --cmd run.pl --nj {cpus} {dest} {dest_mfcc} mfccs",
        directory=egs_path,
    )
    run_command(
        f"steps/compute_cmvn_stats.sh {dest} {dest_mfcc} mfccs",
        directory=egs_path,
    )
    return dest.resolve()


#     # create lms
#     task.run(
#         f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
#         args.dict_nosp_path,
#     )
#     task.run(
#         f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
#         args.lang_nosp_tmp_path,
#     )
#     task.run(
#         f"local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}",
#         [
#             str(args.lang_nosp_path) + "_test_tgsmall",
#             str(args.lang_nosp_path) + "_test_tgmed",
#         ],
#     )
#     if "tgsmall" not in args.lm_names:
#         task.run(
#             f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False
#         )
#     if "tgmed" not in args.lm_names:
#         task.run(
#             f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False
#         )
#     if "tglarge" in args.lm_names:
#         tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
#         task.run(
#             f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}",
#             tglarge_path,
#         )
#     if "fglarge" in args.lm_names:
#         fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
#         task.run(
#             f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}",
#             fglarge_path,
#         )

#     # mfccs
#     for tst in (train_ds, test_ds):
#         tst_path = args.local_data_path / "datasets" / tst
#         if not Path(tst_path).exists() or not Path(tst_path / "feats.scp").exists():
#             # prepare using generate_data
#             audios, texts, speakers = generate_data(tst)
#             dest = Path(args.local_data_path) / "datasets" / tst
#             dest.mkdir(parents=True, exist_ok=True)
#             data_dict = {"speaker": [], "id": [], "wav": [], "text": []}
#             for audio, text, speaker in zip(audios, texts, speakers):
#                 data_dict["speaker"].append(speaker)
#                 data_dict["id"].append(audio.replace(".wav", "").replace("_", "-"))
#                 data_dict["wav"].append(str(audio))
#                 data_dict["text"].append(text)
#             df = pd.DataFrame(data_dict)
#             # spk2utt
#             with open(Path(dest) / "spk2utt", "w") as spk2utt:
#                 for spk in sorted(df["speaker"].unique()):
#                     utts = df[df["speaker"] == spk]["id"].unique()
#                     utts = sorted([f"{utt}" for utt in utts])
#                     spk2utt.write(f'{spk} {" ".join(utts)}\n')
#             # text
#             with open(Path(dest) / "text", "w") as text_file:
#                 for utt in sorted(df["id"].unique()):
#                     text = df[df["id"] == utt]["text"].values[0]
#                     text_file.write(f"{utt} {text}\n")
#             # utt2spk
#             with open(Path(dest) / "utt2spk", "w") as utt2spk:
#                 for spk in sorted(df["speaker"].unique()):
#                     utts = df[df["speaker"] == spk]["id"].unique()
#                     for utt in sorted(utts):
#                         utt2spk.write(f"{utt} {spk}\n")
#             # wav.scp
#             with open(Path(dest) / "wav.scp", "w") as wavscp:
#                 for utt in sorted(df["id"].unique()):
#                     wav = df[df["id"] == utt]["wav"].values[0]
#                     wav = Path(wav).resolve()
#                     wavscp.write(
#                         f"{utt} sox {wav} -t wav -c 1 -b 16 -t wav - rate 16000 |\n"
#                     )
#         exp_path = args.mfcc_path / tst
#         task.run(
#             f"steps/make_mfcc.sh --cmd {args.train_cmd} --nj {cpus} {tst_path} {exp_path} mfccs",
#             tst_path / "feats.scp",
#         )
#         task.run(
#             f"steps/compute_cmvn_stats.sh {tst_path} {exp_path} mfccs",
#             exp_path / f"cmvn_{tst}.log",
#         )

#     train_path = args.local_data_path / "datasets" / train_ds

#     # mono
#     mono_data_path = str(train_path)
#     clean_mono = "mono" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}",
#         args.mono_path,
#         clean=clean_mono,
#     )
#     score_model(task, args, args.mono_path, "mono", test_ds)

#     # tri1
#     tri1_data_path = str(train_path)
#     clean_tri1 = "tri1" in args.clean_stages or "all" in args.clean_stages
#     tri1_ali_path = str(args.tri1_path) + "_ali"
#     task.run(
#         f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}",
#         tri1_ali_path,
#         clean=clean_tri1,
#     )
#     task.run(
#         f"steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}",
#         args.tri1_path,
#         clean=clean_tri1,
#     )
#     score_model(task, args, args.tri1_path, "tri1", test_ds, True)

#     # tri2b
#     tri2b_ali_path = str(args.tri2b_path) + "_ali"
#     clean_tri2b = "tri2b" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}",
#         tri2b_ali_path,
#         clean=clean_tri2b,
#     )

#     task.run(
#         f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
#         args.tri2b_path,
#         clean=clean_tri2b,
#     )
#     score_model(task, args, args.tri2b_path, "tri2b", test_ds, True)

#     # tri3b
#     tri3b_ali_path = str(args.tri3b_path) + "_ali"
#     clean_tri3b = "tri3b" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}",
#         tri3b_ali_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f"steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}",
#         args.tri3b_path,
#         clean=clean_tri3b,
#     )
#     score_model(task, args, args.tri3b_path, "tri3b", test_ds, True)

#     # recompute lm
#     task.run(
#         f"steps/get_prons.sh --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri3b_path}",
#         [
#             args.tri3b_path / "pron_counts_nowb.txt",
#             args.tri3b_path / "sil_counts_nowb.txt",
#             args.tri3b_path / "pron_bigram_counts_nowb.txt",
#         ],
#         clean=clean_tri3b,
#     )
#     task.run(
#         f'utils/dict_dir_add_pronprobs.sh --max-normalize true {args.dict_nosp_path} {args.tri3b_path / "pron_counts_nowb.txt"} {args.tri3b_path / "sil_counts_nowb.txt"} {args.tri3b_path / "pron_bigram_counts_nowb.txt"} {args.dict_path}',
#         args.dict_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f'utils/prepare_lang.sh {args.dict_path} "<UNK>" {args.lang_tmp_path} {args.lang_path}',
#         args.lang_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f"local/format_lms.sh --src-dir {args.lang_path} {args.lm_path}",
#         [
#             str(args.lang_path) + "_test_tgsmall",
#             str(args.lang_path) + "_test_tgmed",
#         ],
#         clean=clean_tri3b,
#     )
#     wer_dist = score_model(
#         task, args, args.tri3b_path, "tri3b-probs", test_ds, True, False
#     )

#     np.save(f"cache/{train_ds}_{test_ds}.npy", wer_dist)
#     return wer_dist

# def score_model(task, args, path, name, test_set, fmllr=False, lang_nosp=True):
#     if args.log_stages == "all" or name in args.log_stages.split(","):
#         print(
#             {
#                 "model": name,
#             }
#         )
#         mkgraph_args = ""
#         if lang_nosp:
#             graph_path = path / "graph_tgsmall"
#             task.run(
#                 f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
#                 graph_path,
#             )
#         else:
#             graph_path = path / "graph_tgsmall_sp"
#             task.run(
#                 f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_path) + '_test_tgsmall'} {path} {graph_path}",
#                 graph_path,
#             )
#         tst = test_set
#         graph_test_path = str(graph_path) + f"_{tst}"
#         tst_path = args.local_data_path / "datasets" / tst
#         # tst_path = args.local_data_path / (tst + "_med")
#         if fmllr:
#             p_decode = "_fmllr"
#         else:
#             p_decode = ""
#         task.run(
#             f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
#             graph_test_path,
#         )
#         scoring_path = Path(graph_test_path) / "scoring_kaldi"
#         task.run(
#             f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
#             scoring_path,
#         )
#         with open(scoring_path / "best_wer", "r") as best_wer:
#             best_wer = best_wer.read()
#         wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
#         ins_err = int(re.findall(r"(\d+) ins", best_wer)[0])
#         del_err = int(re.findall(r"(\d+) del", best_wer)[0])
#         sub_err = int(re.findall(r"(\d+) sub", best_wer)[0])
#         with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
#             bootci = bootci.read()
#         lower_wer, upper_wer = [
#             round(float(c), 2)
#             for c in re.findall(r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci)[
#                 0
#             ]
#         ]
#         ci = round((upper_wer - lower_wer) / 2, 2)
#         print(
#             {
#                 f"{tst}/wer": wer,
#                 f"{tst}/wer_lower": lower_wer,
#                 f"{tst}/wer_upper": upper_wer,
#                 f"{tst}/ci_width": ci,
#                 f"{tst}/ins": ins_err,
#                 f"{tst}/del": del_err,
#                 f"{tst}/sub": sub_err,
#             }
#         )
#         per_utt_info = (
#             (Path(scoring_path) / "wer_details" / "per_utt").read_text().split("\n")
#         )
#         per_utt_counts = [p.split(" ref")[-1] for p in per_utt_info if " ref" in p]
#         per_utt_counts = [
#             float(len([x for x in p.strip().split() if "***" not in x]))
#             for p in per_utt_counts
#         ]
#         per_utt_sid = [
#             p.split("#csid")[-1].strip().split() for p in per_utt_info if "#csid" in p
#         ]
#         per_utt_sid = [float(p[1]) + float(p[2]) + float(p[3]) for p in per_utt_sid]
#         per_utt_wer = [x / y for x, y in zip(per_utt_sid, per_utt_counts)]
#         return per_utt_wer


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
        stage: KaldiStage = KaldiStage.MONO,
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
            install_kaldi(kaldi_path)
        self.test_set = dataset_to_kaldi(test_set, self.egs_path, self.data_path)
        self.stage = stage

    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        egs_path = self.egs_path
        local_data_path = self.data_path
        train_set = dataset_to_kaldi(dataset, egs_path, local_data_path)
        local_exp_path = CACHE_DIR / "kaldi_exp" / str(hash(self) + hash(dataset))
        # rm exp path
        if local_exp_path.exists():
            run_command(f"rm -r {local_exp_path}")
        # validate data for both train and test
        run_command(
            f"utils/validate_data_dir.sh {train_set}",
            directory=egs_path,
        )
        run_command(
            f"utils/validate_data_dir.sh {self.test_set}",
            directory=egs_path,
        )
        lm_path = local_data_path / "local" / "lm"
        run_command(
            f"local/download_lm.sh www.openslr.org/resources/11 {lm_path}",
            directory=egs_path,
        )
        run_command(
            f"local/prepare_dict.sh --stage 3 --nj {cpus} --cmd run.pl \
                {lm_path} {lm_path} {local_data_path / 'local' / 'dict_nosp'}",
            directory=egs_path,
        )
        run_command(
            f"utils/prepare_lang.sh {local_data_path / 'local' / 'dict_nosp'} '<UNK>' \
                {local_data_path / 'local' / 'lang_tmp'} {local_data_path / 'lang_nosp'}",
            directory=egs_path,
        )
        if not (local_data_path / "lang_nosp_test_tgsmall").exists():
            run_command(
                f"local/format_lms.sh --src-dir {local_data_path / 'lang_nosp'} {lm_path}",
                directory=egs_path,
            )
        # always remove tgmed
        if (local_data_path / "lang_nosp_test_tgmed").exists():
            run_command(
                f"rm -r {local_data_path / 'lang_nosp_test_tgmed'}", directory=egs_path
            )
        # mono
        mono_data_path = str(train_set)
        run_command(
            f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {mono_data_path} {local_data_path / 'lang_nosp'} {local_exp_path / 'mono'}",
            directory=egs_path,
        )
        # if self.stage == KaldiStage.MONO:
        score_model(
            egs_path,
            local_exp_path / "mono",
            self.test_set,
            local_exp_path,
            local_data_path,
        )
        # tri1
        tri1_data_path = str(train_set)
        tri1_ali_path = str(local_exp_path / "tri1") + "_ali"
        run_command(
            f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {tri1_data_path} {local_data_path / 'lang_nosp'} {local_exp_path / 'mono'} {tri1_ali_path}",
            directory=egs_path,
        )
        run_command(
            f"steps/train_deltas.sh --boost-silence 1.25 --cmd run.pl 2000 10000 \
                {tri1_data_path} {local_data_path / 'lang_nosp'} {tri1_ali_path} {local_exp_path / 'tri1'}",
            directory=egs_path,
        )
        # if self.stage == KaldiStage.TRI1:
        score_model(
            egs_path,
            local_exp_path / "tri1",
            self.test_set,
            local_exp_path,
            local_data_path,
        )
        # tri2b
        tri2b_ali_path = str(local_exp_path / "tri2b") + "_ali"
        run_command(
            f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd run.pl \
                {train_set} {local_data_path / 'lang_nosp'} {local_exp_path / 'tri1'} {tri2b_ali_path}",
            directory=egs_path,
        )
        run_command(
            f"steps/train_lda_mllt.sh --boost-silence 1.25 --cmd run.pl --splice-opts '--left-context=3 --right-context=3' 2500 15000 \
                {train_set} {local_data_path / 'lang_nosp'} {tri2b_ali_path} {local_exp_path / 'tri2b'}",
            directory=egs_path,
        )
        # if self.stage == KaldiStage.TRI2B:
        score_model(
            egs_path,
            local_exp_path / "tri2b",
            self.test_set,
            local_exp_path,
            local_data_path,
        )
        # tri3b
        tri3b_ali_path = str(local_exp_path / "tri3b") + "_ali"
        run_command(
            f"steps/align_fmllr.sh --nj {cpus} --cmd run.pl --use-graphs true \
                {train_set} {local_data_path / 'lang_nosp'} {local_exp_path / 'tri2b'} {tri3b_ali_path}",
            directory=egs_path,
        )
        run_command(
            f"steps/train_sat.sh --cmd run.pl 2500 15000 {train_set} \
                {local_data_path / 'lang_nosp'} {tri3b_ali_path} {local_exp_path / 'tri3b'}",
            directory=egs_path,
        )
        # if self.stage == KaldiStage.TRI3B:
        score_model(
            egs_path,
            local_exp_path / "tri3b",
            self.test_set,
            local_exp_path,
            local_data_path,
        )


kb = KaldiBenchmark(stage=KaldiStage.TRI3B)

with importlib.resources.path("ttsdb", "data") as data_path:
    dev_ds = TarDataset(data_path / "libritts_dev.tar.gz").sample(100)

kb._get_distribution(dev_ds)

# @dataclass
# class Args:
#     kaldi_path: str = f"{KALDI_PATH}/egs/librispeech/s5"
#     data_name: str = "LibriSpeech"
#     lm_path: str = "{data}/local/lm"
#     dict_path: str = "{data}/local/dict"
#     dict_nosp_path: str = "{data}/local/dict_nosp"
#     lang_path: str = "{data}/lang"
#     lang_tmp_path: str = "{data}/local/lang_tmp"
#     lang_nosp_path: str = "{data}/lang_nosp"
#     lang_nosp_tmp_path: str = "{data}/local/lang_tmp_nosp"
#     lm_url: str = "www.openslr.org/resources/11"
#     log_file: str = "train.log"
#     local_data_path: str = "kaldi_data"
#     local_exp_path: str = "exp"
#     mfcc_path: str = "{data}/mfcc"
#     train_cmd: str = "run.pl"
#     lm_names: str = "tgsmall,tgmed"  # tglarge,fglarge
#     mfcc_path: str = "{exp}/make_mfcc"
#     mono_subset: int = 2000
#     mono_path: str = "{exp}/mono"
#     tri1_subset: int = 4000
#     tri1_path: str = "{exp}/tri1"
#     tri2b_path: str = "{exp}/tri2b"
#     tri3b_path: str = "{exp}/tri3b"
#     log_stages: str = "all"
#     tdnn_path: str = "{exp}/tdnn"
#     verbose: bool = True
#     clean_stages: str = "none"
#     use_cmvn: bool = False
#     use_cnn: bool = False


# args = Args()


# def print(*pargs, **kwargs):
#     if args.verbose:
#         builtins.print(*pargs, **kwargs)
#     else:
#         # log to "kaldi_data/kaldi.log"
#         with open(args.local_data_path / "kaldi.log", "a", encoding="utf-8") as f:
#             builtins.print(*pargs, **kwargs, file=f)


# cache_dir = Path("cache")
# cache_dir.mkdir(exist_ok=True)


# class Tasks:
#     def __init__(self, logfile, kaldi_path):
#         self.logfile = logfile
#         self.kaldi_path = kaldi_path

#     def execute(self, command, **kwargs):
#         p = subprocess.Popen(
#             f"{command}",
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             **kwargs,
#         )

#         sel = selectors.DefaultSelector()
#         sel.register(p.stdout, selectors.EVENT_READ)
#         sel.register(p.stderr, selectors.EVENT_READ)

#         break_loop = False

#         while not break_loop:
#             for key, _ in sel.select():
#                 data = key.fileobj.read1().decode()
#                 if not data:
#                     break_loop = True
#                     break
#                 if key.fileobj is p.stdout:
#                     yield data
#                 else:
#                     yield data

#         p.stdout.close()
#         return_code = p.wait()
#         if return_code:
#             raise subprocess.CalledProcessError(return_code, command)

#     def run(
#         self,
#         command,
#         check_path=None,
#         desc=None,
#         run_in_kaldi=True,
#         run_in=None,
#         clean=False,
#     ):
#         if check_path is not None:
#             if isinstance(check_path, list):
#                 run_command = any([not Path(p).exists() for p in check_path])
#             else:
#                 run_command = not Path(check_path).exists()
#         else:
#             run_command = True
#         if clean:
#             if not isinstance(check_path, list):
#                 check_path = [check_path]
#             for c_p in check_path:
#                 c_p = Path(c_p)
#                 if c_p.is_dir():
#                     shutil.rmtree(c_p)
#                 if c_p.is_file():
#                     os.remove(c_p)
#             run_command = True
#         if run_command:
#             print(command)
#             if run_in_kaldi:
#                 for path in self.execute(command, cwd=self.kaldi_path):
#                     if args.verbose:
#                         print(path, end="")
#             elif run_in is None:
#                 for path in self.execute(command):
#                     if args.verbose:
#                         print(path, end="")
#             else:
#                 for path in self.execute(command, cwd=run_in):
#                     if args.verbose:
#                         print(path, end="")
#             print(f"[green]✓[/green] {desc}")
#         else:
#             if not isinstance(check_path, list):
#                 check_path = [check_path]
#             for p in check_path:
#                 print(f"[blue]{p} already exists[blue]")


# def score_model(task, args, path, name, test_set, fmllr=False, lang_nosp=True):
#     if args.log_stages == "all" or name in args.log_stages.split(","):
#         print(
#             {
#                 "model": name,
#             }
#         )
#         mkgraph_args = ""
#         if lang_nosp:
#             graph_path = path / "graph_tgsmall"
#             task.run(
#                 f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
#                 graph_path,
#             )
#         else:
#             graph_path = path / "graph_tgsmall_sp"
#             task.run(
#                 f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_path) + '_test_tgsmall'} {path} {graph_path}",
#                 graph_path,
#             )
#         tst = test_set
#         graph_test_path = str(graph_path) + f"_{tst}"
#         tst_path = args.local_data_path / "datasets" / tst
#         # tst_path = args.local_data_path / (tst + "_med")
#         if fmllr:
#             p_decode = "_fmllr"
#         else:
#             p_decode = ""
#         task.run(
#             f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
#             graph_test_path,
#         )
#         scoring_path = Path(graph_test_path) / "scoring_kaldi"
#         task.run(
#             f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
#             scoring_path,
#         )
#         with open(scoring_path / "best_wer", "r") as best_wer:
#             best_wer = best_wer.read()
#         wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
#         ins_err = int(re.findall(r"(\d+) ins", best_wer)[0])
#         del_err = int(re.findall(r"(\d+) del", best_wer)[0])
#         sub_err = int(re.findall(r"(\d+) sub", best_wer)[0])
#         with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
#             bootci = bootci.read()
#         lower_wer, upper_wer = [
#             round(float(c), 2)
#             for c in re.findall(r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci)[
#                 0
#             ]
#         ]
#         ci = round((upper_wer - lower_wer) / 2, 2)
#         print(
#             {
#                 f"{tst}/wer": wer,
#                 f"{tst}/wer_lower": lower_wer,
#                 f"{tst}/wer_upper": upper_wer,
#                 f"{tst}/ci_width": ci,
#                 f"{tst}/ins": ins_err,
#                 f"{tst}/del": del_err,
#                 f"{tst}/sub": sub_err,
#             }
#         )
#         per_utt_info = (
#             (Path(scoring_path) / "wer_details" / "per_utt").read_text().split("\n")
#         )
#         per_utt_counts = [p.split(" ref")[-1] for p in per_utt_info if " ref" in p]
#         per_utt_counts = [
#             float(len([x for x in p.strip().split() if "***" not in x]))
#             for p in per_utt_counts
#         ]
#         per_utt_sid = [
#             p.split("#csid")[-1].strip().split() for p in per_utt_info if "#csid" in p
#         ]
#         per_utt_sid = [float(p[1]) + float(p[2]) + float(p[3]) for p in per_utt_sid]
#         per_utt_wer = [x / y for x, y in zip(per_utt_sid, per_utt_counts)]
#         return per_utt_wer


# def get_kaldi_wer(args, train_ds, test_ds):
#     if Path(f"cache/{train_ds}_{test_ds}.npy").exists():
#         return np.load(f"cache/{train_ds}_{test_ds}.npy")
#     task = Tasks(args.log_file, args.kaldi_path)
#     args.lm_names = args.lm_names.split(",")
#     if "," in args.clean_stages:
#         args.clean_stages = args.clean_stages.split(",")
#     else:
#         args.clean_stages = [args.clean_stages]

#     for field in fields(args):
#         k, v = field.name, getattr(args, field.name)
#         if "path" in field.name:
#             if "{data}" in v:
#                 v = v.replace("{data}", str(args.local_data_path))
#             if "{exp}" in v:
#                 v = v.replace("{exp}", str(args.local_exp_path))
#             setattr(args, field.name, Path(v).resolve())

#     # download lm
#     print(f"local/download_lm.sh {args.lm_url} {args.lm_path}")
#     task.run(f"local/download_lm.sh {args.lm_url} {args.lm_path}", args.lm_path)

#     # create lms
#     task.run(
#         f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
#         args.dict_nosp_path,
#     )
#     task.run(
#         f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
#         args.lang_nosp_tmp_path,
#     )
#     task.run(
#         f"local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}",
#         [
#             str(args.lang_nosp_path) + "_test_tgsmall",
#             str(args.lang_nosp_path) + "_test_tgmed",
#         ],
#     )
#     if "tgsmall" not in args.lm_names:
#         task.run(
#             f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False
#         )
#     if "tgmed" not in args.lm_names:
#         task.run(
#             f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False
#         )
#     if "tglarge" in args.lm_names:
#         tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
#         task.run(
#             f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}",
#             tglarge_path,
#         )
#     if "fglarge" in args.lm_names:
#         fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
#         task.run(
#             f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}",
#             fglarge_path,
#         )

#     # mfccs
#     for tst in (train_ds, test_ds):
#         tst_path = args.local_data_path / "datasets" / tst
#         if not Path(tst_path).exists() or not Path(tst_path / "feats.scp").exists():
#             # prepare using generate_data
#             audios, texts, speakers = generate_data(tst)
#             dest = Path(args.local_data_path) / "datasets" / tst
#             dest.mkdir(parents=True, exist_ok=True)
#             data_dict = {"speaker": [], "id": [], "wav": [], "text": []}
#             for audio, text, speaker in zip(audios, texts, speakers):
#                 data_dict["speaker"].append(speaker)
#                 data_dict["id"].append(audio.replace(".wav", "").replace("_", "-"))
#                 data_dict["wav"].append(str(audio))
#                 data_dict["text"].append(text)
#             df = pd.DataFrame(data_dict)
#             # spk2utt
#             with open(Path(dest) / "spk2utt", "w") as spk2utt:
#                 for spk in sorted(df["speaker"].unique()):
#                     utts = df[df["speaker"] == spk]["id"].unique()
#                     utts = sorted([f"{utt}" for utt in utts])
#                     spk2utt.write(f'{spk} {" ".join(utts)}\n')
#             # text
#             with open(Path(dest) / "text", "w") as text_file:
#                 for utt in sorted(df["id"].unique()):
#                     text = df[df["id"] == utt]["text"].values[0]
#                     text_file.write(f"{utt} {text}\n")
#             # utt2spk
#             with open(Path(dest) / "utt2spk", "w") as utt2spk:
#                 for spk in sorted(df["speaker"].unique()):
#                     utts = df[df["speaker"] == spk]["id"].unique()
#                     for utt in sorted(utts):
#                         utt2spk.write(f"{utt} {spk}\n")
#             # wav.scp
#             with open(Path(dest) / "wav.scp", "w") as wavscp:
#                 for utt in sorted(df["id"].unique()):
#                     wav = df[df["id"] == utt]["wav"].values[0]
#                     wav = Path(wav).resolve()
#                     wavscp.write(
#                         f"{utt} sox {wav} -t wav -c 1 -b 16 -t wav - rate 16000 |\n"
#                     )
#         exp_path = args.mfcc_path / tst
#         task.run(
#             f"steps/make_mfcc.sh --cmd {args.train_cmd} --nj {cpus} {tst_path} {exp_path} mfccs",
#             tst_path / "feats.scp",
#         )
#         task.run(
#             f"steps/compute_cmvn_stats.sh {tst_path} {exp_path} mfccs",
#             exp_path / f"cmvn_{tst}.log",
#         )

#     train_path = args.local_data_path / "datasets" / train_ds

#     # mono
#     mono_data_path = str(train_path)
#     clean_mono = "mono" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}",
#         args.mono_path,
#         clean=clean_mono,
#     )
#     score_model(task, args, args.mono_path, "mono", test_ds)

#     # tri1
#     tri1_data_path = str(train_path)
#     clean_tri1 = "tri1" in args.clean_stages or "all" in args.clean_stages
#     tri1_ali_path = str(args.tri1_path) + "_ali"
#     task.run(
#         f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}",
#         tri1_ali_path,
#         clean=clean_tri1,
#     )
#     task.run(
#         f"steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}",
#         args.tri1_path,
#         clean=clean_tri1,
#     )
#     score_model(task, args, args.tri1_path, "tri1", test_ds, True)

#     # tri2b
#     tri2b_ali_path = str(args.tri2b_path) + "_ali"
#     clean_tri2b = "tri2b" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}",
#         tri2b_ali_path,
#         clean=clean_tri2b,
#     )

#     task.run(
#         f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
#         args.tri2b_path,
#         clean=clean_tri2b,
#     )
#     score_model(task, args, args.tri2b_path, "tri2b", test_ds, True)

#     # tri3b
#     tri3b_ali_path = str(args.tri3b_path) + "_ali"
#     clean_tri3b = "tri3b" in args.clean_stages or "all" in args.clean_stages
#     task.run(
#         f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}",
#         tri3b_ali_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f"steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}",
#         args.tri3b_path,
#         clean=clean_tri3b,
#     )
#     score_model(task, args, args.tri3b_path, "tri3b", test_ds, True)

#     # recompute lm
#     task.run(
#         f"steps/get_prons.sh --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri3b_path}",
#         [
#             args.tri3b_path / "pron_counts_nowb.txt",
#             args.tri3b_path / "sil_counts_nowb.txt",
#             args.tri3b_path / "pron_bigram_counts_nowb.txt",
#         ],
#         clean=clean_tri3b,
#     )
#     task.run(
#         f'utils/dict_dir_add_pronprobs.sh --max-normalize true {args.dict_nosp_path} {args.tri3b_path / "pron_counts_nowb.txt"} {args.tri3b_path / "sil_counts_nowb.txt"} {args.tri3b_path / "pron_bigram_counts_nowb.txt"} {args.dict_path}',
#         args.dict_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f'utils/prepare_lang.sh {args.dict_path} "<UNK>" {args.lang_tmp_path} {args.lang_path}',
#         args.lang_path,
#         clean=clean_tri3b,
#     )
#     task.run(
#         f"local/format_lms.sh --src-dir {args.lang_path} {args.lm_path}",
#         [
#             str(args.lang_path) + "_test_tgsmall",
#             str(args.lang_path) + "_test_tgmed",
#         ],
#         clean=clean_tri3b,
#     )
#     wer_dist = score_model(
#         task, args, args.tri3b_path, "tri3b-probs", test_ds, True, False
#     )

#     np.save(f"cache/{train_ds}_{test_ds}.npy", wer_dist)
#     return wer_dist


# def get_kaldi_wasserstein(ds):
#     # rm "exp" folder
#     shutil.rmtree("exp", ignore_errors=True)
#     ref_wer = get_kaldi_wer(Args(), "reference.dev", "reference.test")
#     shutil.rmtree("exp", ignore_errors=True)
#     ds_wer = get_kaldi_wer(Args(), ds, "reference.test")
#     ref_wer = np.sort(ref_wer)
#     ds_wer = np.sort(ds_wer)
#     wasserstein = np.abs(ref_wer - ds_wer).mean()
#     wasserstein_worst = np.abs(ref_wer - np.ones_like(ds_wer)).mean()

#     result = (1 - wasserstein / wasserstein_worst) * 100

#     figure_path = Path("figures")
#     figure_path.mkdir(exist_ok=True)
#     figure_path = figure_path / f"{ds}.png"
#     import matplotlib.pyplot as plt

#     # histogram
#     plt.hist(ref_wer, bins=50, alpha=0.5, label="Reference")
#     plt.hist(ds_wer, bins=50, alpha=0.5, label=ds)
#     plt.legend(loc="upper right")
#     plt.xlabel("WER")
#     plt.ylabel("Frequency")
#     plt.title(f"{ds} vs Reference")
#     plt.savefig(figure_path)
#     plt.close()

#     return result
