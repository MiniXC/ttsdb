from pathlib import Path
from dataclasses import dataclass, fields
import subprocess
import multiprocessing
from typing import List
import os
from multiprocessing import cpu_count
import re
import shutil
from time import sleep
import sys
import selectors
from glob import glob
from shutil import copy
import pandas as pd
import numpy as np
import builtins

cpus = 8  # cpu_count()

KALDI_PATH = os.environ.get("KALDI_PATH", "/kaldi")


@dataclass
class Args:
    kaldi_path: str = f"{KALDI_PATH}/egs/librispeech/s5"
    data_name: str = "LibriSpeech"
    lm_path: str = "{data}/local/lm"
    dict_path: str = "{data}/local/dict"
    dict_nosp_path: str = "{data}/local/dict_nosp"
    lang_path: str = "{data}/lang"
    lang_tmp_path: str = "{data}/local/lang_tmp"
    lang_nosp_path: str = "{data}/lang_nosp"
    lang_nosp_tmp_path: str = "{data}/local/lang_tmp_nosp"
    lm_url: str = "www.openslr.org/resources/11"
    log_file: str = "train.log"
    local_data_path: str = "kaldi_data"
    local_exp_path: str = "exp"
    mfcc_path: str = "{data}/mfcc"
    train_cmd: str = "run.pl"
    lm_names: str = "tgsmall,tgmed"  # tglarge,fglarge
    mfcc_path: str = "{exp}/make_mfcc"
    mono_subset: int = 2000
    mono_path: str = "{exp}/mono"
    tri1_subset: int = 4000
    tri1_path: str = "{exp}/tri1"
    tri2b_path: str = "{exp}/tri2b"
    tri3b_path: str = "{exp}/tri3b"
    log_stages: str = "all"
    tdnn_path: str = "{exp}/tdnn"
    verbose: bool = True
    clean_stages: str = "none"
    use_cmvn: bool = False
    use_cnn: bool = False


args = Args()


def print(*pargs, **kwargs):
    if args.verbose:
        builtins.print(*pargs, **kwargs)
    else:
        # log to "kaldi_data/kaldi.log"
        with open(args.local_data_path / "kaldi.log", "a") as f:
            builtins.print(*pargs, **kwargs, file=f)


class Tasks:
    def __init__(self, logfile, kaldi_path):
        self.logfile = logfile
        self.kaldi_path = kaldi_path

    def execute(self, command, **kwargs):
        p = subprocess.Popen(
            f"{command}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )

        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        break_loop = False

        while not break_loop:
            for key, _ in sel.select():
                data = key.fileobj.read1().decode()
                if not data:
                    break_loop = True
                    break
                if key.fileobj is p.stdout:
                    yield data
                else:
                    yield data

        p.stdout.close()
        return_code = p.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def run(
        self,
        command,
        check_path=None,
        desc=None,
        run_in_kaldi=True,
        run_in=None,
        clean=False,
    ):
        if check_path is not None:
            if isinstance(check_path, list):
                run_command = any([not Path(p).exists() for p in check_path])
            else:
                run_command = not Path(check_path).exists()
        else:
            run_command = True
        if clean:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for c_p in check_path:
                c_p = Path(c_p)
                if c_p.is_dir():
                    shutil.rmtree(c_p)
                if c_p.is_file():
                    os.remove(c_p)
            run_command = True
        if run_command:
            print(command)
            if run_in_kaldi:
                for path in self.execute(command, cwd=self.kaldi_path):
                    if args.verbose:
                        print(path, end="")
            elif run_in is None:
                for path in self.execute(command):
                    if args.verbose:
                        print(path, end="")
            else:
                for path in self.execute(command, cwd=run_in):
                    if args.verbose:
                        print(path, end="")
            print(f"[green]✓[/green] {desc}")
        else:
            if not isinstance(check_path, list):
                check_path = [check_path]
            for p in check_path:
                print(f"[blue]{p} already exists[blue]")


def score_model(task, args, path, name, test_set, fmllr=False, lang_nosp=True):
    if args.log_stages == "all" or name in args.log_stages.split(","):
        mkgraph_args = ""
        if lang_nosp:
            graph_path = path / "graph_tgsmall"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_nosp_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        else:
            graph_path = path / "graph_tgsmall_sp"
            task.run(
                f"utils/mkgraph.sh {mkgraph_args} {str(args.lang_path) + '_test_tgsmall'} {path} {graph_path}",
                graph_path,
            )
        tst = test_set
        graph_test_path = str(graph_path) + f"_{tst.name}"
        tst_path = test_set
        # tst_path = args.local_data_path / (tst + "_med")
        if fmllr:
            p_decode = "_fmllr"
        else:
            p_decode = ""
        task.run(
            f"steps/decode{p_decode}.sh --lattice-beam 2.0 --nj {cpus} --cmd {args.train_cmd} {graph_path} {tst_path} {graph_test_path}",
            graph_test_path,
        )
        scoring_path = Path(graph_test_path) / "scoring_kaldi"
        task.run(
            f"steps/scoring/score_kaldi_wer.sh {tst_path} {graph_path} {graph_test_path}",
            scoring_path,
        )
        with open(scoring_path / "best_wer", "r") as best_wer:
            best_wer = best_wer.read()
        wer = float(re.findall(r"WER (\d+\.\d+)", best_wer)[0])
        try:
            ins_err = int(re.findall(r"(\d+) ins", best_wer)[0])
        except Exception:
            ins_err = int(re.findall(r"(\d+)\s+in", best_wer)[0])
        try:
            del_err = int(re.findall(r"(\d+) del", best_wer)[0])
        except Exception:
            del_err = int(re.findall(r"(\d+)\s+de", best_wer)[0])
        try:
            sub_err = int(re.findall(r"(\d+) sub", best_wer)[0])
        except Exception:
            sub_err = int(re.findall(r"(\d+)\s+ub", best_wer)[0])
        with open(scoring_path / "wer_details" / "wer_bootci", "r") as bootci:
            bootci = bootci.read()
        lower_wer, upper_wer = [
            round(float(c), 2)
            for c in re.findall(r"Conf Interval \[ (\d+\.\d+), (\d+\.\d+) \]", bootci)[
                0
            ]
        ]
        ci = round((upper_wer - lower_wer) / 2, 2)
        per_utt_info = (
            (Path(scoring_path) / "wer_details" / "per_utt").read_text().split("\n")
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
        print(per_utt_wer)
        return per_utt_wer


def get_kaldi_wer(args, train_ds, test_ds, cache_dir="cache"):
    if Path(f"{cache_dir}/{train_ds.name}_{test_ds.name}.npy").exists():
        return np.load(f"{cache_dir}/{train_ds.name}_{test_ds.name}.npy")
    task = Tasks(args.log_file, args.kaldi_path)
    args.lm_names = args.lm_names.split(",")
    if "," in args.clean_stages:
        args.clean_stages = args.clean_stages.split(",")
    else:
        args.clean_stages = [args.clean_stages]

    for field in fields(args):
        k, v = field.name, getattr(args, field.name)
        if "path" in field.name:
            v = str(v)
            if "{data}" in v:
                v = v.replace("{data}", str(args.local_data_path))
            if "{exp}" in v:
                v = v.replace("{exp}", str(args.local_exp_path))
            setattr(args, field.name, Path(v).resolve())

    # download lm
    print(f"local/download_lm.sh {args.lm_url} {args.lm_path}")
    task.run(f"local/download_lm.sh {args.lm_url} {args.lm_path}", args.lm_path)

    # create lms
    task.run(
        f'local/prepare_dict.sh --stage 3 --nj {cpus} --cmd "{args.train_cmd}" {args.lm_path} {args.lm_path} {args.dict_nosp_path}',
        args.dict_nosp_path,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_nosp_path} "<UNK>" {args.lang_nosp_tmp_path} {args.lang_nosp_path}',
        args.lang_nosp_tmp_path,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_nosp_path} {args.lm_path}",
        [
            str(args.lang_nosp_path) + "_test_tgsmall",
            str(args.lang_nosp_path) + "_test_tgmed",
        ],
    )
    if "tgsmall" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgsmall'}", run_in_kaldi=False
        )
    if "tgmed" not in args.lm_names:
        task.run(
            f"rm -r {str(args.lang_nosp_path) + '_test_tgmed'}", run_in_kaldi=False
        )
    if "tglarge" in args.lm_names:
        tglarge_path = str(args.lang_nosp_path) + "_test_tglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_tglarge.arpa.gz {args.lang_nosp_path} {tglarge_path}",
            tglarge_path,
        )
    if "fglarge" in args.lm_names:
        fglarge_path = str(args.lang_nosp_path) + "_test_fglarge"
        task.run(
            f"utils/build_const_arpa_lm.sh {args.lm_path}/lm_fglarge.arpa.gz {args.lang_nosp_path} {fglarge_path}",
            fglarge_path,
        )

    train_path = train_ds

    # mono
    mono_data_path = str(train_path)
    clean_mono = "mono" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/train_mono.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {mono_data_path} {args.lang_nosp_path} {args.mono_path}",
        args.mono_path,
        clean=clean_mono,
    )
    score_model(task, args, args.mono_path, "mono", test_ds)

    # tri1
    tri1_data_path = str(train_path)
    clean_tri1 = "tri1" in args.clean_stages or "all" in args.clean_stages
    tri1_ali_path = str(args.tri1_path) + "_ali"
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {tri1_data_path} {args.lang_nosp_path} {args.mono_path} {tri1_ali_path}",
        tri1_ali_path,
        clean=clean_tri1,
    )
    task.run(
        f"steps/train_deltas.sh --boost-silence 1.25 --cmd {args.train_cmd} 2000 10000 {tri1_data_path} {args.lang_nosp_path} {tri1_ali_path} {args.tri1_path}",
        args.tri1_path,
        clean=clean_tri1,
    )
    score_model(task, args, args.tri1_path, "tri1", test_ds, True)

    # tri2b
    tri2b_ali_path = str(args.tri2b_path) + "_ali"
    clean_tri2b = "tri2b" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/align_si.sh --boost-silence 1.25 --nj {cpus} --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri1_path} {tri2b_ali_path}",
        tri2b_ali_path,
        clean=clean_tri2b,
    )

    task.run(
        f'steps/train_lda_mllt.sh --boost-silence 1.25 --cmd {args.train_cmd} --splice-opts "--left-context=3 --right-context=3" 2500 15000 {train_path} {args.lang_nosp_path} {tri2b_ali_path} {args.tri2b_path}',
        args.tri2b_path,
        clean=clean_tri2b,
    )
    score_model(task, args, args.tri2b_path, "tri2b", test_ds, True)

    # tri3b
    tri3b_ali_path = str(args.tri3b_path) + "_ali"
    clean_tri3b = "tri3b" in args.clean_stages or "all" in args.clean_stages
    task.run(
        f"steps/align_fmllr.sh --nj {cpus} --cmd {args.train_cmd} --use-graphs true {train_path} {args.lang_nosp_path} {args.tri2b_path} {tri3b_ali_path}",
        tri3b_ali_path,
        clean=clean_tri3b,
    )
    task.run(
        f"steps/train_sat.sh --cmd {args.train_cmd} 2500 15000 {train_path} {args.lang_nosp_path} {tri3b_ali_path} {args.tri3b_path}",
        args.tri3b_path,
        clean=clean_tri3b,
    )
    score_model(task, args, args.tri3b_path, "tri3b", test_ds, True)

    # recompute lm
    task.run(
        f"steps/get_prons.sh --cmd {args.train_cmd} {train_path} {args.lang_nosp_path} {args.tri3b_path}",
        [
            args.tri3b_path / "pron_counts_nowb.txt",
            args.tri3b_path / "sil_counts_nowb.txt",
            args.tri3b_path / "pron_bigram_counts_nowb.txt",
        ],
        clean=clean_tri3b,
    )
    task.run(
        f'utils/dict_dir_add_pronprobs.sh --max-normalize true {args.dict_nosp_path} {args.tri3b_path / "pron_counts_nowb.txt"} {args.tri3b_path / "sil_counts_nowb.txt"} {args.tri3b_path / "pron_bigram_counts_nowb.txt"} {args.dict_path}',
        args.dict_path,
        clean=clean_tri3b,
    )
    task.run(
        f'utils/prepare_lang.sh {args.dict_path} "<UNK>" {args.lang_tmp_path} {args.lang_path}',
        args.lang_path,
        clean=clean_tri3b,
    )
    task.run(
        f"local/format_lms.sh --src-dir {args.lang_path} {args.lm_path}",
        [
            str(args.lang_path) + "_test_tgsmall",
            str(args.lang_path) + "_test_tgmed",
        ],
        clean=clean_tri3b,
    )
    wer_dist = score_model(
        task, args, args.tri3b_path, "tri3b-probs", test_ds, True, False
    )

    np.save(f"{cache_dir}/{train_ds.name}_{test_ds.name}.npy", wer_dist)
    return wer_dist


def get_kaldi_wasserstein(ds):
    # rm "exp" folder
    shutil.rmtree("exp", ignore_errors=True)
    ref_wer = get_kaldi_wer(Args(), "reference.dev", "reference.test")
    shutil.rmtree("exp", ignore_errors=True)
    ds_wer = get_kaldi_wer(Args(), ds, "reference.test")
    ref_wer = np.sort(ref_wer)
    ds_wer = np.sort(ds_wer)
    wasserstein = np.abs(ref_wer - ds_wer).mean()
    wasserstein_worst = np.abs(ref_wer - np.ones_like(ds_wer)).mean()

    result = (1 - wasserstein / wasserstein_worst) * 100

    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)
    figure_path = figure_path / f"{ds}.png"
    import matplotlib.pyplot as plt

    # histogram
    plt.hist(ref_wer, bins=50, alpha=0.5, label="Reference")
    plt.hist(ds_wer, bins=50, alpha=0.5, label=ds)
    plt.legend(loc="upper right")
    plt.xlabel("WER")
    plt.ylabel("Frequency")
    plt.title(f"{ds} vs Reference")
    plt.savefig(figure_path)
    plt.close()

    return result
