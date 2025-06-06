import torch
import numpy as np
import argparse
import soundfile as sf
import museval
from museval.metrics import Framing
import norbert
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import scipy.signal
from asteroid.complex_nn import torch_complex_from_magphase
import os
import sys
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from datasets import SynthSODDataset, AaltoAnechoicOrchestralDataset, URMPDataset
from utils import load_model, prepare_parser_from_dict

def polyphony_metrics(outdir):
    df = pd.read_pickle(outdir + "/" + "results.pandas")
    targets = df["target"].unique()

    file = open(os.path.join(outdir, "polyphony_results.txt"), "w")
    
    # add polyphony column to df
    for track in df["track"].unique():
        for time in df["time"].unique():
            frame_df = df[(df['time']==time) & (df['track']==track) & (df['metric']=='SDR')]
            polyphony = frame_df["score"].count()
            if frame_df.empty: continue
            df.loc[frame_df.index,'polyphony'] = polyphony

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for polyphony in range(1, 16):
            # how many frames have this polyphony, counts frame for each instrument, as in if N instruments have the polyphony in the same frame, it will count N times
            n_frames = df[df["polyphony"]==polyphony]["score"].count()
            print(f"\n{polyphony=} {n_frames=}", file=file, flush=True)
            for target in targets:
                medians = df[(df['metric']=='SDR')&(df['target']==target)&(df['polyphony']==polyphony)]["score"].median()
                print(f"{target:17s}: {medians:.4f}", file=file, flush=True)

def inference_args(parser, remaining_args):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        "--softmask",
        dest="softmask",
        action="store_true",
        help=(
            "if enabled, will initialize separation with softmask."
            "otherwise, will use mixture phase with spectrogram"
        ),
    )

    inf_parser.add_argument("--niter", type=int, default=1, help="number of iterations for refining results.")
    inf_parser.add_argument("--alpha", type=float, default=1.0, help="exponent in case of softmask separation")
    inf_parser.add_argument("--samplerate", type=int, default=44100, help="model samplerate")
    inf_parser.add_argument("--residual-model", action="store_true", help="create a model for the residual")
    return inf_parser.parse_args()

if __name__ == "__main__":
    # Settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    parser.add_argument(
        "--outdir",
        type=str,
        default="./x-umx_outputs",
        help="Results path and where " "best_model.pth" " is stored if no --models provided",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=[],
        nargs="+",
        help="Results paths where " "best_model.pth" " are stored",
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        default=['aalto', 'urmp', 'ensembleset', 'synthsod_test'],
        nargs="+",
        help="Datasets to evaluate on. Options: aalto, urmp, ensembleset, synthsod_test, synthsod_train",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA inference")
    parser.add_argument(
        "--minus-one", action="store_true", default=False,
        help="evaluate minus one tracks instead of the separated instruments (untested, might contain bugs)"
    )
    parser.add_argument(
        "--minus-one-model", action="store_true", default=False,
        help="the model is generating the minus one trakcs instead of the separated sources (untested, might contain bugs)"
    )
    parser.add_argument(
        "--exclude-silences", action="store_true", default=False,
        help="Exclude frames where the target is silent in the minus-one mode (untested, might contain bugs)"
    )
    parser.add_argument(
        "--group-families", action="store_true", default=False,
        help="evaluate the separated instruments grouped by families (untested, might contain bugs)"
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)
    assert not (args.exclude_silences and not args.minus_one), "--exclude-silences should only be used with --minus-one"

    targets = None
    if 'aalto' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_Aalto" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the Aalto anechoic orchestra recordings.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        polyphony_metrics(outdir)

    if 'urmp' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_URMP" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the URMP recordings.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        polyphony_metrics(outdir)

    if 'synthsod_test' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_SynthSOD_test" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the test partition of the SynthSOD Dataset.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        polyphony_metrics(outdir)

    if 'synthsod_train' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_SynthSOD_train" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the test partition of the SynthSOD Dataset.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        polyphony_metrics(outdir)

    if 'scoresynthsod_test' in args.eval_on:
        outdir = os.path.join(os.path.abspath(args.outdir), "EvaluateResults_ScoreSynthSOD_test" +
                              ("_group_families" if args.group_families else "") + ("_minus_one" if args.minus_one else ""))
        Path(outdir).mkdir(exist_ok=True, parents=True)
        print("Evaluating models on the test partition of the SynthSOD Dataset.", file=sys.stderr)
        print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)
        polyphony_metrics(outdir)
